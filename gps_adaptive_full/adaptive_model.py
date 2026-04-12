"""
BudgetNet: Graph-Conditioned Adaptive Budget Controller
========================================================
Takes graph-level statistics and a pooled node embedding, then predicts
per-layer keep-ratios for tokens, per-layer survival gates for layers,
and per-layer per-head importance gates for attention heads.

FIX for constant head gate collapse:
  - Added head_prior_mix: injects a graph-size prior into head_gates
    Large graphs → lower head gate (suppress more heads)
    Small graphs → higher head gate (keep more heads)
  - This mirrors the size prior used for token ratios and forces
    per-graph variation right from the start of training.
  - head_mlp uses a dedicated 2-layer MLP (not shared with token/layer)
    to give BudgetNet enough capacity to predict per-head variation.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree


class BudgetNet(nn.Module):
    """
    Outputs:
        - token_ratios: [B, num_layers]
        - layer_gates:  [B, num_layers]
        - head_gates:   [B, num_layers, num_heads]
    """

    def __init__(self,
                 channels: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int = 64,
                 min_token_ratio: float = 0.2,
                 max_token_ratio: float = 1.0,
                 min_head_gate: float = 0.05,
                 size_prior_mix: float = 0.0,
                 size_prior_temp: float = 3.0,
                 head_prior_mix: float = 0.3,
                 head_prior_temp: float = 2.0):
        super().__init__()

        self.num_layers    = num_layers
        self.num_heads     = num_heads
        self.min_ratio     = min_token_ratio
        self.max_ratio     = max_token_ratio
        self.min_head_gate = min_head_gate

        if self.max_ratio <= self.min_ratio:
            raise ValueError("max_token_ratio must be greater than min_token_ratio")
        if not 0.0 <= size_prior_mix <= 1.0:
            raise ValueError("size_prior_mix must be in [0, 1]")
        if not 0.0 <= min_head_gate < 1.0:
            raise ValueError("min_head_gate must be in [0, 1)")
        if not 0.0 <= head_prior_mix <= 1.0:
            raise ValueError("head_prior_mix must be in [0, 1]")

        self.size_prior_mix  = float(size_prior_mix)
        self.size_prior_temp = float(size_prior_temp)
        self.head_prior_mix  = float(head_prior_mix)
        self.head_prior_temp = float(head_prior_temp)

        # ── Stream 1: Structural features ───────────────────────────────────────
        self.struct_norm = nn.Identity()
        self.struct_mlp  = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Stream 2: Pooled embedding ───────────────────────────────────────────
        self.emb_proj = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Combined ─────────────────────────────────────────────────────────────
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Token + Layer heads (shared hidden) ──────────────────────────────────
        self.token_head = nn.Linear(hidden_dim, num_layers)
        self.layer_head = nn.Linear(hidden_dim, num_layers)

        # ── Head gate MLP (dedicated, 2-layer for extra capacity) ────────────────
        # Separate from token/layer heads so gradients don't interfere
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_layers * num_heads),
        )

    def forward(self, x, edge_index, batch, node_emb):
        device     = x.device
        batch_size = int(batch.max().item()) + 1

        # ── Graph features ────────────────────────────────────────────────────
        ones_n = torch.ones(x.size(0), device=device)
        N      = torch.zeros(batch_size, device=device).scatter_add_(0, batch, ones_n)

        edge_graph = batch[edge_index[0]]
        ones_e     = torch.ones(edge_index.size(1), device=device)
        E          = torch.zeros(batch_size, device=device).scatter_add_(
                         0, edge_graph, ones_e) / 2.0

        log_N   = torch.log(N + 1.0).unsqueeze(-1)
        log_E   = torch.log(E + 1.0).unsqueeze(-1)
        density = (2.0 * E / (N * (N - 1.0) + 1e-8)).unsqueeze(-1)

        deg     = degree(edge_index[0], num_nodes=x.size(0))
        avg_deg = global_mean_pool(deg.unsqueeze(-1), batch)
        deg_sq  = global_mean_pool((deg ** 2).unsqueeze(-1), batch)
        deg_var = (deg_sq - avg_deg ** 2).clamp(min=0)

        struct_feats = torch.cat([log_N, log_E, density, avg_deg, deg_var], dim=-1)

        # ── Forward streams ───────────────────────────────────────────────────
        struct_h = self.struct_mlp(self.struct_norm(struct_feats))
        pooled   = global_mean_pool(node_emb.detach(), batch)
        emb_h    = self.emb_proj(pooled)
        h        = self.combine(struct_h + emb_h)

        # ── Token ratios ──────────────────────────────────────────────────────
        raw_tok              = torch.sigmoid(self.token_head(h))
        learned_token_ratios = self.min_ratio + (self.max_ratio - self.min_ratio) * raw_tok

        if self.size_prior_mix > 0.0:
            size_std   = log_N.std(unbiased=False).clamp_min(1e-6)
            size_z     = (log_N - log_N.mean()) / size_std
            size_prior = torch.sigmoid(-self.size_prior_temp * size_z).expand(
                -1, self.num_layers)
            prior_token_ratios = self.min_ratio + (self.max_ratio - self.min_ratio) * size_prior
            token_ratios = ((1.0 - self.size_prior_mix) * learned_token_ratios
                            + self.size_prior_mix * prior_token_ratios)
        else:
            token_ratios = learned_token_ratios

        # ── Layer gates ───────────────────────────────────────────────────────
        layer_gates = torch.sigmoid(self.layer_head(h))

        # ── Head gates (with size prior) ──────────────────────────────────────
        raw_head           = torch.sigmoid(self.head_mlp(h))
        learned_head_gates = (
            self.min_head_gate + (1.0 - self.min_head_gate) * raw_head
        ).view(batch_size, self.num_layers, self.num_heads)

        if self.head_prior_mix > 0.0:
            # Standardise log_N across graphs in batch
            if batch_size > 1:
                size_std_h = log_N.std(unbiased=False).clamp_min(1e-6)
                size_z_h   = (log_N - log_N.mean()) / size_std_h
            else:
                size_z_h = torch.zeros_like(log_N)

            # Large graph → lower gate value (suppress more heads)
            head_size_prior = torch.sigmoid(-self.head_prior_temp * size_z_h)
            head_size_prior = (
                self.min_head_gate + (1.0 - self.min_head_gate) * head_size_prior
            ).unsqueeze(-1).expand(batch_size, self.num_layers, self.num_heads)

            head_gates = ((1.0 - self.head_prior_mix) * learned_head_gates
                          + self.head_prior_mix * head_size_prior)
        else:
            head_gates = learned_head_gates

        return token_ratios, layer_gates, head_gates