# adaptive_model.py
"""
BudgetNet: Graph-Conditioned Adaptive Budget Controller
========================================================
Takes graph-level statistics and a pooled node embedding, then predicts
per-layer keep-ratios for tokens, per-layer survival gates for layers,
and per-layer per-head importance gates for attention heads.

Architecture: Two-stream design ensures structural features (num_nodes,
num_edges, density, degree stats) have equal influence to the pooled
node embedding, preventing the embedding from drowning out the
graph-structure signal that drives adaptive budget decisions.

Changes over gps_adaptive version:
  - BudgetNet now accepts `num_heads` parameter
  - Added `head_head` linear: hidden_dim → num_layers * num_heads
  - Forward now returns a third output:
      head_gates [B, num_layers, num_heads]  in [min_head_gate, 1.0]
  - Head gates use sigmoid gating (soft pruning / importance weighting)
    so all heads are still computed but less important ones are suppressed.
    This is fully differentiable and compatible with PyTorch's fused MHA.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree


class BudgetNet(nn.Module):
    """
    Predicts adaptive budgets for each GPS layer based on graph-level features.

    Inputs (computed from the raw graph):
        - x:            Node features              [N_total, C]
        - edge_index:   Edge connectivity           [2, E_total]
        - batch:        Graph membership vector     [N_total]
        - node_emb:     Embedded node features      [N_total, channels]

    Outputs:
        - token_ratios: [B, num_layers]             keep-ratio in [min_ratio, max_ratio]
        - layer_gates:  [B, num_layers]             survival probability in [0, 1]
        - head_gates:   [B, num_layers, num_heads]  head importance in [min_head_gate, 1.0]
    """

    def __init__(self,
                 channels: int,
                 num_layers: int,
                 num_heads: int,                        # ← NEW
                 hidden_dim: int = 64,
                 min_token_ratio: float = 0.2,
                 max_token_ratio: float = 1.0,
                 min_head_gate: float = 0.0,            # ← NEW  lower bound for head gates
                 size_prior_mix: float = 0.0,
                 size_prior_temp: float = 3.0):
        super().__init__()

        # ── Validate ────────────────────────────────────────────────────────────
        self.num_layers    = num_layers
        self.num_heads     = num_heads
        self.min_ratio     = min_token_ratio
        self.max_ratio     = max_token_ratio
        self.min_head_gate = min_head_gate

        if self.max_ratio <= self.min_ratio:
            raise ValueError("max_token_ratio must be greater than min_token_ratio")
        if not 0.0 <= size_prior_mix <= 1.0:
            raise ValueError("size_prior_mix must be in [0, 1]")
        if size_prior_temp <= 0.0:
            raise ValueError("size_prior_temp must be > 0")
        if not 0.0 <= min_head_gate < 1.0:
            raise ValueError("min_head_gate must be in [0, 1)")

        self.size_prior_mix  = float(size_prior_mix)
        self.size_prior_temp = float(size_prior_temp)

        # ── Stream 1: Structural features (5 scalars) ───────────────────────────
        # Remove LayerNorm to preserve absolute scale (log N, log E magnitude matters)
        self.struct_norm = nn.Identity()
        self.struct_mlp  = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Stream 2: Pooled node embedding ─────────────────────────────────────
        # Compressed to prevent dominance over structural features
        self.emb_proj = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Combined processing ──────────────────────────────────────────────────
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Output heads ────────────────────────────────────────────────────────
        self.token_head = nn.Linear(hidden_dim, num_layers)
        self.layer_head = nn.Linear(hidden_dim, num_layers)

        # NEW: predicts a gate for every (layer, head) pair
        # Output shape after reshape: [B, num_layers, num_heads]
        self.head_head  = nn.Linear(hidden_dim, num_layers * num_heads)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x, edge_index, batch, node_emb):
        device     = x.device
        batch_size = int(batch.max().item()) + 1

        # ── Graph-level scalar features ──────────────────────────────────────
        ones_n    = torch.ones(x.size(0), device=device)
        N         = torch.zeros(batch_size, device=device).scatter_add_(0, batch, ones_n)

        edge_graph = batch[edge_index[0]]
        ones_e     = torch.ones(edge_index.size(1), device=device)
        E          = torch.zeros(batch_size, device=device).scatter_add_(
                         0, edge_graph, ones_e) / 2.0

        log_N   = torch.log(N + 1.0).unsqueeze(-1)                           # [B, 1]
        log_E   = torch.log(E + 1.0).unsqueeze(-1)                           # [B, 1]
        density = (2.0 * E / (N * (N - 1.0) + 1e-8)).unsqueeze(-1)          # [B, 1]

        deg     = degree(edge_index[0], num_nodes=x.size(0))                 # [N_total]
        avg_deg = global_mean_pool(deg.unsqueeze(-1), batch)                 # [B, 1]
        deg_sq  = global_mean_pool((deg ** 2).unsqueeze(-1), batch)          # [B, 1]
        deg_var = (deg_sq - avg_deg ** 2).clamp(min=0)                       # [B, 1]

        struct_feats = torch.cat(
            [log_N, log_E, density, avg_deg, deg_var], dim=-1)               # [B, 5]

        # ── Stream 1: Structural MLP ──────────────────────────────────────────
        struct_h = self.struct_mlp(self.struct_norm(struct_feats))           # [B, hidden_dim]

        # ── Stream 2: Pooled embedding (detached → forces reliance on structure)
        pooled = global_mean_pool(node_emb.detach(), batch)                  # [B, channels]
        emb_h  = self.emb_proj(pooled)                                       # [B, hidden_dim]

        # ── Additive combination ─────────────────────────────────────────────
        h = self.combine(struct_h + emb_h)                                   # [B, hidden_dim]

        # ── Token ratios ─────────────────────────────────────────────────────
        raw_tok = torch.sigmoid(self.token_head(h))                          # [B, L]
        learned_token_ratios = (
            self.min_ratio + (self.max_ratio - self.min_ratio) * raw_tok)    # [B, L]

        if self.size_prior_mix > 0.0:
            # Monotonic size prior: smaller graphs keep more tokens
            size_std  = log_N.std(unbiased=False).clamp_min(1e-6)
            size_z    = (log_N - log_N.mean()) / size_std                    # [B, 1]
            size_prior = torch.sigmoid(-self.size_prior_temp * size_z)       # [B, 1]
            size_prior = size_prior.expand(-1, self.num_layers)              # [B, L]
            prior_token_ratios = (
                self.min_ratio + (self.max_ratio - self.min_ratio) * size_prior)
            token_ratios = ((1.0 - self.size_prior_mix) * learned_token_ratios
                            + self.size_prior_mix * prior_token_ratios)
        else:
            token_ratios = learned_token_ratios                              # [B, L]

        # ── Layer gates ──────────────────────────────────────────────────────
        layer_gates = torch.sigmoid(self.layer_head(h))                      # [B, L]

        # ── Head gates (NEW) ─────────────────────────────────────────────────
        # sigmoid → [0, 1] then affine → [min_head_gate, 1.0]
        # Shape: [B, L*H] → [B, L, H]
        raw_head  = torch.sigmoid(self.head_head(h))                         # [B, L*H]
        head_gates = (
            self.min_head_gate
            + (1.0 - self.min_head_gate) * raw_head
        ).view(-1, self.num_layers, self.num_heads)                          # [B, L, H]

        return token_ratios, layer_gates, head_gates