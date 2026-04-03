"""
BudgetNet: Graph-Conditioned Adaptive Budget Controller
========================================================
Takes graph-level statistics and a pooled node embedding, then predicts
per-layer keep-ratios for tokens and per-layer survival gates for layers.

Architecture: Two-stream design ensures structural features (num_nodes,
num_edges, density, degree stats) have equal influence to the pooled
node embedding, preventing the embedding from drowning out the
graph-structure signal that drives adaptive budget decisions.
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
        - token_ratios: [B, num_layers]  keep-ratio in [min_ratio, 1.0]
        - layer_gates:  [B, num_layers]  survival probability in [0, 1]
    """

    def __init__(self, channels: int, num_layers: int,
                 hidden_dim: int = 64, min_token_ratio: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.min_ratio = min_token_ratio

        # --- Stream 1: Structural features (5 scalars) ---
        # Dedicated pathway so graph-structure signal can't be drowned out
        # Remove LayerNorm to preserve absolute scale (log N, log E magnitude matters)
        self.struct_norm = nn.Identity()
        self.struct_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --- Stream 2: Pooled node embedding ---
        # Compressed to prevent dominance over structural features
        self.emb_proj = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --- Combined processing ---
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.token_head = nn.Linear(hidden_dim, num_layers)
        self.layer_head = nn.Linear(hidden_dim, num_layers)

    # ------------------------------------------------------------------
    def forward(self, x, edge_index, batch, node_emb):
        device = x.device
        batch_size = int(batch.max().item()) + 1

        # --- graph-level scalar features ---
        ones_n = torch.ones(x.size(0), device=device)
        N = torch.zeros(batch_size, device=device).scatter_add_(0, batch, ones_n)

        edge_graph = batch[edge_index[0]]
        ones_e = torch.ones(edge_index.size(1), device=device)
        E = torch.zeros(batch_size, device=device).scatter_add_(0, edge_graph, ones_e) / 2.0

        log_N = torch.log(N + 1.0).unsqueeze(-1)
        log_E = torch.log(E + 1.0).unsqueeze(-1)
        density = (2.0 * E / (N * (N - 1.0) + 1e-8)).unsqueeze(-1)

        deg = degree(edge_index[0], num_nodes=x.size(0))            # [N_total]
        avg_deg = global_mean_pool(deg.unsqueeze(-1), batch)         # [B, 1]
        deg_sq   = global_mean_pool((deg ** 2).unsqueeze(-1), batch) # [B, 1]
        deg_var  = (deg_sq - avg_deg ** 2).clamp(min=0)              # [B, 1]

        struct_feats = torch.cat([log_N, log_E, density, avg_deg, deg_var], dim=-1)  # [B, 5]

        # --- Stream 1: Structural features with normalization ---
        struct_h = self.struct_mlp(self.struct_norm(struct_feats))    # [B, hidden_dim]

        # --- Stream 2: Pooled embedding (detached to force reliance on structure) ---
        pooled = global_mean_pool(node_emb.detach(), batch)          # [B, channels]
        emb_h = self.emb_proj(pooled)                                # [B, hidden_dim]

        # --- Additive combination (equal weighting of both streams) ---
        h = self.combine(struct_h + emb_h)

        # token ratios:  sigmoid -> [0,1]  then  affine -> [min_ratio, 1.0]
        raw_tok = torch.sigmoid(self.token_head(h))
        token_ratios = self.min_ratio + (1.0 - self.min_ratio) * raw_tok   # [B, L]

        # layer gates:  sigmoid  (used as soft gate or hard via Gumbel)
        layer_gates = torch.sigmoid(self.layer_head(h))                    # [B, L]

        return token_ratios, layer_gates
