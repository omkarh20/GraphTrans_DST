"""
Adaptive GPSConv — Token + Layer + Head
========================================
Modifications over gps_adaptive version:

1. **True token pruning**   – top-k tokens gathered before attention → O(k²)
2. **Adaptive layer gating** – soft/hard gate skips the entire layer
3. **Learnable node scorer** – small MLP scores node importance for top-k
4. **Adaptive head gating**  – NEW: per-graph, per-head importance weights
                               from BudgetNet scale each head's contribution
                               after MHA output, creating a differentiable
                               soft pruning signal without breaking PyTorch's
                               fused MultiheadAttention kernel.

Head gating mechanics
---------------------
PyTorch's nn.MultiheadAttention fuses all heads into a single kernel, so we
cannot physically drop individual heads per sample in a batch.  Instead we
use *soft gating*:

    attn_out  [B, k, C]  →  reshape  →  [B, k, H, d]
    head_gate [B, H]     →  scale each head dimension
    reshape back         →  [B, k, C]

This is fully differentiable:  task_loss → attn_out → head_gate → BudgetNet
Heads with low gate values contribute little to the residual, effectively
suppressing them.  BudgetNet learns which heads are useful per graph type.
"""

import inspect
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential, ReLU

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch


# -----------------------------------------------------------------------
# Gumbel-Softmax helpers  (unchanged from gps_adaptive)
# -----------------------------------------------------------------------
def _scatter_onehot(logits: Tensor, index: Tensor, k: int) -> Tensor:
    B = logits.size(0)
    x_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, k).reshape(-1)
    y_idx = index.reshape(-1)
    out = torch.zeros_like(logits)
    out[x_idx, y_idx] = 1.0
    return out


def gumbel_topk(logits: Tensor, k: int, tau: float = 1.0,
                hard: bool = True) -> Tensor:
    """Differentiable top-k via Gumbel-Softmax (straight-through in hard mode)."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim=-1)
    if hard:
        index  = y_soft.topk(k, dim=-1)[1]
        y_hard = _scatter_onehot(logits, index, k)
        return y_hard - y_soft.detach() + y_soft   # straight-through
    return y_soft


# -----------------------------------------------------------------------
# Adaptive GPSConv  (Token + Layer + Head)
# -----------------------------------------------------------------------
class GPSConv(torch.nn.Module):
    r"""GPS convolution with adaptive token pruning, layer gating, and head gating.

    New vs gps_adaptive:
      * ``head_gate`` parameter  [B, H]  – per-graph head importance weights
        produced by BudgetNet.  Applied after MHA by reshaping the output
        into [B, k, H, d_head], scaling, then reshaping back.
    """

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        max_k: int = 512,
    ):
        super().__init__()

        self.channels = channels
        self.conv     = conv
        self.heads    = heads
        self.dropout  = dropout
        self.max_k    = max_k

        # channels must be divisible by heads for reshape to work
        assert channels % heads == 0, \
            f"channels ({channels}) must be divisible by heads ({heads})"
        self.d_head = channels // heads

        self.attn = torch.nn.MultiheadAttention(
            channels, heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

        # Node importance scorer for token pruning
        self.scorer = Sequential(
            Linear(channels, channels),
            ReLU(),
            Linear(channels, 1),
        )

    # -----------------------------------------------------------------
    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        for norm in [self.norm1, self.norm2, self.norm3]:
            if norm is not None:
                norm.reset_parameters()

    # -----------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[Tensor] = None,
        token_ratio: Optional[Tensor] = None,  # [B]
        layer_gate:  Optional[Tensor] = None,  # [B]
        head_gate:   Optional[Tensor] = None,  # [B, H]  ← NEW
        tau: float = 1.0,
        **kwargs,
    ) -> Tensor:
        """
        Parameters
        ----------
        x           : [N_total, C]
        edge_index  : Adj
        batch       : [N_total]
        token_ratio : [B]   fraction of tokens to keep (0, 1]
        layer_gate  : [B]   layer survival probability  [0, 1]
        head_gate   : [B, H] per-head importance weights [min_gate, 1]  ← NEW
        tau         : Gumbel temperature for token selection
        """

        # ============================================================
        # 0.  Layer gate weight  (expand B → N_total for residual mix)
        # ============================================================
        gate_weight = (layer_gate[batch].unsqueeze(-1)   # [N_total, 1]
                       if layer_gate is not None else None)

        # ============================================================
        # 1.  Local MPNN
        # ============================================================
        hs = []
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                h = self.norm1(h, batch=batch) if self.norm_with_batch else self.norm1(h)
            hs.append(h)

        # ============================================================
        # 2.  Global attention with adaptive token + head gating
        # ============================================================
        h_dense, mask = to_dense_batch(x, batch)   # [B, L, C], [B, L]
        B, L, C = h_dense.shape

        if token_ratio is not None and L > 1:
            # ── Token selection ──────────────────────────────────────
            real_lengths = mask.sum(dim=1)                              # [B]
            k_per_graph  = (token_ratio * real_lengths.float()).ceil().long()
            k_per_graph  = k_per_graph.clamp(min=1, max=self.max_k)
            k_per_graph  = torch.where(k_per_graph > real_lengths,
                                       real_lengths, k_per_graph)
            k_max = int(k_per_graph.max().item())

            scores     = self.scorer(h_dense).squeeze(-1)               # [B, L]
            scores     = scores.masked_fill(~mask, float('-inf'))
            log_scores = F.log_softmax(scores, dim=-1)
            token_mask = gumbel_topk(log_scores, k=k_max, tau=tau, hard=True)

            # Enforce per-graph individual k budgets
            sorted_mask, sort_idx = token_mask.sort(dim=-1, descending=True)
            for i in range(B):
                ki = int(k_per_graph[i].item())
                if ki < k_max:
                    sorted_mask[i, ki:] = 0.0
            token_mask_refined = torch.zeros_like(sorted_mask)
            token_mask_refined.scatter_(1, sort_idx, sorted_mask)

            weighted     = h_dense * token_mask_refined.unsqueeze(-1)   # [B, L, C]
            _, topk_indices = token_mask_refined.topk(k_max, dim=-1)    # [B, k_max]
            topk_exp     = topk_indices.unsqueeze(-1).expand(B, k_max, C)
            h_compact    = torch.gather(weighted, 1, topk_exp)          # [B, k_max, C]

            compact_mask = torch.zeros(B, k_max, dtype=torch.bool, device=x.device)
            for i in range(B):
                compact_mask[i, :int(k_per_graph[i].item())] = True

            # ── Run MHA on compact tensor ─────────────────────────────
            h_attn, _ = self.attn(h_compact, h_compact, h_compact,
                                  key_padding_mask=~compact_mask,
                                  need_weights=False)
            # h_attn: [B, k_max, C]

            # ── Head gating (NEW) ─────────────────────────────────────
            # Reshape → scale per head → reshape back
            # [B, k_max, C] → [B, k_max, H, d] → scale → [B, k_max, C]
            if head_gate is not None:
                h_attn = h_attn.view(B, k_max, self.heads, self.d_head)
                # head_gate: [B, H] → [B, 1, H, 1] to broadcast over k and d
                h_attn = h_attn * head_gate.unsqueeze(1).unsqueeze(-1)
                h_attn = h_attn.reshape(B, k_max, C)

            # ── Scatter back to full-length dense tensor ──────────────
            h_full = torch.zeros_like(h_dense)
            h_full.scatter_(1, topk_exp, h_attn)
            h = h_full[mask]                                            # [N_total, C]

            # MAC counters
            self._last_k_avg  = k_per_graph.float().mean()
            self._last_L      = float(L)
            self._actual_macs = float(B) * (k_max ** 2) * self.heads * self.d_head
            real_L            = real_lengths.float()
            self._dense_macs  = float((real_L ** 2).sum().item()) * self.heads * self.d_head

        else:
            # ── Fallback: full attention, no token pruning ────────────
            h_attn, _ = self.attn(h_dense, h_dense, h_dense,
                                  key_padding_mask=~mask,
                                  need_weights=False)
            # h_attn: [B, L, C]

            # Head gating still applies in fallback path
            if head_gate is not None:
                h_attn = h_attn.view(B, L, self.heads, self.d_head)
                h_attn = h_attn * head_gate.unsqueeze(1).unsqueeze(-1)
                h_attn = h_attn.reshape(B, L, C)

            h = h_attn[mask]
            self._last_k_avg = float(L)
            self._last_L     = float(L)
            real_L           = mask.sum(dim=1).float()
            dense_cost       = float((real_L ** 2).sum().item()) * self.heads * self.d_head
            self._actual_macs = dense_cost
            self._dense_macs  = dense_cost

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x   # residual

        # Differentiable gradient coupling for token_ratio → BudgetNet
        if token_ratio is not None:
            h = h * token_ratio[batch].unsqueeze(-1)

        if self.norm2 is not None:
            h = self.norm2(h, batch=batch) if self.norm_with_batch else self.norm2(h)
        hs.append(h)

        # ============================================================
        # 3.  Combine + MLP
        # ============================================================
        out = sum(hs)
        out = out + self.mlp(out)
        if self.norm3 is not None:
            out = self.norm3(out, batch=batch) if self.norm_with_batch else self.norm3(out)

        # ============================================================
        # 4.  Layer gate  (soft residual mix)
        # ============================================================
        if gate_weight is not None:
            out = gate_weight * out + (1.0 - gate_weight) * x

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')