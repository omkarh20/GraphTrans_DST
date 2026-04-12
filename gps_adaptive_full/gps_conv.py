"""
Adaptive GPSConv — Token + Layer + Head
========================================
Modifications over gps_adaptive version:

1. **True token pruning**    – top-k tokens gathered before attention → O(k²)
2. **Adaptive layer gating** – soft/hard gate skips the entire layer
3. **Learnable node scorer** – small MLP scores node importance for top-k
4. **Adaptive head gating**  – per-graph, per-head importance weights from
                               BudgetNet scale each head's output after MHA.

Head gating mechanics
---------------------
PyTorch's nn.MultiheadAttention fuses all heads into a single kernel, so we
cannot physically drop individual heads per sample in a batch. Instead:

    attn_out  [B, k, C]  →  reshape  →  [B, k, H, d]
    head_gate [B, H]     →  unsqueeze → [B, 1, H, 1]  →  multiply
    reshape back         →  [B, k, C]

FIX for constant head gate collapse:
  - Added gradient coupling: after head gating, scale h by mean(head_gate)
    This creates:  task_loss → h → mean(hg) → BudgetNet
    Without this, head_gate has no direct gradient path from task loss,
    so BudgetNet collapses to a constant value that minimises entropy loss.
"""

import inspect
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
# Gumbel-Softmax helpers  (unchanged)
# -----------------------------------------------------------------------
def _scatter_onehot(logits: Tensor, index: Tensor, k: int) -> Tensor:
    B = logits.size(0)
    x_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, k).reshape(-1)
    y_idx = index.reshape(-1)
    out   = torch.zeros_like(logits)
    out[x_idx, y_idx] = 1.0
    return out


def gumbel_topk(logits: Tensor, k: int, tau: float = 1.0,
                hard: bool = True) -> Tensor:
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft  = gumbels.softmax(dim=-1)
    if hard:
        index  = y_soft.topk(k, dim=-1)[1]
        y_hard = _scatter_onehot(logits, index, k)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


# -----------------------------------------------------------------------
# Adaptive GPSConv  (Token + Layer + Head)
# -----------------------------------------------------------------------
class GPSConv(torch.nn.Module):
    r"""GPS convolution with adaptive token pruning, layer gating, head gating."""

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

        assert channels % heads == 0, \
            f"channels ({channels}) must be divisible by heads ({heads})"
        self.d_head = channels // heads

        self.attn = torch.nn.MultiheadAttention(
            channels, heads, dropout=attn_dropout, batch_first=True)

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1  = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2  = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3  = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

        self.scorer = Sequential(
            Linear(channels, channels), ReLU(), Linear(channels, 1))

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
        token_ratio: Optional[Tensor] = None,   # [B]
        layer_gate:  Optional[Tensor] = None,   # [B]
        head_gate:   Optional[Tensor] = None,   # [B, H]
        tau: float = 1.0,
        **kwargs,
    ) -> Tensor:

        # ============================================================
        # 0.  Layer gate weight
        # ============================================================
        gate_weight = (layer_gate[batch].unsqueeze(-1)
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
                h = (self.norm1(h, batch=batch) if self.norm_with_batch
                     else self.norm1(h))
            hs.append(h)

        # ============================================================
        # 2.  Global attention with token pruning + head gating
        # ============================================================
        h_dense, mask = to_dense_batch(x, batch)
        B, L, C = h_dense.shape

        if token_ratio is not None and L > 1:
            # ── Token selection ──────────────────────────────────────
            real_lengths = mask.sum(dim=1)
            k_per_graph  = (token_ratio * real_lengths.float()).ceil().long()
            k_per_graph  = k_per_graph.clamp(min=1, max=self.max_k)
            k_per_graph  = torch.where(
                k_per_graph > real_lengths, real_lengths, k_per_graph)
            k_max = int(k_per_graph.max().item())

            scores     = self.scorer(h_dense).squeeze(-1)
            scores     = scores.masked_fill(~mask, float('-inf'))
            log_scores = F.log_softmax(scores, dim=-1)
            token_mask = gumbel_topk(log_scores, k=k_max, tau=tau, hard=True)

            sorted_mask, sort_idx = token_mask.sort(dim=-1, descending=True)
            for i in range(B):
                ki = int(k_per_graph[i].item())
                if ki < k_max:
                    sorted_mask[i, ki:] = 0.0
            token_mask_refined = torch.zeros_like(sorted_mask)
            token_mask_refined.scatter_(1, sort_idx, sorted_mask)

            weighted        = h_dense * token_mask_refined.unsqueeze(-1)
            _, topk_indices = token_mask_refined.topk(k_max, dim=-1)
            topk_exp        = topk_indices.unsqueeze(-1).expand(B, k_max, C)
            h_compact       = torch.gather(weighted, 1, topk_exp)

            compact_mask = torch.zeros(B, k_max, dtype=torch.bool, device=x.device)
            for i in range(B):
                compact_mask[i, :int(k_per_graph[i].item())] = True

            # ── MHA ───────────────────────────────────────────────────
            h_attn, _ = self.attn(h_compact, h_compact, h_compact,
                                  key_padding_mask=~compact_mask,
                                  need_weights=False)

            # ── Head gating ───────────────────────────────────────────
            if head_gate is not None:
                h_attn = h_attn.view(B, k_max, self.heads, self.d_head)
                h_attn = h_attn * head_gate.unsqueeze(1).unsqueeze(-1)
                h_attn = h_attn.reshape(B, k_max, C)

            # ── Scatter back ──────────────────────────────────────────
            h_full = torch.zeros_like(h_dense)
            h_full.scatter_(1, topk_exp, h_attn)
            h = h_full[mask]

            self._last_k_avg  = k_per_graph.float().mean()
            self._last_L      = float(L)
            self._actual_macs = float(B) * (k_max ** 2) * self.heads * self.d_head
            real_L            = real_lengths.float()
            self._dense_macs  = float((real_L ** 2).sum().item()) * self.heads * self.d_head

        else:
            # ── Fallback: full attention ──────────────────────────────
            h_attn, _ = self.attn(h_dense, h_dense, h_dense,
                                  key_padding_mask=~mask,
                                  need_weights=False)

            if head_gate is not None:
                h_attn = h_attn.view(B, L, self.heads, self.d_head)
                h_attn = h_attn * head_gate.unsqueeze(1).unsqueeze(-1)
                h_attn = h_attn.reshape(B, L, C)

            h = h_attn[mask]
            self._last_k_avg  = float(L)
            self._last_L      = float(L)
            real_L            = mask.sum(dim=1).float()
            dense_cost        = float((real_L ** 2).sum().item()) * self.heads * self.d_head
            self._actual_macs = dense_cost
            self._dense_macs  = dense_cost

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x

        # ── Gradient coupling: token_ratio → BudgetNet ────────────────
        if token_ratio is not None:
            h = h * token_ratio[batch].unsqueeze(-1)

        # ── Gradient coupling: head_gate → BudgetNet (KEY FIX) ────────
        # avg_head_gate per graph expanded to [N_total, 1]
        # Creates path:  task_loss → h → avg_hg → BudgetNet.head_mlp
        # Graphs that NEED heads to classify correctly push avg_hg UP.
        # Graphs where heads are redundant can afford low avg_hg.
        if head_gate is not None:
            avg_hg_per_node = head_gate.mean(dim=-1)[batch].unsqueeze(-1)
            h = h * avg_hg_per_node

        if self.norm2 is not None:
            h = (self.norm2(h, batch=batch) if self.norm_with_batch
                 else self.norm2(h))
        hs.append(h)

        # ============================================================
        # 3.  Combine + MLP
        # ============================================================
        out = sum(hs)
        out = out + self.mlp(out)
        if self.norm3 is not None:
            out = (self.norm3(out, batch=batch) if self.norm_with_batch
                   else self.norm3(out))

        # ============================================================
        # 4.  Layer gate
        # ============================================================
        if gate_weight is not None:
            out = gate_weight * out + (1.0 - gate_weight) * x

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')