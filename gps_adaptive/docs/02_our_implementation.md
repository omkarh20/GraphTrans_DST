# Our Implementation: Adaptive Graph Transformer Sparsification (Adaptive GTSP)

> **Location:** `gps_adaptive/` directory  
> **Key files:** `gps_conv.py`, `adaptive_model.py`, `main.py`, `utils.py`  
> **Built on:** Original GTSP token pruning (`gps_token/`)

---

## 1. What We Changed (High-Level)

We took the original GTSP token pruning approach and solved its four major limitations:

| Original GTSP Problem | Our Solution |
|----------------------|-------------|
| Static `token_ratio=0.5` for all graphs | **BudgetNet** predicts a unique ratio per graph, per layer |
| No layer skipping capability | **Layer gates** allow the model to skip entire layers |
| Zero-masking (no real FLOP savings) | **Physical gathering** into a shorter tensor before attention |
| No awareness of graph structure | **Graph-conditioned routing** using size, density, degree stats |

---

## 2. Architecture Overview

```
Input Graph (x, edge_index, batch)
              │
              ▼
    ┌──────────────────────┐
    │   Node Embedding     │  x = Linear(features) + Linear(PE)
    └──────────────────────┘
              │
              ├─────────────────────────────────────┐
              ▼                                     ▼
    ┌──────────────────────┐          ┌──────────────────────────┐
    │   BudgetNet          │          │   GPS Layer 1            │
    │   (runs ONCE)        │ ──────── │   GPS Layer 2            │
    │                      │ ratios   │   GPS Layer 3            │
    │   Predicts per-layer │ & gates  │   GPS Layer 4            │
    │   token_ratios [B,L] │          │   (each receives its own │
    │   layer_gates  [B,L] │          │    ratio + gate)         │
    └──────────────────────┘          └──────────────────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │  Global Pooling   │
                                        │  + Classification │
                                        └──────────────────┘
                                                  │
                                                  ▼
                                          Prediction + Loss
                               L = L_task + λ · L_compute
```

---

## 3. File-by-File Breakdown

### 3.1 `adaptive_model.py` — The BudgetNet Controller

**Purpose:** A lightweight MLP that looks at the *entire graph* before any transformer layer runs, and decides how much compute each layer should use.

**Input features (per graph):**

| Feature | Shape | What it captures |
|---------|-------|-----------------|
| `log(N+1)` | `[B, 1]` | Graph size (number of nodes) |
| `log(E+1)` | `[B, 1]` | Edge count |
| `density` | `[B, 1]` | `2E / (N(N-1))` — how connected the graph is |
| `avg_degree` | `[B, 1]` | Mean node degree |
| `degree_variance` | `[B, 1]` | Variance in degree distribution (captures hub structure) |
| `pooled_embedding` | `[B, C]` | Mean-pooled node embeddings (semantic content) |

**Total input dimension:** `5 + channels` (e.g., 5 + 64 = 69)

**Architecture:**
```python
Input [B, 69] → Linear(69, 64) → ReLU → Linear(64, 64) → ReLU
                                                │
                    ┌───────────────────────────┤
                    ▼                           ▼
           token_head: Linear(64, 4)    layer_head: Linear(64, 4)
                    │                           │
                    ▼                           ▼
            sigmoid → affine scale       sigmoid
            [min_ratio, 1.0]             [0.0, 1.0]
                    │                           │
                    ▼                           ▼
          token_ratios [B, 4]          layer_gates [B, 4]
```

**Key design decisions:**
- The token ratio output is clamped to `[min_ratio, 1.0]` (default `min_ratio=0.2`) so the model can never drop *all* nodes.
- The layer gate is a soft sigmoid during training, allowing gradient flow. During eval, values near 0 effectively skip the layer.

> [!NOTE]
> BudgetNet runs **once per forward pass**, not once per layer. It predicts budgets for all 4 layers simultaneously. This makes the routing decision "holistic" — it can allocate more compute to early layers and less to later ones, or vice versa.

---

### 3.2 `gps_conv.py` — The Adaptive GPSConv Layer

**Purpose:** A single GPS transformer layer that accepts external `token_ratio` and `layer_gate` values and performs **true token pruning** (physical gathering, not masking).

**The forward pass has 4 stages:**

#### Stage 0: Layer Gating
```python
gate_weight = layer_gate[batch].unsqueeze(-1)   # [N_total, 1]
```
- Expands the per-graph gate to per-node.
- Applied at the end: `out = gate * layer_output + (1 - gate) * input`
- If gate ≈ 0, output ≈ input (layer is effectively skipped).

#### Stage 1: Local MPNN
```python
h = self.conv(x, edge_index)   # GINConv message passing
h = h + x                       # Residual
h = norm1(h)                     # BatchNorm
```
Standard local message passing — unchanged from the original.

#### Stage 2: Adaptive Token Selection + Attention (THE KEY CHANGE)

This is where our implementation fundamentally differs from the original:

```
     Dense batch [B, L, C]
              │
              ▼
     ┌─────────────────────┐
     │ Score each node      │   scorer: 2-layer MLP → [B, L]
     │ (learned importance) │   (vs. original's single Linear)
     └─────────────────────┘
              │
              ▼
     ┌─────────────────────┐
     │ Compute per-graph k  │   k_i = ceil(token_ratio_i × N_i)
     │ (ADAPTIVE per graph) │   clamped to [1, max_k]
     └─────────────────────┘
              │
              ▼
     ┌─────────────────────┐
     │ Gumbel top-k select  │   Differentiable selection with
     │ (with τ annealing)   │   temperature annealing τ: 2.0 → 0.5
     └─────────────────────┘
              │
              ▼
     ┌─────────────────────────────┐
     │ GATHER into compact tensor  │   h_compact: [B, k_max, C]
     │ (physically shorter!)       │   k_max << L  ← THIS IS THE KEY
     └─────────────────────────────┘
              │
              ▼
     ┌─────────────────────────────┐
     │ Attention on COMPACT tensor │   Cost: O(k_max²), NOT O(L²)
     │ self.attn(h_compact, ...)   │   ← REAL FLOP SAVINGS
     └─────────────────────────────┘
              │
              ▼
     ┌─────────────────────────────┐
     │ SCATTER back to full size   │   h_full: [B, L, C]
     │ (put results back in place) │   Non-selected positions stay zero
     └─────────────────────────────┘
              │
              ▼
     Extract sparse: h = h_full[mask]   → [N_total, C]
```

> [!IMPORTANT]
> **The critical difference:** The original code does `h = h * mask; attn(h, h, h)` — attention runs on the full `[B, L, C]` tensor. Our code does `h_compact = gather(h, topk_indices); attn(h_compact, h_compact, h_compact)` — attention runs on the shorter `[B, k_max, C]` tensor. This is the source of **real, measurable FLOP reduction**.

#### Stage 3: Combine + MLP
```python
out = local_mpnn_output + attention_output
out = out + MLP(out)
out = norm3(out)
```

#### Stage 4: Apply Layer Gate
```python
out = gate * out + (1 - gate) * x   # Soft skip connection
```

#### FLOP Tracking (Phase 2)
Each forward pass also records:
- `_actual_macs`: The real attention cost with pruning (`k_max² × heads × d_head`)
- `_dense_macs`: What it would have cost without pruning (`L² × heads × d_head`)

These are collected in `main.py` to compute `FLOP Reduction = 1 - (actual / dense)`.

---

### 3.3 `main.py` — Training Script

**Purpose:** End-to-end training with cross-validation, Gumbel temperature annealing, and compute-aware loss.

#### The AdaptiveGPS Model Class
Wraps everything together:
```python
class AdaptiveGPS:
    node_emb          # Linear projection of input features
    pe_lin            # Linear projection of positional encodings (Random Walk PE)
    convs             # ModuleList of 4 GPSConv layers
    budget_net        # BudgetNet controller
    lin               # Final classification head
```

#### The Loss Function
```python
loss = cross_entropy(logits, labels) + λ × compute_cost
```
Where `compute_cost` is a differentiable FLOP proxy:
```python
layer_cost = (k_avg² / L²) × mean(layer_gate)
avg_compute = mean(layer_costs across all layers)
```
- When `λ = 0`: No sparsity pressure; model keeps all tokens.
- When `λ > 0`: Model is penalized for using more compute, encouraging it to learn efficient routing.
- Default: `λ = 0.01`.

#### Gumbel Temperature Annealing
```python
τ = τ_start + (τ_end - τ_start) × (epoch / total_epochs)
# Default: τ: 2.0 → 0.5 over training
```
- **High τ (early training):** Gumbel-Softmax produces soft, exploratory selections. The model can try different token subsets.
- **Low τ (late training):** Selections become nearly hard (binary). The model commits to specific pruning patterns.

#### Cross-Validation
- Uses `StratifiedKFold` from scikit-learn (default 10 folds).
- Each fold trains a **fresh model from scratch**.
- Results reported as `Mean ± Std` across folds.

#### What Gets Saved

| File | Location | Contents |
|------|----------|---------|
| `best_model.pt` | `exps/<run>/fold_X/` | Best model weights for each fold |
| `epoch_log.csv` | `exps/<run>/fold_X/` | Per-epoch: loss, accuracy, token ratio, FLOP reduction, τ |
| `cv_result_<seed>.csv` | `results/<dataset>/` | Per-fold final results |
| `cv_summary_<seed>.csv` | `results/<dataset>/` | Aggregated mean ± std across all folds |

---

### 3.4 `utils.py` — Result Logging

Handles writing results to CSV files. Tracks:
- Accuracy (mean ± std)
- Average token ratio (mean ± std)
- FLOP reduction percentage (mean ± std)

---

## 4. Hardware Safety: The `max_k` Cap

The DD dataset contains graphs ranging from ~30 to >5,700 nodes. Without safeguards, a 5,700-node graph would create a `[5700, 5700]` attention matrix — instant OOM on consumer GPUs.

Our solution:
```python
k_per_graph = k_per_graph.clamp(min=1, max=self.max_k)  # default max_k=512
```

Even if `token_ratio=1.0` (keep everything), the attention sequence is capped at 512 tokens. This guarantees the model can train on an RTX 3050 (4GB VRAM) without crashes.

---

## 5. How to Run

### Quick test (3 folds, 5 epochs):
```bash
python main.py --dataset DD --epochs 5 --batch_size 4 --eval_batch_size 4 --n_folds 3
```

### Full run (10 folds, 100 epochs):
```bash
python main.py --dataset DD --epochs 100 --batch_size 4 --eval_batch_size 4 --n_folds 10 --lambda_compute 0.01
```

### Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `DD` | TUDataset name |
| `--epochs` | `100` | Training epochs per fold |
| `--n_folds` | `10` | Number of CV folds |
| `--batch_size` | `4` | Training batch size (keep small for large graphs) |
| `--lambda_compute` | `0.01` | Weight for compute penalty in loss |
| `--tau_start` | `2.0` | Initial Gumbel temperature |
| `--tau_end` | `0.5` | Final Gumbel temperature |
| `--min_token_ratio` | `0.2` | Minimum fraction of tokens BudgetNet can output |
| `--max_k` | `512` | Hard cap on attention sequence length |
| `--num_layers` | `4` | Number of GPS transformer layers |
| `--gnn_emb_dim` | `64` | Hidden dimension |
| `--nhead` | `4` | Number of attention heads |

---

## 6. Sample Output

```
Dataset: DD
  Total graphs: 1178
  Classes: 2
  Node features: 89
  Cross-validation folds: 3

============================================================
  FOLD 1 / 3
  Train size: 785, Test size: 393
============================================================
  Fold 1 | Epoch 001 | Loss 2.9631 (task 2.9629, comp 0.0170) | Test 0.4198 | tau 2.00 | avg_ratio 0.565 | FLOP_red 42.3%
  Fold 1 | Epoch 100 | Loss 0.5012 (task 0.5011, comp 0.0008) | Test 0.7788 | tau 0.50 | avg_ratio 0.603 | FLOP_red 47.1%
  Epoch log saved to: exps/.../fold_0/epoch_log.csv
  Fold 1 BEST => Acc: 0.7788 | Avg Token Ratio: 0.603 | FLOP Reduction: 47.1%

============================================================
Cross-Validation Summary (3 folds)
  Val  Acc: 0.7054 ± 0.0218
  Test Acc: 0.7054 ± 0.0218
  Avg Token Ratio: 0.5866 ± 0.0283
  FLOP Reduction:  45.2% ± 2.1%
  Results saved to: ./results/DD/cv_summary_12344.csv
============================================================
```

> [!TIP]
> **Read next:** `03_roadmap.md` — The phased plan to collect all data needed for the research paper.
