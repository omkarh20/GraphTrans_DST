# Original Paper Summary: Exploring Sparsity in Graph Transformers (GTSP)

> **Paper Title:** Exploring Sparsity in Graph Transformers  
> **Codebase:** [GraphTrans_DST](https://github.com/omkarh20/GraphTrans_DST)  
> **Baseline Transformers Studied:** GraphTrans, Graphormer, GraphGPS

---

## 1. The Problem

Graph Transformers (GTs) achieve state-of-the-art results on graph classification, molecular property prediction, and similar tasks. However, they suffer from a **quadratic computational bottleneck**: the global self-attention mechanism has $O(N^2)$ cost, where $N$ is the number of nodes in the graph.

This makes GTs expensive to deploy, especially:
- On **large graphs** (thousands of nodes), where the attention matrix alone can consume gigabytes of GPU memory.
- In **resource-constrained environments** (edge devices, consumer GPUs), where both memory and FLOPs are limited.

> [!IMPORTANT]
> The core research question: *Can we make Graph Transformers significantly cheaper without significantly hurting accuracy?*

---

## 2. The Proposed Solution: GTSP Framework

The paper proposes **Graph Transformer SParsification (GTSP)**, a unified framework that reduces the computational cost of GTs by pruning redundancy across **four dimensions**:

### 2.1 Token Pruning (Node-level sparsity)
- Not all nodes in a graph contribute equally to the final classification. Some are structurally redundant.
- GTSP learns to identify and mask out unimportant nodes using **Gumbel-Softmax** based differentiable selection.
- A small linear scorer (`gumbel = nn.Linear(channels, 1)`) scores each node, then the top-k nodes are selected using the Gumbel-Softmax trick.
- The keep ratio is controlled by a **static hyperparameter** `token_ratio` (e.g., 0.5 means keep 50% of nodes).

### 2.2 Head Pruning (Attention head-level sparsity)
- Multi-head attention uses multiple parallel attention heads. Not all heads are equally useful.
- GTSP learns to prune entire attention heads that contribute the least.

### 2.3 Layer Pruning (Depth-level sparsity)
- Deep transformers stack many layers, but not all layers are necessary for every input.
- GTSP explores skipping entire transformer layers.

### 2.4 Weight Pruning (Parameter-level sparsity)
- Standard weight pruning (zeroing out individual parameters) is applied to compress the model.

> [!NOTE]
> The codebase in this repository is organized by dimension. Each folder corresponds to one pruning dimension:
> - `gps_base/` — Vanilla GraphGPS (no pruning, the unpruned baseline)
> - `gps_token/` — Token (node) pruning
> - `gps_head/` — Attention head pruning
> - `gps_layer/` — Layer pruning
> - `gps_weight/` — Weight pruning

---

## 3. How Token Pruning Works (Most Relevant to Our Work)

Since our adaptive extension builds directly on top of the **token pruning** dimension, here is a detailed walkthrough of how it works in the original code (`gps_token/gps_conv.py`):

### Step-by-step flow inside `GPSConv.forward()`:

```
Input: x [N_total, C], edge_index, batch
                    │
                    ▼
    ┌───────────────────────────────┐
    │  1. Local MPNN (GIN/GCN)      │  ← Message passing on the graph structure
    │     h = conv(x, edge_index)   │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  2. to_dense_batch(x, batch)  │  ← Converts sparse [N, C] → dense [B, L, C]
    │     B, L, C = h.shape         │     (L = max nodes in any graph in batch)
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  3. Score each node           │
    │     scores = Linear(h) → [B,L]│  ← Single linear layer scores importance
    │     k = int(L × token_ratio)  │  ← STATIC: same ratio for ALL graphs
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  4. Gumbel-Softmax top-k      │
    │     mask = gumbel_topk(scores) │  ← Differentiable selection of k nodes
    │     h = h * mask              │  ← MASKING: zeroes out dropped nodes
    └───────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │  5. Full attention on FULL tensor   │
    │     attn(h, h, h)  → still [B,L,C] │  ← O(L²) cost, NOT O(k²)!
    └─────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  6. Residual + MLP + Norm     │
    └───────────────────────────────┘
                    │
                    ▼
              Output: x [N_total, C]
```

### Key observations about the original implementation:

| Aspect | Original GTSP Behavior |
|--------|----------------------|
| **Keep ratio** | Fixed hyperparameter (`token_ratio=0.5`). Same for every graph, every layer. |
| **Selection** | Gumbel-Softmax with `tau=1` (fixed, no annealing). |
| **Masking strategy** | Multiply-by-zero: `h = h * mask`. Dropped tokens become zero vectors. |
| **Attention cost** | Still $O(L^2)$. The full `[B, L, C]` tensor (including zeroed-out tokens) is passed to `self.attn()`. |
| **Real FLOP savings** | **Minimal.** The attention matrix is still computed over the full padded length $L$. The zeros technically don't contribute to the output, but the GPU still does all the multiplications. |

> [!WARNING]
> **Critical Limitation:** Because the original code passes the *full-length* tensor (with masked values set to zero) into the attention layer, the actual GPU compute cost is virtually unchanged. The sparsity exists logically but is **not exploited at the hardware level**. This is the key limitation that our adaptive extension addresses.

---

## 4. Reported Results

The paper demonstrates that GTSP can achieve:
- **~30% reduction in FLOPs** (theoretical, across all four dimensions combined).
- **Marginal accuracy loss** or even slight accuracy improvements (e.g., +1.8% AUC on some benchmarks).
- Results are shown across three GT architectures: GraphTrans, Graphormer, and GraphGPS.

Datasets used in the paper include standard graph classification benchmarks from TUDataset (like DD, PROTEINS, NCI1) and molecular benchmarks.

---

## 5. Why We Are Extending This Work

The original GTSP paper established that sparsification is viable for Graph Transformers. However, it has several limitations that motivate our **Adaptive GTSP** extension:

| Limitation | Impact |
|-----------|--------|
| **Static token ratio** | A fixed `token_ratio=0.5` applied uniformly to all graphs. Small graphs (20 nodes) lose critical structure when halved, while large graphs (5000 nodes) still OOM even at 50%. |
| **No layer-level adaptation** | Every graph must pass through every layer, even if a simple graph could be classified after 2 layers. |
| **Zero-masking, not gathering** | Dropped tokens are zeroed but not removed. Attention still runs at $O(L^2)$, so FLOP savings are theoretical, not realized. |
| **No graph-conditioned routing** | The model has no awareness of graph-level properties (size, density, degree distribution) when making pruning decisions. |

> [!TIP]
> **Read next:** `02_our_implementation.md` — How we solved each of these limitations with our Adaptive GTSP architecture.

---

## 6. Key Terminology

| Term | Definition |
|------|-----------|
| **GPS** | General, Powerful, Scalable — a Graph Transformer architecture that combines local MPNN + global attention. |
| **GTSP** | Graph Transformer SParsification — the original paper's framework for pruning GTs. |
| **Token** | In graph context, a "token" = a node. Token pruning = node pruning. |
| **Gumbel-Softmax** | A reparameterization trick that makes discrete selection (top-k) differentiable, enabling end-to-end training of the selector. |
| **`token_ratio`** | Fraction of nodes to keep (0.0 to 1.0). Original paper uses a static value; our extension makes it dynamic per-graph. |
| **`to_dense_batch`** | PyG utility that converts a batched sparse graph `[N_total, C]` into a padded dense tensor `[B, L_max, C]` for attention. |
| **MHA** | Multi-Head Attention — the global attention mechanism inside each transformer layer. |
| **MPNN** | Message Passing Neural Network — the local GNN component (GIN or GCN) inside each GPS layer. |
