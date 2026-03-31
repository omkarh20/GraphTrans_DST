# Roadmap: Collecting All Data for the Research Paper

> **Goal:** Produce a complete set of tables, plots, and ablation results required for a Scopus-indexed publication.  
> **Current status:** Phase 1 ✅ and Phase 2 ✅ are complete. Phase 1 full run is in progress.

---

## Progress Tracker

| Phase | Description | Code | Run | Status |
|:-----:|------------|:----:|:---:|:------:|
| 1 | Cross-Validation + FLOP Tracking | ✅ Done | 🔄 Running | **In Progress** |
| 2 | FLOP / MAC Profiling | ✅ Done | — | Merged into Phase 1 |
| 3 | Ablation Switches | ❌ Not started | ❌ | Pending |
| 4 | Baseline Comparison Models | ❌ Not started | ❌ | Pending |
| 5 | Multi-Dataset Support | ❌ Not started | ❌ | Pending |
| 6 | Automated Experiment Runner | ❌ Not started | ❌ | Pending |

---

## What a Scopus Reviewer Expects to See

Before diving into the phases, here's the big picture of what the final paper needs:

### Tables the paper must contain:

| Table | Contents | Which phase produces it |
|-------|----------|------------------------|
| **Table 1:** Main Results | Accuracy ± std across 4 datasets × 3 models | Phases 4, 5, 6 |
| **Table 2:** Efficiency | FLOP Reduction %, Token Ratio, Inference Latency | Phases 1, 2 |
| **Table 3:** Ablation | Accuracy when each component is disabled | Phase 3 |
| **Table 4:** λ Sensitivity | Accuracy vs Token Ratio at different λ values | Phase 6 |

### Figures the paper must contain:

| Figure | Contents | Which phase produces it |
|--------|----------|------------------------|
| **Fig 1:** Architecture diagram | BudgetNet + GPSConv data flow | Already available (Doc 02) |
| **Fig 2:** Training curves | Loss / Accuracy / Token Ratio over epochs | Phase 1 (epoch_log.csv) |
| **Fig 3:** Accuracy-Efficiency tradeoff | Pareto curve: Accuracy vs FLOP Reduction | Phase 6 |
| **Fig 4:** Token ratio vs graph size | Shows adaptive routing behavior | Phase 1 data (post-hoc analysis) |

---

## Phase 1: Statistical Rigor ✅ COMPLETE

**What was done:**
- Replaced single random split with `StratifiedKFold` (10-fold) cross-validation
- Each fold trains a fresh model and records the best test accuracy
- Results aggregated as **Mean ± Std** for accuracy, token ratio, and FLOP reduction
- Per-epoch data saved to `epoch_log.csv` for training curve plots

**What it produces for the paper:**
- ✅ Statistically valid accuracy numbers (Mean ± Std)
- ✅ Training curve data (loss, accuracy, ratio vs epochs)
- ✅ Reproducibility (controlled by `--seed` argument)

**Output files:**
```
results/DD/cv_summary_12344.csv     → Aggregated results (1 row: means ± stds)
results/DD/cv_result_12344.csv      → Per-fold results (10 rows)
exps/<run>/fold_X/epoch_log.csv     → Per-epoch metrics for each fold
exps/<run>/fold_X/best_model.pt     → Saved model weights
```

**Current run command:**
```bash
python main.py --dataset DD --epochs 100 --batch_size 4 --eval_batch_size 4 --n_folds 10 --lambda_compute 0.01
```

---

## Phase 2: FLOP / MAC Profiling ✅ COMPLETE

**What was done:**
- Added internal MAC counters to `GPSConv` (`_actual_macs` and `_dense_macs`)
- During evaluation, MACs are summed across all layers and batches
- FLOP Reduction = `1 - (actual_macs / dense_macs)` reported as a percentage
- Integrated into the Phase 1 run — no separate run needed

**What it produces for the paper:**
- ✅ Exact FLOP reduction percentage (e.g., "47.3% FLOP reduction")
- ✅ Per-fold FLOP numbers in CSV for statistical reporting

> [!NOTE]
> Phase 2 was implemented *before* the full Phase 1 run so that all data is captured in a single overnight run. No re-runs needed.

---

## Phase 3: Ablation Switches ❌ PENDING

**Why this is required:**
Reviewers will ask: *"Does every component actually help? What if you remove the layer gate?"* Ablation studies prove that each architectural choice contributes to the final result.

**What needs to be coded:**
- Add 3 command-line flags to `main.py`:
  - `--disable_layer_gate` → Forces all layer gates to 1.0 (always active)
  - `--disable_token_pruning` → Forces all token ratios to 1.0 (keep all tokens)
  - `--disable_graph_stats` → Zeros out the 5 structural scalars in BudgetNet input
- Modify `adaptive_model.py` to respect a `use_graph_stats` flag

**What it produces for the paper:**

> **Table 3: Ablation Study on DD Dataset**
> 
> | Variant | Token Pruning | Layer Gate | Graph Stats | Acc (%) | FLOP Red. (%) |
> |---------|:---:|:---:|:---:|:---:|:---:|
> | **Full Model** | ✓ | ✓ | ✓ | XX.X ± X.X | XX.X ± X.X |
> | No Layer Gate | ✓ | ✗ | ✓ | XX.X ± X.X | XX.X ± X.X |
> | No Token Pruning | ✗ | ✓ | ✓ | XX.X ± X.X | XX.X ± X.X |
> | No Graph Stats | ✓ | ✓ | ✗ | XX.X ± X.X | XX.X ± X.X |

**Runs required:** 3 ablation variants × 10 folds × 100 epochs each (~12 hours total)

**Commands:**
```bash
python main.py --dataset DD --epochs 100 --n_folds 10 --disable_layer_gate
python main.py --dataset DD --epochs 100 --n_folds 10 --disable_token_pruning
python main.py --dataset DD --epochs 100 --n_folds 10 --disable_graph_stats
```

---

## Phase 4: Baseline Comparison Models ❌ PENDING

**Why this is required:**
You cannot claim "our model is better" without showing what you're better *than*. Reviewers require fair, apples-to-apples comparisons.

**What needs to be coded:**
- `run_baseline.py` — Trains the vanilla `gps_base` model (no pruning at all) with identical cross-validation
- `run_static.py` — Trains the original `gps_token` model (static `token_ratio=0.5`) with identical cross-validation
- Both scripts must use the **exact same** data splits, epochs, and evaluation as our adaptive model

**What it produces for the paper:**

> **Table 1: Main Results on DD Dataset**
> 
> | Model | Acc (%) | FLOP Red. (%) | Token Ratio |
> |-------|:---:|:---:|:---:|
> | GPS (unpruned) | XX.X ± X.X | 0.0% | 1.000 |
> | GTSP (static, r=0.5) | XX.X ± X.X | ~XX% | 0.500 |
> | **Adaptive GTSP (ours)** | **XX.X ± X.X** | **XX.X ± X.X%** | **0.XXX** |

**Runs required:** 2 baselines × 10 folds × 100 epochs (~8 hours total)

---

## Phase 5: Multi-Dataset Support ❌ PENDING

**Why this is required:**
A paper with results on only 1 dataset will be rejected. Reviewers expect at least 3-4 diverse datasets.

**Target datasets:**

| Dataset | Type | Graphs | Avg Nodes | Classes | Challenge |
|---------|------|:------:|:---------:|:-------:|-----------|
| **DD** | Protein structure | 1,178 | 284 | 2 | Extreme size variance (30–5,700 nodes) |
| **PROTEINS** | Protein structure | 1,113 | 39 | 2 | Smaller graphs, tests if pruning helps small graphs |
| **NCI1** | Molecular | 4,110 | 30 | 2 | Large dataset, very small graphs |
| **IMDB-BINARY** | Social network | 1,000 | 20 | 2 | No node features — needs degree-based features |

**What needs to be coded:**
- Generalize `load_dataset()` in `main.py` to handle all 4 datasets
- Add per-dataset default hyperparameters (e.g., `max_k=128` for NCI1)
- Handle datasets with no node features (IMDB-BINARY)

> [!IMPORTANT]
> **REDDIT-BINARY** (graphs up to ~3,800 nodes, no node features) is a stretch goal. Only attempt after the 4 core datasets are stable.

**What it produces for the paper:**
The full **Table 1** across all datasets — this is the centerpiece of the paper.

---

## Phase 6: Automated Experiment Runner ❌ PENDING

**Why this is required:**
The full experiment matrix is too large to run manually:
- 4 datasets × 3 models × 5 λ values × 10 folds = **600 individual training runs**

**What needs to be coded:**
- `run_all_experiments.py` — A Python script that:
  1. Loops over datasets: `[DD, PROTEINS, NCI1, IMDB-BINARY]`
  2. Loops over models: `[baseline, static, adaptive]`
  3. Loops over λ values: `[0.0, 0.005, 0.01, 0.05]`
  4. Runs each config with 10-fold CV
  5. Dumps everything into `results/all_experiments.csv`

**What it produces for the paper:**
- Complete data for **all tables and figures**
- λ sensitivity analysis (Accuracy vs Token Ratio at different λ)
- The Pareto frontier plot (Accuracy vs Efficiency)

**Estimated runtime:** 1–2 days (can run overnight)

---

## Execution Timeline

```
Week 1:
  ├── [✅] Phase 1+2 code complete
  ├── [🔄] Phase 1 full DD run (overnight, ~4 hours)
  ├── [ ] Phase 3 code (ablation switches, ~30 min)
  └── [ ] Phase 3 ablation runs (overnight, ~12 hours)

Week 2:
  ├── [ ] Phase 4 code (baseline scripts, ~1 hour)
  ├── [ ] Phase 4 baseline runs (overnight, ~8 hours)
  ├── [ ] Phase 5 code (multi-dataset, ~1 hour)
  └── [ ] Phase 5 test runs on PROTEINS + NCI1

Week 3:
  ├── [ ] Phase 6 code (experiment runner, ~1 hour)
  ├── [ ] Phase 6 full run (1-2 days)
  └── [ ] Compile all results into paper tables and figures
```

> [!TIP]
> **Strategy:** Most coding work can be done during the day, and compute-heavy runs can be kicked off overnight. The bottleneck is GPU time, not coding time.

---

## Summary of Data We Need vs. What We Have

| Data Point | Status | Source |
|-----------|:------:|--------|
| DD accuracy (Mean ± Std, 10-fold) | 🔄 Running | Phase 1 |
| DD FLOP reduction % | 🔄 Running | Phase 2 (in Phase 1 run) |
| DD training curves (epoch logs) | 🔄 Running | Phase 1 epoch_log.csv |
| DD ablation table | ❌ Pending | Phase 3 |
| DD baseline comparison | ❌ Pending | Phase 4 |
| PROTEINS results | ❌ Pending | Phase 5 |
| NCI1 results | ❌ Pending | Phase 5 |
| IMDB-BINARY results | ❌ Pending | Phase 5 |
| λ sensitivity curve | ❌ Pending | Phase 6 |
| Accuracy-Efficiency Pareto plot | ❌ Pending | Phase 6 |
