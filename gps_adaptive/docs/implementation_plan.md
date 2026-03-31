# Roadmap: Adaptive GTSP → Scopus-Indexed Paper

We will execute the following phases **sequentially**, one at a time. Each phase builds on the previous one.

---

## Phase 1: Statistical Rigor (Cross-Validation + Multi-Seed)

**Why:** A single random split with one seed is anecdotal. Reviewers demand reproducibility.

#### [MODIFY] [main.py](file:///c:/Users/Omkar%20S%20Hegde/Desktop/Projects/GraphTrans_DST/gps_adaptive/main.py)
- Replace `random_split` with `StratifiedKFold` (10-fold) from `scikit-learn`.
- Each fold trains from scratch, records best val/test accuracy.
- After all folds, print and log: **Mean ± Std** for accuracy & avg_token_ratio.
- Support `--n_folds 10` argument (default 10, use 3 for quick debugging).

#### [MODIFY] [utils.py](file:///c:/Users/Omkar%20S%20Hegde/Desktop/Projects/GraphTrans_DST/gps_adaptive/utils.py)
- Update `results_to_file` to accept and log fold-aggregated results (mean, std).

---

## Phase 2: FLOP / MAC Profiling

**Why:** Our `compute_loss` proxy is good for training, but reviewers need exact theoretical FLOP counts to verify efficiency claims.

#### [MODIFY] [gps_conv.py](file:///c:/Users/Omkar%20S%20Hegde/Desktop/Projects/GraphTrans_DST/gps_adaptive/gps_conv.py)
- Add an internal counter that tracks, per forward pass:
  - **Attention MACs:** `k_max² × heads × (channels / heads)` (the actual cost after pruning)
  - **Dense baseline MACs:** `L² × heads × (channels / heads)` (what it would have been without pruning)
- Store these as `self._actual_macs` and `self._dense_macs`.

#### [MODIFY] [main.py](file:///c:/Users/Omkar%20S%20Hegde/Desktop/Projects/GraphTrans_DST/gps_adaptive/main.py)
- At the end of evaluation, sum up MACs across layers.
- Log: `FLOP Reduction = 1 - (actual_macs / dense_macs)` as a percentage.
- This gives you a hard number like "47.3% FLOP reduction" for your paper tables.

---

## Phase 3: Ablation Switches

**Why:** Proving that *each component contributes* is required for acceptance. Reviewers will ask "what if you remove the layer gate?"

#### [MODIFY] [main.py](file:///c:/Users/Omkar%20S%20Hegde/Desktop/Projects/GraphTrans_DST/gps_adaptive/main.py)
- Add `--disable_layer_gate` flag: forces all layer gates to 1.0 (always active).
- Add `--disable_token_pruning` flag: forces all token ratios to 1.0 (keep everything).
- Add `--disable_graph_stats` flag: removes structural scalars from BudgetNet input (only uses pooled embedding).

#### [MODIFY] [adaptive_model.py](file:///c:/Users/Omkar%20S%20Hegde/Desktop/Projects/GraphTrans_DST/gps_adaptive/adaptive_model.py)
- Accept a `use_graph_stats` boolean; when False, zero out the 5 scalar features.

This gives you a clean ablation table:

| Variant | Token Pruning | Layer Gate | Graph Stats | Acc |
|---------|:---:|:---:|:---:|:---:|
| Full Model | ✓ | ✓ | ✓ | X.X |
| No Layer Gate | ✓ | ✗ | ✓ | X.X |
| No Token Pruning | ✗ | ✓ | ✓ | X.X |
| No Graph Stats | ✓ | ✓ | ✗ | X.X |

---

## Phase 4: Baseline Comparison Models

**Why:** You must compare against the dense baseline and the static-pruning baseline from the original paper.

#### [NEW] `run_baseline.py`
- A standalone script that trains the vanilla `gps_base` model on the same datasets with the same cross-validation procedure.
- This gives you the "unpruned" accuracy and FLOP count as the upper-bound reference.

#### [NEW] `run_static.py`
- A standalone script that trains the `gps_token` model (static `token_ratio=0.5`) on the same datasets.
- This is the "GTSP baseline" you are improving upon.

These two scripts share the same data loading and evaluation code but use different model classes.

---

## Phase 5: Multi-Dataset Support

**Why:** A paper with results on only DD will not be accepted. You need at least 3-4 datasets.

#### [MODIFY] [main.py](file:///c:/Users/Omkar%20S%20Hegde/Desktop/Projects/GraphTrans_DST/gps_adaptive/main.py)
- Generalize `load_data` to cleanly handle: `DD`, `PROTEINS`, `NCI1`, `IMDB-BINARY`.
- Each dataset may have different feature dimensions, number of classes, and graph sizes.
- Add per-dataset default hyperparameters (e.g., `max_k=512` for DD but `max_k=128` for NCI1).

> [!IMPORTANT]
> `REDDIT-BINARY` has graphs with up to ~3,800 nodes and NO node features. It will need special handling (degree-based features) and a very low `max_k`. We should tackle it last after the smaller datasets are stable.

---

## Phase 6: Automated Experiment Runner

**Why:** Running 4 datasets × 3 models × 5 lambda values × 10 folds manually is impossible. We automate it.

#### [NEW] `run_all_experiments.py`
- A Python script (not bash, since you're on Windows) that:
  1. Loops over datasets: `[DD, PROTEINS, NCI1, IMDB-BINARY]`
  2. Loops over models: `[baseline, static, adaptive]`
  3. Loops over lambda values: `[0.0, 0.005, 0.01, 0.05]`
  4. Runs each configuration for 10-fold CV
  5. Dumps all results into a single `results/all_experiments.csv`
- This script can run overnight and you wake up with a complete paper results table.

---

## Execution Order

| Step | Phase | Estimated Time | What You Get |
|:---:|:---:|:---:|:---|
| 1 | Phase 1 | ~1 hour code + overnight run | Statistically valid DD results |
| 2 | Phase 2 | ~30 min code | Exact FLOP reduction numbers |
| 3 | Phase 3 | ~30 min code + overnight run | Ablation table |
| 4 | Phase 4 | ~1 hour code + overnight run | Baseline comparison |
| 5 | Phase 5 | ~1 hour code | Multi-dataset support |
| 6 | Phase 6 | ~1 hour code + 1-2 day run | Complete paper results |

## Open Questions

1. **Phase 1 first?** Shall I start implementing 10-fold cross-validation now so you can kick off an overnight DD run tonight?
2. **OGB datasets:** Do you want to include `ogbg-molhiv` (an OGB molecular benchmark) as a 5th dataset for extra credibility, or stick with 4 TUDatasets to keep things manageable?
