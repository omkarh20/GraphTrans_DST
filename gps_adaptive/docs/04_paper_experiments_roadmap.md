# Research Paper: Experiments & Insights Roadmap

This roadmap outlines the systematic experiments and data-gathering steps needed to write a mathematically rigorous and empirically strong research paper on the **Monotonic Size-Prior Adaptive Token Pruning** architecture.

## Phase 1: Main Performance Benchmarking (The Core Claims)
*Validate that the method works without hurting accuracy, while saving compute.*
- [ ] **Run Baseline GPS Model**: Run the standard, unpruned GPS model (Token Ratio = 1.0) on 10-fold CV to establish the upper bound for accuracy and compute.
- [ ] **Run Static Pruning Baselines**: Run static token ratios (e.g., Fixed 0.40, Fixed 0.70) to show the advantage of *adaptive* vs *fixed* pruning.
- [ ] **Current 10-Fold Run**: Consolidate the results of the ongoing `size_prior_mix=0.45` 10-fold CV. Compare its accuracy and FLOPs/latency against the baselines.

## Phase 2: Ablation Studies (Proving the Architecture)
*Prove geometrically and empirically why the monotonic size prior was necessary.*
- [ ] **The "Flatline" Ablation (`size_prior_mix = 0.0`)**: Run the model using only the MLP (task-loss driven). Show that the token ratio compresses into a narrow band and ignores structural graph efficiency.
- [ ] **The "Hard Prior" Ablation (`size_prior_mix = 1.0`)**: Run the model relying *only* on the graph size prior, completely ignoring the MLP's task-specific adaptations.
- [ ] **Temperature Sweeps (`size_prior_temp`)**: Test temps like `1.0`, `3.0` (current), and `5.0`. Plot how this controls the aggressiveness of token dropping for large graphs.

## Phase 3: Analyzing Token Dropping Behavior (The "Intelligence")
*Explore what the model is actually doing underneath.*
- [ ] **Graph-Level Property Correlations**: Use our `analyze_token_ratio_properties.py` script on the final 10-fold `graph_stats.csv` to generate the 3x3 panel figures showing how topological features (size, density, degree) dictate pruning.
- [ ] **Layer-Wise Token Dynamics**: Extract average token ratios per layer. Does the model drop more tokens in Layer 1 or Layer 10? Plot Token Ratio vs. Layer Depth.
- [ ] **Node-Level Intelligence (Qualitative)**: Analyze *which* nodes are being dropped. Are they low-degree nodes? High-degree hubs? Visualize a few sample graphs highlighting retained vs. dropped nodes.

## Phase 4: Compute & Hardware Efficiency (The System Impact)
*Translate theoretical claims ($O(N^2)$ reduction) into hardware metrics.*
- [ ] **MACs / FLOPs Tracking**: Calculate the theoretical reduction in compute operations for the Global Attention mechanism.
- [ ] **Real-world Latency/Throughput**: Measure the inference time (milliseconds per batch) of the baseline vs our adaptive method.
- [ ] **Peak Memory (VRAM) Profiling**: Measure maximum GPU memory allocation during evaluation. Show that our method allows processing larger graphs without OOM errors.

## Phase 5: Writing and Visualization Prep
- [ ] Generate the architecture diagram showing the MLP parallel with the Size Prior formula.
- [ ] Stitch together the 3x3 correlation plots.
- [ ] Format the 10-fold CV Results table (Mean $\pm$ Std).
