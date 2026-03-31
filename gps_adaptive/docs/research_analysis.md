# Roadmap to a Scopus-Indexed Paper

Publishing in a Scopus-indexed conference or journal requires demonstrating that your method is not just an interesting idea, but a **robust, reproducible, and superior** alternative to existing works across multiple scenarios. 

> [!WARNING]
> A single run of 77.7% accuracy on the DD dataset is a strong proof-of-concept, but it will **not** be enough for acceptance. Reviewers will immediately ask for statistical significance, comparisons to baselines, and a demonstration across varied graph types.

Here is a detailed analysis of what you need to achieve publication quality.

## 1. Multiple Datasets (Diverse Graph Structures)
Reviewers want to see that your adaptive routing doesn't just work on one specific type of protein. You need to prove it generalizes.

*   **TUDataset (Biological):** `DD`, `PROTEINS`, `NCI1`. 
*   **TUDataset (Social Networks):** `REDDIT-BINARY` or `IMDB-BINARY`. These are critical because social networks have drastically different topological features (high degree hubs) than proteins.
*   **OGB (Scale):** Use a subset of `ogbg-molhiv` or `ogbg-code2` to prove the method scales to standard modern benchmarks.

*Recommendation: Select 4 datasets (2 biological, 1 social, 1 OGB).*

## 2. Rigorous Baselines (Fair Comparison)
You must empirically prove that your `Adaptive-GPS` is better than standard models.

*   **Standard GNNs:** Report scores for `GIN`, `GCN`, and `GraphSAGE`.
*   **The Unpruned Baseline:** You must run the exact same hyperparameter sweep on the vanilla `gps_base`.
*   **The Pruned Baseline:** You must run the `gps_token` model from the original paper (where the token keep ratio is a static 0.5 for everything).
*   **Your Model:** `Adaptive-GPS`.

*Recommendation: For every dataset, report [Accuracy, MACs/FLOPs, Memory] across these 4-5 models.*

## 3. Statistical Significance
A single seed (e.g., `--seed 12344`) is considered anecdotal in modern deep learning research. 

*   You must run every model 5-10 times with different random seeds.
*   Report the results as `Mean ± Standard Deviation` (e.g., $77.8 \pm 1.2 \%$).
*   *(Optional but impressive):* Use 10-fold cross-validation for the smaller TUDatasets.

## 4. Required Ablation Studies
Ablation studies prove you understand *why* your model works. Reviewers love a robust ablation section.

*   **The $\lambda_{compute}$ Trade-off Curve:** Run your model with $\lambda \in \{0.0, 0.005, 0.01, 0.05, 0.1\}$. Plot a line graph of **Accuracy vs. Avg Token Ratio**. This shows the exact Pareto frontier of your model.
*   **Gating vs. Token Pruning:** Does the model work if you disable layer skipping and only use token pruning? Code changes will be required to toggle these independently to prove both components contribute.
*   **BudgetNet Input Features:** Are the degree variance and density actually helping? Remove the scalar inputs to the `BudgetNet` and see if the pure node embedding is enough.

## 5. Required Code Modications (Do Not Implement Yet)
If you decide to proceed with this roadmap, we will need to augment your codebase significantly:

1.  **FLOPs / MACs Profiler:** Instead of reporting a "compute proxy loss", we need to run a library like `thop` or PyTorch Profiler to report exact theoretical FLOP reductions, or log wall-clock inference latency (ms/graph).
2.  **Cross-Validation Loop:** Modify `main.py` to support `StratifiedKFold` from scikit-learn for small datasets to report average performance.
3.  **Seed Looping:** Write a master bash script (`run_all.sh`) that automatically loops over seeds, models, and datasets, dumping everything into an organized CSV.
4.  **Ablation Switches:** Add command-line arguments to `main.py` like `--disable_layer_gate` to easily trigger ablations.

## Conclusion
Your current result proves the core mathematics of the paper are sound. To get Scopus indexed, the project shifts from "Method Development" to "Exhaustive Evaluation." You need to execute the runs structurally, cleanly, and compile the massive amount of data into convincing graphs and tables.
