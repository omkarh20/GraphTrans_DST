"""
utils.py — gps_adaptive_full
=============================
Changes over gps_adaptive version:
  - results_cv_to_file now handles 'avg_head_gate' in fold_results dicts
  - CV per-fold CSV and summary CSV both include avg_head_gate columns
  - Console summary prints avg_head_gate mean ± std
"""

import os
import csv
import numpy as np


def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def results_to_file(args, test_acc, val_acc):
    """Log a single-run result (unchanged)."""
    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("=" * 20)
        print("Creating Results Directory")
        os.makedirs('./results/{}'.format(args.dataset))

    filename   = "./results/{}/result_{}.csv".format(args.dataset, args.seed)
    headerList = ["Method", "lambda_compute", "::::::::", "test_acc", "val_acc"]

    with open(filename, "a+") as f:
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',', fieldnames=headerList)
            dw.writeheader()
        line = "{}, {}, :::::::::, {:.4f}, {:.4f}\n".format(
            args.model_type, args.lambda_compute, test_acc, val_acc)
        f.write(line)


def results_cv_to_file(args, fold_results):
    """
    Log cross-validation results.

    Parameters
    ----------
    args         : argparse.Namespace
    fold_results : list of dict, each with keys:
                     fold, val_acc, test_acc,
                     avg_token_ratio, avg_head_gate, flop_reduction
    """
    if not os.path.exists('./results/{}'.format(args.dataset)):
        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/cv_result_{}.csv".format(args.dataset, args.seed)

    # ── Per-fold CSV ──────────────────────────────────────────────────────────
    headerList = [
        "Method", "lambda_compute", "n_folds",
        "fold", "val_acc", "test_acc",
        "avg_token_ratio", "avg_head_gate",   # ← avg_head_gate added
        "flop_reduction",
    ]

    with open(filename, "w", newline='') as f:
        dw = csv.DictWriter(f, delimiter=',', fieldnames=headerList)
        dw.writeheader()
        for r in fold_results:
            dw.writerow({
                "Method":          args.model_type,
                "lambda_compute":  args.lambda_compute,
                "n_folds":         args.n_folds,
                "fold":            r["fold"],
                "val_acc":         f"{r['val_acc']:.4f}",
                "test_acc":        f"{r['test_acc']:.4f}",
                "avg_token_ratio": f"{r['avg_token_ratio']:.4f}",
                "avg_head_gate":   f"{r['avg_head_gate']:.4f}",   # ← NEW
                "flop_reduction":  f"{r['flop_reduction']:.4f}",
            })

    # ── Summary CSV ───────────────────────────────────────────────────────────
    val_accs   = [r["val_acc"]         for r in fold_results]
    test_accs  = [r["test_acc"]        for r in fold_results]
    ratios     = [r["avg_token_ratio"] for r in fold_results]
    head_gates = [r["avg_head_gate"]   for r in fold_results]   # ← NEW
    flop_reds  = [r["flop_reduction"]  for r in fold_results]

    summary_file = "./results/{}/cv_summary_{}.csv".format(
        args.dataset, args.seed)

    with open(summary_file, "w", newline='') as f:
        headers = [
            "Method", "lambda_compute", "n_folds",
            "val_mean",  "val_std",
            "test_mean", "test_std",
            "ratio_mean",     "ratio_std",
            "head_gate_mean", "head_gate_std",   # ← NEW
            "flop_red_mean",  "flop_red_std",
        ]
        dw = csv.DictWriter(f, delimiter=',', fieldnames=headers)
        dw.writeheader()
        dw.writerow({
            "Method":          args.model_type,
            "lambda_compute":  args.lambda_compute,
            "n_folds":         args.n_folds,
            "val_mean":        f"{np.mean(val_accs):.4f}",
            "val_std":         f"{np.std(val_accs):.4f}",
            "test_mean":       f"{np.mean(test_accs):.4f}",
            "test_std":        f"{np.std(test_accs):.4f}",
            "ratio_mean":      f"{np.mean(ratios):.4f}",
            "ratio_std":       f"{np.std(ratios):.4f}",
            "head_gate_mean":  f"{np.mean(head_gates):.4f}",   # ← NEW
            "head_gate_std":   f"{np.std(head_gates):.4f}",    # ← NEW
            "flop_red_mean":   f"{np.mean(flop_reds):.4f}",
            "flop_red_std":    f"{np.std(flop_reds):.4f}",
        })

    print("\n" + "=" * 60)
    print(f"Cross-Validation Summary ({args.n_folds} folds)")
    print(f"  Val  Acc:        {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"  Test Acc:        {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"  Avg Token Ratio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    print(f"  Avg Head Gate:   {np.mean(head_gates):.4f} ± {np.std(head_gates):.4f}")  # ← NEW
    print(f"  FLOP Reduction:  {np.mean(flop_reds):.1%} ± {np.std(flop_reds):.1%}")
    print(f"  Results saved to: {summary_file}")
    print("=" * 60)