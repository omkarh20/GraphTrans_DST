import os
import csv
import numpy as np

def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def results_to_file(args, test_acc, test_std):

    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("="*20)
        print("Creat Resulrts File !!!")
        #os.mkdir('./results/{}'.format(args.data))

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/result_{}.csv".format(
                            args.dataset, args.seed)

    headerList = ["Method","token_remain_ratio",
                "::::::::",
                "test_acc", "val_acc"]

    #filename = "./results/{}/{}_{}_{}_{}_result.csv".format(sparse_way, args.model, args.data, args.final_density, args.final_density_adj)
    with open(filename, "a+") as f:

        # reader = csv.reader(f)
        # row1 = next(reader)
        f.seek(0)
        header = f.read(6)
        if  header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                        fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, :::::::::, {:.4f}, {:.4f}\n".format(
            args.model_type, args.token_ratio,
            test_acc, test_std
        )
        f.write(line)


def results_cv_to_file(args, fold_results):
    """
    Log cross-validation aggregated results for static token ratio.
    
    Parameters
    ----------
    args : argparse.Namespace
    fold_results : list of dict
        Each dict has keys: 'fold', 'val_acc', 'test_acc'
    """
    if not os.path.exists('./results/{}'.format(args.dataset)):
        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/cv_result_{}.csv".format(args.dataset, args.seed)

    # --- Write per-fold rows ---
    headerList = ["Method", "token_ratio", "n_folds",
                  "fold", "val_acc", "test_acc"]

    with open(filename, "w", newline='') as f:
        dw = csv.DictWriter(f, delimiter=',', fieldnames=headerList)
        dw.writeheader()
        for r in fold_results:
            dw.writerow({
                "Method": args.model_type,
                "token_ratio": args.token_ratio,
                "n_folds": args.n_folds,
                "fold": r["fold"],
                "val_acc": f"{r['val_acc']:.4f}",
                "test_acc": f"{r['test_acc']:.4f}",
            })

    # --- Write summary row ---
    test_accs = [r["test_acc"] for r in fold_results]
    val_accs = [r["val_acc"] for r in fold_results]

    summary_file = "./results/{}/cv_summary_{}.csv".format(args.dataset, args.seed)
    with open(summary_file, "w", newline='') as f:
        headers = ["Method", "token_ratio", "n_folds",
                   "val_mean", "val_std", "test_mean", "test_std"]
        dw = csv.DictWriter(f, delimiter=',', fieldnames=headers)
        dw.writeheader()
        dw.writerow({
            "Method": args.model_type,
            "token_ratio": args.token_ratio,
            "n_folds": args.n_folds,
            "val_mean": f"{np.mean(val_accs):.4f}",
            "val_std": f"{np.std(val_accs):.4f}",
            "test_mean": f"{np.mean(test_accs):.4f}",
            "test_std": f"{np.std(test_accs):.4f}",
        })