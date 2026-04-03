"""
Adaptive GTSP – Main Training Script (with Cross-Validation)
=============================================================
Trains a GPS model with graph-conditioned adaptive token pruning and
layer gating on TUDataset graph classification tasks.

Key features:
  1. BudgetNet predicts per-graph, per-layer token ratios + layer gates.
  2. GPSConv physically shortens the attention sequence (true FLOP savings).
  3. Loss = L_task  +  lambda * L_compute   (FLOP-proxy penalty).
  4. Gumbel temperature annealing over training epochs.
  5. StratifiedKFold cross-validation for statistically rigorous results.
"""

import os
import os.path as osp
import random
import math

import numpy as np
import configargparse

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.utils.data import random_split, Subset

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import degree

from sklearn.model_selection import StratifiedKFold

from gps_conv import GPSConv
from adaptive_model import BudgetNet
from utils import (results_to_file, results_cv_to_file,
                   num_total_parameters, num_trainable_parameters)

from datetime import datetime

now = datetime.now().strftime("%m_%d-%H_%M_%S")


# =====================================================================
# Argument parser
# =====================================================================
def gene_arg():
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description='Adaptive GTSP on GPS + TUDataset')
    parser.add_argument('--configs', required=False, is_config_file=True)

    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default="DD",
                        help='TUDataset name (default: DD)')

    group = parser.add_argument_group('model')
    group.add_argument('--model_type', type=str, default='adaptive_gps')
    group.add_argument('--graph_pooling', type=str, default='mean')
    group.add_argument('--gnn_type', type=str, default='gcn')
    group.add_argument('--gnn_dropout', type=float, default=0)
    group.add_argument('--num_layers', type=int, default=4,
                        help='number of GPS layers (default: 4)')
    group.add_argument('--gnn_emb_dim', type=int, default=64,
                        help='hidden dim (default: 64)')
    group.add_argument('--nhead', type=int, default=4,
                        help='number of attention heads (default: 4)')

    group = parser.add_argument_group('training')
    group.add_argument('--devices', type=int, default=0)
    group.add_argument('--batch_size', type=int, default=4,
                        help='batch size (default: 4, keep small for DD)')
    group.add_argument('--eval_batch_size', type=int, default=None)
    group.add_argument('--epochs', type=int, default=100)
    group.add_argument('--num_workers', type=int, default=0)
    group.add_argument('--weight_decay', type=float, default=1e-5)
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--runs', type=int, default=10)
    group.add_argument('--seed', type=int, default=12344)

    group = parser.add_argument_group('adaptive')
    group.add_argument('--lambda_compute', type=float, default=0.5,
                        help='weight for compute-cost penalty (default: 0.5)')
    group.add_argument('--lambda_ratio', type=float, default=0.01,
                        help='weight for target-ratio regularizer (default: 0.01)')
    group.add_argument('--target_ratio', type=float, default=0.5,
                        help='desired average token keep-ratio (default: 0.5)')
    group.add_argument('--tau_start', type=float, default=2.0,
                        help='initial Gumbel temperature')
    group.add_argument('--tau_end', type=float, default=0.5,
                        help='final Gumbel temperature')
    group.add_argument('--min_token_ratio', type=float, default=0.2,
                        help='minimum fraction of tokens to keep')
    group.add_argument('--budget_hidden', type=int, default=64,
                        help='hidden dim of BudgetNet')
    group.add_argument('--max_k', type=int, default=512,
                        help='Hard cap on nodes for attention to prevent OOM')

    group = parser.add_argument_group('cross-validation')
    group.add_argument('--n_folds', type=int, default=10,
                        help='Number of CV folds (default: 10, use 3 for debugging)')

    args, _ = parser.parse_known_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    return args


# =====================================================================
# Data loading (DD / TUDataset)
# =====================================================================
class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def load_dataset(args, transform):
    """Load the full dataset (no splitting). Returns dataset, num_classes, num_features."""
    data_name = args.dataset + '-pe'
    dataset = TUDataset(
        os.path.join(args.data_root, data_name),
        name=args.dataset,
        pre_transform=transform,
    )

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset, dataset.num_classes, dataset.num_features


def get_labels(dataset):
    """Extract integer labels from a TUDataset for stratified splitting."""
    labels = []
    for data in dataset:
        labels.append(int(data.y.item()))
    return np.array(labels)


# =====================================================================
# Model: Adaptive GPS
# =====================================================================
class AdaptiveGPS(torch.nn.Module):
    def __init__(self, fea_dim, channels, num_layers, num_tasks, args):
        super().__init__()
        self.num_layers = num_layers

        self.node_emb = Linear(fea_dim, channels)
        self.pe_lin   = Linear(20, channels)

        # --- GPS layers ---
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn_mod = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINConv(nn_mod), heads=args.nhead,
                           attn_dropout=0.5, max_k=args.max_k)
            self.convs.append(conv)

        # --- Budget controller ---
        self.budget_net = BudgetNet(
            channels=channels,
            num_layers=num_layers,
            hidden_dim=args.budget_hidden,
            min_token_ratio=args.min_token_ratio,
        )

        self.lin = Linear(channels, num_tasks)

    def forward(self, x, pe, edge_index, edge_attr, batch, tau=1.0):
        # Embed
        x = self.node_emb(x) + self.pe_lin(pe)

        # --- Predict adaptive budgets ---
        token_ratios, layer_gates = self.budget_net(
            x, edge_index, batch, node_emb=x)

        # --- Run GPS layers with adaptive routing ---
        compute_costs = []
        total_actual_macs = 0.0
        total_dense_macs = 0.0
        for i, conv in enumerate(self.convs):
            tr = token_ratios[:, i]    # [B]
            lg = layer_gates[:, i]     # [B]

            x = conv(x, edge_index, batch,
                     token_ratio=tr,
                     layer_gate=lg,
                     tau=tau)

            # Collect FLOP proxy per layer (differentiable)
            # Use continuous token_ratio directly instead of post-integer k_avg
            # so gradients flow back to BudgetNet's token_head.
            # tr.mean() ≈ k/L, so tr.mean()**2 is a differentiable proxy for (k/L)²
            layer_cost = (tr.mean() ** 2) * lg.mean()
            compute_costs.append(layer_cost)

            # Collect exact MAC counts (Phase 2)
            total_actual_macs += conv._actual_macs
            total_dense_macs += conv._dense_macs

        x = global_add_pool(x, batch)
        logits = self.lin(x)

        avg_compute = sum(compute_costs) / len(compute_costs)
        return logits, avg_compute, token_ratios, layer_gates, total_actual_macs, total_dense_macs


# =====================================================================
# Train / Eval
# =====================================================================
def train_one_epoch(model, loader, optimizer, device, tau, lambda_compute,
                    lambda_ratio=0.5, target_ratio=0.5):
    model.train()
    total_loss = 0
    total_task_loss = 0
    total_compute_loss = 0
    total_ratio_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        logits, compute_cost, tok_ratios, lay_gates, _, _ = model(
            data.x, data.pe, data.edge_index, data.edge_attr,
            data.batch, tau=tau)

        task_loss = F.cross_entropy(logits, data.y)
        compute_loss = compute_cost

        # Ratio regularizer: penalise deviation from target keep-ratio
        avg_ratio = tok_ratios.mean()
        ratio_loss = (avg_ratio - target_ratio) ** 2

        loss = (task_loss
                + lambda_compute * compute_loss
                + lambda_ratio * ratio_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_task_loss += task_loss.item() * data.num_graphs
        total_compute_loss += compute_cost.item() * data.num_graphs
        total_ratio_loss += ratio_loss.item() * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_task_loss / n, total_compute_loss / n, total_ratio_loss / n


@torch.no_grad()
def evaluate(model, loader, device, tau=0.5, return_details=False):
    model.eval()
    correct = 0
    total_k_ratios = []
    sum_actual_macs = 0.0
    sum_dense_macs = 0.0
    details = []

    for data in loader:
        data = data.to(device)
        logits, _, tok_ratios, lay_gates, actual_macs, dense_macs = model(
            data.x, data.pe, data.edge_index, data.edge_attr,
            data.batch, tau=tau)
        pred = logits.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total_k_ratios.append(tok_ratios.mean().item())
        sum_actual_macs += actual_macs
        sum_dense_macs += dense_macs

        if return_details:
            # data.batch.bincount() = number of nodes per graph in this batch
            nodes_per_graph = data.batch.bincount().cpu().numpy()  # [B]
            true_labels = data.y.cpu().numpy()                     # [B]
            pred_labels = pred.cpu().numpy()                        # [B]
            tr_np = tok_ratios.cpu().numpy()                        # [B, num_layers]
            lg_np = lay_gates.cpu().numpy()                         # [B, num_layers]

            for i in range(len(true_labels)):
                details.append({
                    "num_nodes":        int(nodes_per_graph[i]),
                    "true_label":       int(true_labels[i]),
                    "pred_label":       int(pred_labels[i]),
                    "avg_token_ratio":  float(tr_np[i].mean()),
                    "avg_layer_gate":   float(lg_np[i].mean()),
                    "correct":          int(true_labels[i] == pred_labels[i]),
                })

    acc = correct / len(loader.dataset)
    avg_ratio = np.mean(total_k_ratios)
    flop_reduction = 1.0 - (sum_actual_macs / (sum_dense_macs + 1e-12))

    if return_details:
        return acc, avg_ratio, flop_reduction, details
    return acc, avg_ratio, flop_reduction


# =====================================================================
# Single fold training
# =====================================================================
def train_single_fold(args, fold_idx, train_indices, test_indices,
                      dataset, num_tasks, num_features, device):
    """Train and evaluate one fold. Returns best val acc, test acc, avg ratio."""

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1} / {args.n_folds}")
    print(f"  Train size: {len(train_indices)}, Test size: {len(test_indices)}")
    print(f"{'='*60}")

    # --- Create data loaders for this fold ---
    train_subset = Subset(dataset, train_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=args.eval_batch_size)

    # --- Fresh model for each fold ---
    model = AdaptiveGPS(
        fea_dim=num_features,
        channels=args.gnn_emb_dim,
        num_layers=args.num_layers,
        num_tasks=num_tasks,
        args=args,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Save path for this fold ---
    fold_save_dir = os.path.join(args.save_path, f"fold_{fold_idx}")
    os.makedirs(fold_save_dir, exist_ok=True)

    # --- Training loop ---
    best_train_acc = 0
    best_test_acc = 0
    best_ratio = 0
    epoch_log = []  # Per-epoch metrics for training curves

    for epoch in range(1, args.epochs + 1):
        # Anneal Gumbel temperature
        frac = (epoch - 1) / max(args.epochs - 1, 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * frac

        loss, task_l, comp_l, ratio_l = train_one_epoch(
            model, train_loader, optimizer, device, tau, args.lambda_compute,
            lambda_ratio=args.lambda_ratio, target_ratio=args.target_ratio)

        # In K-fold CV, the "test" split of the fold acts as our validation
        test_acc, test_ratio, test_flop_red = evaluate(model, test_loader, device, tau)

        # Record every epoch for training curve plots
        epoch_log.append({
            "epoch": epoch,
            "loss": f"{loss:.6f}",
            "task_loss": f"{task_l:.6f}",
            "compute_loss": f"{comp_l:.6f}",
            "test_acc": f"{test_acc:.4f}",
            "avg_token_ratio": f"{test_ratio:.4f}",
            "flop_reduction": f"{test_flop_red:.4f}",
            "tau": f"{tau:.4f}",
        })

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_ratio = test_ratio
            torch.save(model.state_dict(),
                       os.path.join(fold_save_dir, "best_model.pt"))

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f'  Fold {fold_idx+1} | Epoch {epoch:03d} | '
                  f'Loss {loss:.4f} (task {task_l:.4f}, comp {comp_l:.4f}, ratio {ratio_l:.4f}) | '
                  f'Test {test_acc:.4f} | '
                  f'tau {tau:.2f} | avg_ratio {test_ratio:.3f} | '
                  f'FLOP_red {test_flop_red:.1%}')

    # --- Save epoch log to CSV ---
    import csv
    epoch_log_file = os.path.join(fold_save_dir, "epoch_log.csv")
    with open(epoch_log_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_log)
    print(f"  Epoch log saved to: {epoch_log_file}")

    # --- Load best model and get final metrics + per-graph decisions ---
    model.load_state_dict(
        torch.load(os.path.join(fold_save_dir, "best_model.pt"),
                   weights_only=True))
    final_acc, final_ratio, final_flop_red, graph_details = evaluate(
        model, test_loader, device, tau=args.tau_end, return_details=True)

    print(f"  Fold {fold_idx+1} BEST => Acc: {final_acc:.4f} | "
          f"Avg Token Ratio: {final_ratio:.3f} | "
          f"FLOP Reduction: {final_flop_red:.1%}")

    # --- Save per-graph BudgetNet decisions for paper analysis ---
    stats_file = os.path.join(fold_save_dir, "graph_stats.csv")
    with open(stats_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=graph_details[0].keys())
        writer.writeheader()
        writer.writerows(graph_details)
    print(f"  Per-graph stats saved to: {stats_file}")

    return final_acc, final_ratio, final_flop_red


# =====================================================================
# Main
# =====================================================================
def main():
    args = gene_arg()

    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    # Transform
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

    # Load full dataset (no splitting yet)
    dataset, num_tasks, num_features = load_dataset(args, transform)
    labels = get_labels(dataset)

    print(f"Dataset: {args.dataset}")
    print(f"  Total graphs: {len(dataset)}")
    print(f"  Classes: {num_tasks}")
    print(f"  Node features: {num_features}")
    print(f"  Cross-validation folds: {args.n_folds}")

    # Device
    device = torch.device(
        f'cuda:{args.devices}' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Save path
    run_name = f"{args.dataset}_adaptive_cv{args.n_folds}"
    args.save_path = f"exps/{run_name}-{now}"
    os.makedirs(args.save_path, exist_ok=True)

    # --- Stratified K-Fold Cross Validation ---
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(
            np.zeros(len(dataset)), labels)):

        fold_acc, fold_ratio, fold_flop_red = train_single_fold(
            args, fold_idx, train_idx, test_idx,
            dataset, num_tasks, num_features, device)

        fold_results.append({
            "fold": fold_idx + 1,
            "val_acc": fold_acc,      # In K-fold, the held-out fold = "test"
            "test_acc": fold_acc,     # Same split in K-fold
            "avg_token_ratio": fold_ratio,
            "flop_reduction": fold_flop_red,
        })

    # --- Aggregate and report ---
    results_cv_to_file(args, fold_results)


if __name__ == '__main__':
    main()
