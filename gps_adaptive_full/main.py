"""
Adaptive GTSP – Token + Layer + Head  (main.py)
================================================
Trains a GPS model with graph-conditioned adaptive token pruning,
layer gating, AND head gating on TUDataset graph classification tasks.

Changes over gps_adaptive version:
  1. BudgetNet now returns head_gates [B, L, H] in addition to
     token_ratios and layer_gates.
  2. GPSConv receives head_gate [B, H] per layer.
  3. Compute loss now includes avg head gate:
       layer_cost = tr² * lg * avg_head_gate_per_layer
  4. New args: --min_head_gate, --lambda_head
  5. head gate logged in epoch_log, graph_stats CSV, and fold results.
  6. utils.results_cv_to_file receives avg_head_gate per fold.
"""

import os
import os.path as osp
import random
import math
import csv

import numpy as np
import configargparse

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.utils.data import Subset

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import degree

from sklearn.model_selection import StratifiedKFold

from gps_conv import GPSConv
from adaptive_model import BudgetNet
from utils import (results_cv_to_file,
                   num_total_parameters, num_trainable_parameters)

from datetime import datetime

now = datetime.now().strftime("%m_%d-%H_%M_%S")


# =====================================================================
# Argument parser
# =====================================================================
def gene_arg():
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description='Adaptive GTSP (Token+Layer+Head) on GPS + TUDataset')
    parser.add_argument('--configs', required=False, is_config_file=True)

    parser.add_argument('--data_root',  type=str, default='../data')
    parser.add_argument('--dataset',    type=str, default='DD',
                        help='TUDataset name (default: DD)')

    group = parser.add_argument_group('model')
    group.add_argument('--model_type',    type=str,   default='adaptive_gps_full')
    group.add_argument('--graph_pooling', type=str,   default='mean')
    group.add_argument('--gnn_type',      type=str,   default='gcn')
    group.add_argument('--gnn_dropout',   type=float, default=0)
    group.add_argument('--num_layers',    type=int,   default=4)
    group.add_argument('--gnn_emb_dim',   type=int,   default=64)
    group.add_argument('--nhead',         type=int,   default=4,
                        help='number of attention heads (default: 4)')

    group = parser.add_argument_group('training')
    group.add_argument('--devices',         type=int,   default=0)
    group.add_argument('--batch_size',      type=int,   default=4)
    group.add_argument('--eval_batch_size', type=int,   default=None)
    group.add_argument('--epochs',          type=int,   default=100)
    group.add_argument('--num_workers',     type=int,   default=0)
    group.add_argument('--weight_decay',    type=float, default=1e-5)
    group.add_argument('--lr',              type=float, default=0.001)
    group.add_argument('--runs',            type=int,   default=10)
    group.add_argument('--seed',            type=int,   default=12344)

    group = parser.add_argument_group('adaptive')
    group.add_argument('--lambda_compute', type=float, default=0.5,
                        help='weight for compute-cost penalty')
    group.add_argument('--lambda_ratio',   type=float, default=0.1,
                        help='weight for target-ratio regularizer')
    group.add_argument('--lambda_head',    type=float, default=0.1,
                        help='weight for head-gate entropy regularizer (NEW)')
    group.add_argument('--target_ratio',   type=float, default=0.7,
                        help='desired average token keep-ratio')
    group.add_argument('--tau_start',      type=float, default=2.0)
    group.add_argument('--tau_end',        type=float, default=0.5)
    group.add_argument('--min_token_ratio',type=float, default=0.40)
    group.add_argument('--max_token_ratio',type=float, default=0.70)
    group.add_argument('--min_head_gate',  type=float, default=0.0,
                        help='minimum head gate value output by BudgetNet (NEW)')
    group.add_argument('--size_prior_mix', type=float, default=0.45)
    group.add_argument('--size_prior_temp',type=float, default=3.0)
    group.add_argument('--budget_hidden',  type=int,   default=64)
    group.add_argument('--max_k',          type=int,   default=512)

    group = parser.add_argument_group('cross-validation')
    group.add_argument('--n_folds', type=int, default=10)

    args, _ = parser.parse_known_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    return args


# =====================================================================
# Data loading
# =====================================================================
class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def load_dataset(args, transform):
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
    return np.array([int(data.y.item()) for data in dataset])


# =====================================================================
# Model
# =====================================================================
class AdaptiveGPS(torch.nn.Module):
    def __init__(self, fea_dim, channels, num_layers, num_tasks, args):
        super().__init__()
        self.num_layers = num_layers
        self.nhead      = args.nhead

        self.node_emb = Linear(fea_dim, channels)
        self.pe_lin   = Linear(20, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn_mod = Sequential(
                Linear(channels, channels), ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINConv(nn_mod),
                           heads=args.nhead, attn_dropout=0.5,
                           max_k=args.max_k)
            self.convs.append(conv)

        # BudgetNet now takes num_heads and min_head_gate  ← NEW
        self.budget_net = BudgetNet(
            channels=channels,
            num_layers=num_layers,
            num_heads=args.nhead,                    # ← NEW
            hidden_dim=args.budget_hidden,
            min_token_ratio=args.min_token_ratio,
            max_token_ratio=args.max_token_ratio,
            min_head_gate=args.min_head_gate,        # ← NEW
            size_prior_mix=args.size_prior_mix,
            size_prior_temp=args.size_prior_temp,
        )

        self.lin = Linear(channels, num_tasks)

    def forward(self, x, pe, edge_index, edge_attr, batch, tau=1.0):
        x = self.node_emb(x) + self.pe_lin(pe)

        # BudgetNet returns 3 tensors now
        token_ratios, layer_gates, head_gates = self.budget_net(
            x, edge_index, batch, node_emb=x)
        # token_ratios : [B, L]
        # layer_gates  : [B, L]
        # head_gates   : [B, L, H]

        compute_costs     = []
        total_actual_macs = 0.0
        total_dense_macs  = 0.0

        for i, conv in enumerate(self.convs):
            tr = token_ratios[:, i]       # [B]
            lg = layer_gates[:, i]        # [B]
            hg = head_gates[:, i, :]      # [B, H]  ← NEW

            x = conv(x, edge_index, batch,
                     token_ratio=tr,
                     layer_gate=lg,
                     head_gate=hg,        # ← NEW
                     tau=tau)

            # Compute loss: tr² * lg * mean(hg)
            # avg head gate across heads gives a scalar per graph
            avg_hg    = hg.mean(dim=-1)   # [B]
            layer_cost = (tr ** 2 * lg * avg_hg).mean()
            compute_costs.append(layer_cost)

            total_actual_macs += conv._actual_macs
            total_dense_macs  += conv._dense_macs

        x      = global_add_pool(x, batch)
        logits = self.lin(x)

        avg_compute = sum(compute_costs) / len(compute_costs)
        return (logits, avg_compute,
                token_ratios, layer_gates, head_gates,
                total_actual_macs, total_dense_macs)


# =====================================================================
# Train / Eval
# =====================================================================
def train_one_epoch(model, loader, optimizer, device, tau,
                    lambda_compute, lambda_ratio, lambda_head,
                    target_ratio):
    model.train()
    total_loss = total_task = total_compute = total_ratio = total_head = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        (logits, compute_cost,
         tok_ratios, lay_gates, head_gates,
         _, _) = model(data.x, data.pe, data.edge_index,
                       data.edge_attr, data.batch, tau=tau)

        N_per_graph  = torch.bincount(data.batch).float()   # [B]

        # Task loss (size-normalised)
        task_loss_pg = F.cross_entropy(logits, data.y, reduction='none')
        task_loss    = (task_loss_pg / torch.log1p(N_per_graph)).mean()

        # Compute loss: token + layer + head  (node-count weighted)
        node_weights = N_per_graph / N_per_graph.mean()      # [B]
        avg_hg       = head_gates.mean(dim=-1)               # [B, L]
        compute_loss = (tok_ratios ** 2
                        * lay_gates
                        * avg_hg                             # ← head term
                        * node_weights.unsqueeze(1)).mean()

        # Token ratio regulariser
        ratio_loss = (tok_ratios.mean() - target_ratio) ** 2

        # Head entropy regulariser: encourage BudgetNet to make decisions
        # (push gates away from 0.5 → either use a head or suppress it)
        # H(p) = -p*log(p) - (1-p)*log(1-p);  we minimise entropy → sharp gates
        hg_clamped = head_gates.clamp(1e-6, 1.0 - 1e-6)
        head_entropy = -(hg_clamped * hg_clamped.log()
                         + (1 - hg_clamped) * (1 - hg_clamped).log()).mean()

        loss = (task_loss
                + lambda_compute * compute_loss
                + lambda_ratio   * ratio_loss
                + lambda_head    * head_entropy)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        B = data.num_graphs
        total_loss    += loss.item()         * B
        total_task    += task_loss.item()    * B
        total_compute += compute_cost.item() * B
        total_ratio   += ratio_loss.item()   * B
        total_head    += head_entropy.item() * B

    n = len(loader.dataset)
    return (total_loss / n, total_task / n,
            total_compute / n, total_ratio / n, total_head / n)


@torch.no_grad()
def evaluate(model, loader, device, tau=0.5, return_details=False):
    model.eval()
    correct          = 0
    total_k_ratios   = []
    total_head_gates = []
    sum_actual_macs  = 0.0
    sum_dense_macs   = 0.0
    details          = []

    for data in loader:
        data = data.to(device)
        (logits, _, tok_ratios, lay_gates, head_gates,
         actual_macs, dense_macs) = model(
            data.x, data.pe, data.edge_index,
            data.edge_attr, data.batch, tau=tau)

        pred     = logits.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total_k_ratios.append(tok_ratios.mean().item())
        total_head_gates.append(head_gates.mean().item())   # ← NEW
        sum_actual_macs += actual_macs
        sum_dense_macs  += dense_macs

        if return_details:
            nodes_per_graph = data.batch.bincount().cpu().numpy()
            true_labels     = data.y.cpu().numpy()
            pred_labels     = pred.cpu().numpy()
            tr_np  = tok_ratios.cpu().numpy()    # [B, L]
            lg_np  = lay_gates.cpu().numpy()     # [B, L]
            hg_np  = head_gates.cpu().numpy()    # [B, L, H]

            batch_cpu      = data.batch.cpu()
            edge_index_cpu = data.edge_index.cpu()

            edges_per_graph = torch.zeros(len(nodes_per_graph), dtype=torch.long)
            for edge_src, edge_dst in edge_index_cpu.t():
                edges_per_graph[batch_cpu[edge_src.item()]] += 1
            edges_per_graph = edges_per_graph.numpy() / 2.0

            from torch_geometric.utils import degree as compute_degree
            deg = compute_degree(edge_index_cpu[0],
                                 num_nodes=data.x.size(0)).cpu().numpy()
            deg_mean = np.zeros(len(nodes_per_graph))
            deg_var  = np.zeros(len(nodes_per_graph))
            for g_id in range(len(nodes_per_graph)):
                g_mask = (batch_cpu == g_id).numpy()
                if g_mask.sum() > 0:
                    g_degs = deg[g_mask]
                    deg_mean[g_id] = float(g_degs.mean())
                    deg_var[g_id]  = float(g_degs.var()) if len(g_degs) > 1 else 0.0

            for i in range(len(true_labels)):
                num_n   = int(nodes_per_graph[i])
                num_e   = edges_per_graph[i]
                density = ((2.0 * num_e) / (num_n * (num_n - 1) + 1e-8)
                           if num_n > 1 else 0.0)
                details.append({
                    "num_nodes":       int(num_n),
                    "num_edges":       int(num_e),
                    "density":         float(density),
                    "avg_degree":      float(deg_mean[i]),
                    "degree_variance": float(deg_var[i]),
                    "true_label":      int(true_labels[i]),
                    "pred_label":      int(pred_labels[i]),
                    "avg_token_ratio": float(tr_np[i].mean()),
                    "avg_layer_gate":  float(lg_np[i].mean()),
                    "avg_head_gate":   float(hg_np[i].mean()),  # ← NEW
                    "correct":         int(true_labels[i] == pred_labels[i]),
                })

    acc          = correct / len(loader.dataset)
    avg_ratio    = float(np.mean(total_k_ratios))
    avg_hg_val   = float(np.mean(total_head_gates))            # ← NEW
    flop_red     = 1.0 - (sum_actual_macs / (sum_dense_macs + 1e-12))

    if return_details:
        return acc, avg_ratio, avg_hg_val, flop_red, details
    return acc, avg_ratio, avg_hg_val, flop_red


# =====================================================================
# Single fold
# =====================================================================
def train_single_fold(args, fold_idx, train_indices, test_indices,
                      dataset, num_tasks, num_features, device):

    train_indices = train_indices.copy()
    np.random.shuffle(train_indices)
    val_split     = int(len(train_indices) / 9)
    val_indices   = train_indices[:val_split]
    train_indices = train_indices[val_split:]

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx+1} / {args.n_folds}")
    print(f"  Train: {len(train_indices)}  Val: {len(val_indices)}  Test: {len(test_indices)}")
    print(f"{'='*60}")

    train_loader = DataLoader(Subset(dataset, train_indices.tolist()),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_indices.tolist()),
                              batch_size=args.eval_batch_size)
    test_loader  = DataLoader(Subset(dataset, test_indices.tolist()),
                              batch_size=args.eval_batch_size)

    model = AdaptiveGPS(
        fea_dim=num_features, channels=args.gnn_emb_dim,
        num_layers=args.num_layers, num_tasks=num_tasks, args=args,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fold_save_dir = os.path.join(args.save_path, f"fold_{fold_idx}")
    os.makedirs(fold_save_dir, exist_ok=True)

    best_val_acc = 0.0
    epoch_log    = []

    for epoch in range(1, args.epochs + 1):
        frac = (epoch - 1) / max(args.epochs - 1, 1)
        tau  = args.tau_start + (args.tau_end - args.tau_start) * frac

        loss, task_l, comp_l, ratio_l, head_l = train_one_epoch(
            model, train_loader, optimizer, device, tau,
            args.lambda_compute, args.lambda_ratio,
            args.lambda_head, args.target_ratio)

        val_acc,  val_ratio,  val_hg,  val_flop  = evaluate(
            model, val_loader,  device, tau)
        test_acc, test_ratio, test_hg, test_flop = evaluate(
            model, test_loader, device, tau)

        epoch_log.append({
            "epoch":           epoch,
            "loss":            f"{loss:.6f}",
            "task_loss":       f"{task_l:.6f}",
            "compute_loss":    f"{comp_l:.6f}",
            "head_entropy":    f"{head_l:.6f}",      # ← NEW
            "val_acc":         f"{val_acc:.4f}",
            "test_acc":        f"{test_acc:.4f}",
            "avg_token_ratio": f"{val_ratio:.4f}",
            "avg_head_gate":   f"{val_hg:.4f}",      # ← NEW
            "flop_reduction":  f"{val_flop:.4f}",
            "tau":             f"{tau:.4f}",
        })

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(fold_save_dir, "best_model.pt"))

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f'  Fold {fold_idx+1} | Ep {epoch:03d} | '
                  f'Loss {loss:.4f} (task {task_l:.4f} comp {comp_l:.4f} '
                  f'ratio {ratio_l:.4f} head {head_l:.4f}) | '
                  f'Val {val_acc:.4f} | Test {test_acc:.4f} | '
                  f'tau {tau:.2f} | ratio {val_ratio:.3f} | '
                  f'hg {val_hg:.3f} | FLOP {val_flop:.1%}')

    # Save epoch log
    epoch_log_file = os.path.join(fold_save_dir, "epoch_log.csv")
    with open(epoch_log_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_log)

    # Load best model → final eval
    model.load_state_dict(
        torch.load(os.path.join(fold_save_dir, "best_model.pt"),
                   weights_only=True))

    (final_test_acc, final_test_ratio,
     final_test_hg, final_test_flop, graph_details) = evaluate(
        model, test_loader, device, tau=args.tau_end, return_details=True)

    final_val_acc, _, _, _ = evaluate(
        model, val_loader, device, tau=args.tau_end)

    print(f"  Fold {fold_idx+1} BEST => "
          f"Val {final_val_acc:.4f} | Test {final_test_acc:.4f} | "
          f"Ratio {final_test_ratio:.3f} | HeadGate {final_test_hg:.3f} | "
          f"FLOP ↓{final_test_flop:.1%}")

    # Save per-graph stats CSV
    stats_file = os.path.join(fold_save_dir, "graph_stats.csv")
    with open(stats_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=graph_details[0].keys())
        writer.writeheader()
        writer.writerows(graph_details)

    return (final_val_acc, final_test_acc,
            final_test_ratio, final_test_hg, final_test_flop)


# =====================================================================
# Main
# =====================================================================
def main():
    args = gene_arg()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    dataset, num_tasks, num_features = load_dataset(args, transform)
    labels = get_labels(dataset)

    print(f"Dataset: {args.dataset}  |  Graphs: {len(dataset)}  |  "
          f"Classes: {num_tasks}  |  Features: {num_features}  |  "
          f"Folds: {args.n_folds}")

    device = torch.device(
        f'cuda:{args.devices}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    run_name       = f"{args.dataset}_adaptive_full_cv{args.n_folds}"
    args.save_path = f"exps/{run_name}-{now}"
    os.makedirs(args.save_path, exist_ok=True)

    skf          = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                   random_state=args.seed)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(dataset)), labels)):

        (fold_val_acc, fold_test_acc,
         fold_ratio, fold_hg, fold_flop) = train_single_fold(
            args, fold_idx, train_idx, test_idx,
            dataset, num_tasks, num_features, device)

        fold_results.append({
            "fold":            fold_idx + 1,
            "val_acc":         fold_val_acc,
            "test_acc":        fold_test_acc,
            "avg_token_ratio": fold_ratio,
            "avg_head_gate":   fold_hg,       # ← NEW
            "flop_reduction":  fold_flop,
        })

    results_cv_to_file(args, fold_results)


if __name__ == '__main__':
    main()