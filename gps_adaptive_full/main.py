"""
Adaptive GTSP – Token + Layer + Head  (main.py)
================================================
FIX for constant head gate collapse:
  - Replaced head_entropy (too weak) with two targeted losses:
      1. head_sparsity_loss  – penalises mean head gate directly,
                               forcing BudgetNet to suppress heads
      2. head_diversity_loss – penalises low variance of head gates
                               across graphs in the batch,
                               directly punishing the constant-collapse
      3. Removed head_diversity via negative var (numerically unstable)
         replaced with a cleaner MSE-based diversity term
  - New args: --lambda_head_sparse, --lambda_head_div
  - head_prior_mix and head_prior_temp passed to BudgetNet
"""

import os
import random
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
# Args
# =====================================================================
def gene_arg():
    parser = configargparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--configs',    required=False, is_config_file=True)
    parser.add_argument('--data_root',  type=str,   default='../data')
    parser.add_argument('--dataset',    type=str,   default='DD')

    g = parser.add_argument_group('model')
    g.add_argument('--model_type',    type=str,   default='adaptive_gps_full')
    g.add_argument('--graph_pooling', type=str,   default='mean')
    g.add_argument('--gnn_type',      type=str,   default='gcn')
    g.add_argument('--gnn_dropout',   type=float, default=0)
    g.add_argument('--num_layers',    type=int,   default=4)
    g.add_argument('--gnn_emb_dim',   type=int,   default=64)
    g.add_argument('--nhead',         type=int,   default=4)

    g = parser.add_argument_group('training')
    g.add_argument('--devices',         type=int,   default=0)
    g.add_argument('--batch_size',      type=int,   default=4)
    g.add_argument('--eval_batch_size', type=int,   default=None)
    g.add_argument('--epochs',          type=int,   default=100)
    g.add_argument('--num_workers',     type=int,   default=0)
    g.add_argument('--weight_decay',    type=float, default=1e-5)
    g.add_argument('--lr',              type=float, default=0.001)
    g.add_argument('--runs',            type=int,   default=10)
    g.add_argument('--seed',            type=int,   default=12344)

    g = parser.add_argument_group('adaptive')
    g.add_argument('--lambda_compute',    type=float, default=0.5)
    g.add_argument('--lambda_ratio',      type=float, default=0.1)
    g.add_argument('--lambda_head_sparse',type=float, default=0.2,
                   help='penalise mean head gate → forces actual suppression')
    g.add_argument('--lambda_head_div',   type=float, default=0.3,
                   help='penalise low head gate variance across graphs → breaks collapse')
    g.add_argument('--target_ratio',      type=float, default=0.7)
    g.add_argument('--target_head_gate',  type=float, default=0.6,
                   help='desired average head gate (sparsity target)')
    g.add_argument('--tau_start',         type=float, default=2.0)
    g.add_argument('--tau_end',           type=float, default=0.5)
    g.add_argument('--min_token_ratio',   type=float, default=0.40)
    g.add_argument('--max_token_ratio',   type=float, default=0.70)
    g.add_argument('--min_head_gate',     type=float, default=0.05)
    g.add_argument('--size_prior_mix',    type=float, default=0.45)
    g.add_argument('--size_prior_temp',   type=float, default=3.0)
    g.add_argument('--head_prior_mix',    type=float, default=0.3,
                   help='blend weight for graph-size prior on head gates')
    g.add_argument('--head_prior_temp',   type=float, default=2.0,
                   help='sharpness of head size prior')
    g.add_argument('--budget_hidden',     type=int,   default=64)
    g.add_argument('--max_k',             type=int,   default=512)

    g = parser.add_argument_group('cross-validation')
    g.add_argument('--n_folds', type=int, default=10)

    args, _ = parser.parse_known_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    return args


# =====================================================================
# Data
# =====================================================================
class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.x = ((deg - self.mean) / self.std).view(-1, 1)
        return data


def load_dataset(args, transform):
    data_name = args.dataset + '-pe'
    dataset   = TUDataset(os.path.join(args.data_root, data_name),
                          name=args.dataset, pre_transform=transform)
    if dataset.data.x is None:
        max_degree, degs = 0, []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            dataset.transform = NormalizedDegree(
                deg.mean().item(), deg.std().item())
    return dataset, dataset.num_classes, dataset.num_features


def get_labels(dataset):
    return np.array([int(d.y.item()) for d in dataset])


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
            nn_mod = Sequential(Linear(channels, channels), ReLU(),
                                Linear(channels, channels))
            self.convs.append(
                GPSConv(channels, GINConv(nn_mod),
                        heads=args.nhead, attn_dropout=0.5,
                        max_k=args.max_k))

        self.budget_net = BudgetNet(
            channels=channels,
            num_layers=num_layers,
            num_heads=args.nhead,
            hidden_dim=args.budget_hidden,
            min_token_ratio=args.min_token_ratio,
            max_token_ratio=args.max_token_ratio,
            min_head_gate=args.min_head_gate,
            size_prior_mix=args.size_prior_mix,
            size_prior_temp=args.size_prior_temp,
            head_prior_mix=args.head_prior_mix,       # ← NEW
            head_prior_temp=args.head_prior_temp,     # ← NEW
        )

        self.lin = Linear(channels, num_tasks)

    def forward(self, x, pe, edge_index, edge_attr, batch, tau=1.0):
        x = self.node_emb(x) + self.pe_lin(pe)

        token_ratios, layer_gates, head_gates = self.budget_net(
            x, edge_index, batch, node_emb=x)

        compute_costs     = []
        total_actual_macs = 0.0
        total_dense_macs  = 0.0

        for i, conv in enumerate(self.convs):
            tr = token_ratios[:, i]
            lg = layer_gates[:, i]
            hg = head_gates[:, i, :]

            x = conv(x, edge_index, batch,
                     token_ratio=tr, layer_gate=lg, head_gate=hg, tau=tau)

            avg_hg     = hg.mean(dim=-1)
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
# Train
# =====================================================================
def train_one_epoch(model, loader, optimizer, device, tau, args):
    model.train()
    total_loss = total_task = total_compute = total_ratio = 0.0
    total_hsp  = total_hdiv = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        (logits, compute_cost,
         tok_ratios, lay_gates, head_gates,
         _, _) = model(data.x, data.pe, data.edge_index,
                       data.edge_attr, data.batch, tau=tau)
        # print("\n--- HEAD GATES SAMPLE ---")
        # print(head_gates[:2])
        # print("-------------------------\n")

        N_per_graph  = torch.bincount(data.batch).float()

        # Task loss
        task_loss_pg = F.cross_entropy(logits, data.y, reduction='none')
        task_loss    = (task_loss_pg / torch.log1p(N_per_graph)).mean()

        # Compute loss (token + layer + head)
        node_weights = N_per_graph / N_per_graph.mean()
        avg_hg       = head_gates.mean(dim=-1)
        compute_loss = (tok_ratios ** 2 * lay_gates * avg_hg
                        * node_weights.unsqueeze(1)).mean()

        # Token ratio regulariser
        ratio_loss = (tok_ratios.mean() - args.target_ratio) ** 2

        # ── Head gate losses (FIX) ─────────────────────────────────────────
        # 1. Head sparsity loss
        #    Penalise deviation from target_head_gate.
        #    Pushes BudgetNet to use a specific suppression level per graph
        #    rather than settling at an arbitrary constant.
        head_sparsity_loss = (head_gates.mean() - args.target_head_gate) ** 2

        # 2. Head diversity loss
        #    Penalise low variance of avg head gate ACROSS graphs in batch.
        #    head_gates: [B, L, H] → mean over L,H → [B] (one scalar per graph)
        #    We want this [B] vector to have HIGH variance (different per graph).
        #    Loss = -var → minimising it maximises variance → forces diversity.
        #    Clamp batch size ≥ 2 to avoid degenerate single-graph batches.
        B = head_gates.size(0)
        if B >= 2:
            per_graph_hg = head_gates.mean(dim=[1, 2])  # [B]
            head_diversity_loss = -per_graph_hg.var(unbiased=False)
        else:
            head_diversity_loss = torch.tensor(0.0, device=device)

        loss = (task_loss
                + args.lambda_compute     * compute_loss
                + args.lambda_ratio       * ratio_loss
                + args.lambda_head_sparse * head_sparsity_loss
                + args.lambda_head_div    * head_diversity_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        Bg = data.num_graphs
        total_loss    += loss.item()          * Bg
        total_task    += task_loss.item()     * Bg
        total_compute += compute_cost.item()  * Bg
        total_ratio   += ratio_loss.item()    * Bg
        total_hsp     += head_sparsity_loss.item() * Bg
        total_hdiv    += head_diversity_loss.item() * Bg

    n = len(loader.dataset)
    return (total_loss / n, total_task / n, total_compute / n,
            total_ratio / n, total_hsp / n, total_hdiv / n)


# =====================================================================
# Evaluate
# =====================================================================
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
        total_head_gates.append(head_gates.mean().item())
        sum_actual_macs += actual_macs
        sum_dense_macs  += dense_macs

        if return_details:
            nodes_per_graph = data.batch.bincount().cpu().numpy()
            true_labels     = data.y.cpu().numpy()
            pred_labels     = pred.cpu().numpy()
            tr_np  = tok_ratios.cpu().numpy()
            lg_np  = lay_gates.cpu().numpy()
            hg_np  = head_gates.cpu().numpy()

            batch_cpu      = data.batch.cpu()
            edge_index_cpu = data.edge_index.cpu()

            edges_per_graph = torch.zeros(len(nodes_per_graph), dtype=torch.long)
            for edge_src, edge_dst in edge_index_cpu.t():
                edges_per_graph[batch_cpu[edge_src.item()]] += 1
            edges_per_graph = edges_per_graph.numpy() / 2.0

            from torch_geometric.utils import degree as compute_degree
            deg      = compute_degree(edge_index_cpu[0],
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
                density = (2.0 * num_e / (num_n * (num_n - 1) + 1e-8)
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
                    "avg_head_gate":   float(hg_np[i].mean()),
                    "correct":         int(true_labels[i] == pred_labels[i]),
                })

    acc        = correct / len(loader.dataset)
    avg_ratio  = float(np.mean(total_k_ratios))
    avg_hg_val = float(np.mean(total_head_gates))
    flop_red   = 1.0 - (sum_actual_macs / (sum_dense_macs + 1e-12))

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
    print(f"  FOLD {fold_idx+1}/{args.n_folds}  "
          f"Train:{len(train_indices)}  Val:{len(val_indices)}  Test:{len(test_indices)}")
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

        (loss, task_l, comp_l,
         ratio_l, hsp_l, hdiv_l) = train_one_epoch(
            model, train_loader, optimizer, device, tau, args)

        val_acc,  val_ratio,  val_hg,  val_flop  = evaluate(
            model, val_loader,  device, tau)
        test_acc, test_ratio, test_hg, test_flop = evaluate(
            model, test_loader, device, tau)

        epoch_log.append({
            "epoch":           epoch,
            "loss":            f"{loss:.6f}",
            "task_loss":       f"{task_l:.6f}",
            "compute_loss":    f"{comp_l:.6f}",
            "head_sparse_loss":f"{hsp_l:.6f}",
            "head_div_loss":   f"{hdiv_l:.6f}",
            "val_acc":         f"{val_acc:.4f}",
            "test_acc":        f"{test_acc:.4f}",
            "avg_token_ratio": f"{val_ratio:.4f}",
            "avg_head_gate":   f"{val_hg:.4f}",
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
                  f'hsp {hsp_l:.4f} hdiv {hdiv_l:.4f}) | '
                  f'Val {val_acc:.4f} | Test {test_acc:.4f} | '
                  f'ratio {val_ratio:.3f} | hg {val_hg:.3f} | '
                  f'FLOP {val_flop:.1%} | tau {tau:.2f}')

    # Save epoch log
    epoch_log_file = os.path.join(fold_save_dir, "epoch_log.csv")
    with open(epoch_log_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_log)

    # Load best → final eval
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

    print(f"Dataset:{args.dataset}  Graphs:{len(dataset)}  "
          f"Classes:{num_tasks}  Features:{num_features}  Folds:{args.n_folds}")

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
            "avg_head_gate":   fold_hg,
            "flop_reduction":  fold_flop,
        })

    results_cv_to_file(args, fold_results)


if __name__ == '__main__':
    main()