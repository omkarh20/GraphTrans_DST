import os.path as osp
import os

import torch
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, GINConv
from gps_conv import GPSConv
from torch_geometric.utils import degree
from torch.utils.data import random_split


import random
import numpy as np
import configargparse
import csv

import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from utils import results_to_file, results_cv_to_file



from datetime import datetime
now = datetime.now()
now = now.strftime("%m_%d-%H_%M_%S")



#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'zinc-pe')
transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
# train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
# val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
# test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)

def gene_arg():

    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                    description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)
    parser.add_argument('--wandb_run_idx', type=str, default=None)


    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--aug', type=str, default='baseline',
                        help='augment method to use [baseline|flag|augment]')

    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='maximum sequence length to predict (default: None)')

    group = parser.add_argument_group('model')
    group.add_argument('--model_type', type=str, default='gnn', help='gnn|pna|gnn-transformer')
    group.add_argument('--graph_pooling', type=str, default='mean')
    group = parser.add_argument_group('gnn')
    group.add_argument('--gnn_type', type=str, default='gcn')
    group.add_argument('--gnn_virtual_node', action='store_true')
    group.add_argument('--gnn_dropout', type=float, default=0)
    group.add_argument('--gnn_num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    group.add_argument('--gnn_emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    group.add_argument('--gnn_JK', type=str, default='last')
    group.add_argument('--gnn_residual', action='store_true', default=False)
    group.add_argument('--num_layers', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')
    group.add_argument('--nhead', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')


    group = parser.add_argument_group('training')
    group.add_argument('--devices', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    group.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    group.add_argument('--eval_batch_size', type=int, default=None,
                        help='input batch size for training (default: train batch size)')
    group.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    group.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    group.add_argument('--scheduler', type=str, default=None)
    group.add_argument('--pct_start', type=float, default=0.3)
    group.add_argument('--weight_decay', type=float, default=0.0)
    group.add_argument('--grad_clip', type=float, default=None)
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--max_lr', type=float, default=0.001)
    group.add_argument('--runs', type=int, default=10)
    group.add_argument('--test-freq', type=int, default=1)
    group.add_argument('--start-eval', type=int, default=15)
    group.add_argument('--resume', type=str, default=None)
    group.add_argument('--seed', type=int, default=12344)
    group.add_argument('--token_ratio', type=float, default=0.5)

    group = parser.add_argument_group('cross-validation')
    group.add_argument('--n_folds', type=int, default=10,
                        help='Number of CV folds (default: 10, use 3 for debugging)')

    # fmt: on

    args, _ = parser.parse_known_args()

    return args

args = gene_arg()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
            # cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def load_dataset(args, transform):
    """Load full dataset without splitting."""
    data_name = args.dataset + '-pe'
    dataset = TUDataset(os.path.join(args.data_root, data_name),
                        name=args.dataset,
                        pre_transform=transform
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

    return dataset


def get_labels(dataset):
    """Extract integer labels from dataset for stratified splitting."""
    labels = []
    for data in dataset:
        labels.append(int(data.y.item()))
    return np.array(labels)


# Load full dataset for k-fold cross-validation
full_dataset = load_dataset(args, transform)
num_tasks = full_dataset.num_classes
num_features = full_dataset.num_features
labels = get_labels(full_dataset)

# Prepare cross-validation splits
skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
fold_results = []



class GPS(torch.nn.Module):
    def __init__(self, fea_dim, channels: int, num_layers: int, num_tasks, args):
        super().__init__()

        self.node_emb = Linear(fea_dim, channels)
        self.pe_lin = Linear(20, channels)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINConv(nn), heads=4,
                                                  attn_dropout=0.5,
                                                  token_ratio = args.token_ratio)
            self.convs.append(conv)

        self.lin = Linear(channels, num_tasks)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x = self.node_emb(x) + self.pe_lin(pe)
        #edge_attr = self.edge_emb(edge_attr)
        edge_attr = None

        for conv in self.convs:
            x = conv(x, edge_index, batch)
        x = global_add_pool(x, batch)
        return self.lin(x)


def train(epoch, model, train_loader, optimizer, device):
    """Training function (redefined per fold)."""
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()

        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader, model, device, return_details=False):
    """Testing function with optional per-graph details (redefined per fold)."""
    model.eval()

    correct = 0
    details = []
    total_n_tokens = 0
    total_nodes = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        true_labels = data.y.cpu().numpy()
        pred_labels = out.max(dim=1)[1].cpu().numpy()
        correct += (true_labels == pred_labels).sum().item()
        
        # Compute graph stats per example in batch
        batch_cpu = data.batch.cpu().numpy()
        nodes_per_graph = np.bincount(batch_cpu)
        edge_index_cpu = data.edge_index.cpu()
        edges_per_graph = np.zeros(len(nodes_per_graph))
        
        for edge_idx in range(edge_index_cpu.shape[1]):
            src, dst = edge_index_cpu[0, edge_idx].item(), edge_index_cpu[1, edge_idx].item()
            g_id = batch_cpu[src]
            edges_per_graph[g_id] += 1
        
        edges_per_graph = edges_per_graph / 2.0  # undirected edges
        
        # Compute degree stats per graph
        from torch_geometric.utils import degree as compute_degree
        deg = compute_degree(edge_index_cpu[0], num_nodes=data.x.size(0)).cpu().numpy()
        
        deg_per_graph_mean = np.zeros(len(nodes_per_graph))
        deg_per_graph_var = np.zeros(len(nodes_per_graph))
        for g_id in range(len(nodes_per_graph)):
            mask = (batch_cpu == g_id)
            if mask.sum() > 0:
                g_degs = deg[mask]
                deg_per_graph_mean[g_id] = float(g_degs.mean())
                deg_per_graph_var[g_id] = float(g_degs.var()) if len(g_degs) > 1 else 0.0
        
        # Token efficiency: static ratio for all graphs
        for i in range(len(true_labels)):
            num_n = int(nodes_per_graph[i])
            num_e = int(edges_per_graph[i])
            density = (2.0 * num_e) / (num_n * (num_n - 1) + 1e-8) if num_n > 1 else 0.0
            total_nodes += num_n
            total_n_tokens += int(num_n * args.token_ratio)
            
            if return_details:
                details.append({
                    "num_nodes":       int(num_n),
                    "num_edges":       int(num_e),
                    "density":         float(density),
                    "avg_degree":      float(deg_per_graph_mean[i]),
                    "degree_variance": float(deg_per_graph_var[i]),
                    "true_label":      int(true_labels[i]),
                    "pred_label":      int(pred_labels[i]),
                    "token_ratio":     float(args.token_ratio),
                    "correct":         int(true_labels[i] == pred_labels[i]),
                })
    
    acc = correct / len(loader.dataset)
    avg_token_ratio = args.token_ratio
    # Estimate FLOP reduction assuming quadratic attention: pruning from k to k*r reduces FLOPs by ~1 - r²
    flop_reduction = 1.0 - (args.token_ratio ** 2)
    
    if return_details:
        return acc, avg_token_ratio, flop_reduction, details
    return acc, avg_token_ratio, flop_reduction


device = torch.device('cuda:{}'.format(args.devices) if torch.cuda.is_available() else 'cpu')

run_name = f"{args.dataset}_cv{args.n_folds}"
args.save_path = f"exps/{run_name}-{now}"
os.makedirs(args.save_path, exist_ok=True)

# =====================================================================
# K-Fold Cross-Validation Loop
# =====================================================================
for fold_idx, (train_indices, test_indices) in enumerate(skf.split(np.zeros(len(full_dataset)), labels)):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{args.n_folds}")
    print(f"{'='*60}")
    
    # Create fold directory
    fold_dir = os.path.join(args.save_path, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1} / {args.n_folds}")
    print(f"  Train size: {len(train_indices)} (80%), Val size: {len(val_indices)} (10%), Test size: {len(test_indices)} (10%)")
    print(f"{'='*60}")
    
    # Sub-split training data into train/val (80/10 of training data, remaining 10 is external test from k-fold)
    train_indices = train_indices.copy()
    np.random.shuffle(train_indices)
    val_split = int(len(train_indices) / 9)  # 1/9 = 10% of training data for validation
    val_indices = train_indices[:val_split]
    train_indices = train_indices[val_split:]
    
    # Create datasets and loaders for this fold
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size)
    
    # Reinitialize model and optimizer for this fold
    model = GPS(fea_dim=num_features, channels=64, num_layers=args.num_layers, num_tasks=num_tasks, args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop for this fold (original logic preserved)
    best_val = 0
    epoch_log = []
    
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch, model, train_loader, optimizer, device)
        val_acc, val_ratio, val_flop_red = test(val_loader, model, device)
        test_acc, test_ratio, test_flop_red = test(test_loader, model, device)
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        
        # Log epoch results
        epoch_log.append({
            'epoch': epoch,
            'loss': f"{loss:.6f}",
            'val_acc': f"{val_acc:.4f}",
            'test_acc': f"{test_acc:.4f}",
            'avg_token_ratio': f"{val_ratio:.4f}",
            'flop_reduction': f"{val_flop_red:.4f}",
        })
        
        if best_val < val_acc:
            best_val = val_acc
            torch.save(state_dict, os.path.join(fold_dir, "best_model.pt"))
        
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f'  Fold {fold_idx+1} | Epoch {epoch:03d} | Loss {loss:.4f} | '
                  f'Val {val_acc:.4f} | Test {test_acc:.4f} | '
                  f'token_ratio {val_ratio:.3f} | FLOP_red {val_flop_red:.1%}')
    
    # Load best model and evaluate on test set
    state_dict = torch.load(os.path.join(fold_dir, "best_model.pt"))
    model.load_state_dict(state_dict["model"])
    best_val_acc, best_val_ratio, best_val_flop = test(val_loader, model, device)
    best_test_acc, best_test_ratio, best_test_flop, test_details = test(test_loader, model, device, return_details=True)
    
    # Save epoch log for this fold
    epoch_log_path = os.path.join(fold_dir, "epoch_log.csv")
    with open(epoch_log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'val_acc', 'test_acc', 'avg_token_ratio', 'flop_reduction'])
        writer.writeheader()
        writer.writerows(epoch_log)
    
    # Save per-graph details for this fold
    if test_details:
        details_path = os.path.join(fold_dir, "test_graph_details.csv")
        with open(details_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_details[0].keys())
            writer.writeheader()
            writer.writerows(test_details)
    
    fold_results.append({
        'fold': fold_idx,
        'val_acc': best_val_acc,
        'test_acc': best_test_acc,
        'token_ratio': best_test_ratio,
        'flop_reduction': best_test_flop
    })
    
    print(f'  Fold {fold_idx}: Best Val Acc: {best_val_acc:.4f}, Test Acc: {best_test_acc:.4f}, Token Ratio: {best_test_ratio:.3f}, FLOP Red: {best_test_flop:.1%}')

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*60}")
print("Cross-Validation Summary")
print(f"{'='*60}")

test_accs = [r['test_acc'] for r in fold_results]
token_ratios = [r['token_ratio'] for r in fold_results]
flop_reds = [r['flop_reduction'] for r in fold_results]

print(f"Mean Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
print(f"Range: {np.min(test_accs):.4f} - {np.max(test_accs):.4f}")
print(f"Token Ratio: {np.mean(token_ratios):.4f} ± {np.std(token_ratios):.4f}")
print(f"Est. FLOP Reduction: {np.mean(flop_reds):.1%} ± {np.std(flop_reds):.1%}")

results_cv_to_file(args, fold_results)
