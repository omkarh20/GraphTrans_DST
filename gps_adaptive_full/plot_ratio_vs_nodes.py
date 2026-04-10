"""
plot_ratio_vs_nodes.py — gps_adaptive_full
===========================================
Changes over gps_adaptive version:
  - Added head gate correlation printout
  - Added second subplot: Head Gate vs Num Nodes
  - Figure now 2×2: token ratio vs nodes, head gate vs nodes,
    token ratio distribution, head gate distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# ── Config ────────────────────────────────────────────────────────────────────
csv_path = r"exps\DD_adaptive_full_cv10-04_03-21_06_08\fold_0\graph_stats.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df['log_num_nodes'] = np.log(df['num_nodes'] + 1)

# ── Correlations ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CORRELATIONS: Graph Properties vs Token Ratio / Head Gate")
print("="*70)

cols_to_check = ['num_nodes', 'log_num_nodes']
targets       = ['avg_token_ratio', 'avg_head_gate']

for target in targets:
    if target not in df.columns:
        print(f"  (skipping {target} — column not found)")
        continue
    print(f"\n  Target: {target}")
    for col in cols_to_check:
        if col in df.columns:
            rho, pval = spearmanr(df[col], df[target])
            print(f"    {col:20s}: ρ = {rho:+.4f}  (p={pval:.2e})")

print("="*70 + "\n")

# ── Outlier removal (IQR on num_nodes only) ───────────────────────────────────
Q1 = df['num_nodes'].quantile(0.25)
Q3 = df['num_nodes'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = df[
    (df['num_nodes'] >= Q1 - 1.5 * IQR) &
    (df['num_nodes'] <= Q3 + 1.5 * IQR)
]

# ── 2×2 Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Adaptive Budget Decisions vs Graph Size',
             fontsize=15, fontweight='bold')

# ── Panel 1: Token Ratio vs Num Nodes ────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(df_filtered['num_nodes'], df_filtered['avg_token_ratio'],
           alpha=0.6, s=50, color='steelblue')
z = np.polyfit(df_filtered['num_nodes'], df_filtered['avg_token_ratio'], 1)
p = np.poly1d(z)
sx = df_filtered['num_nodes'].sort_values()
ax.plot(sx, p(sx), 'r--', linewidth=2,
        label=f'Trend: y={z[0]:.2e}x+{z[1]:.2f}')
rho, pval = spearmanr(df_filtered['num_nodes'], df_filtered['avg_token_ratio'])
ax.text(0.05, 0.95, f'ρ={rho:+.3f}', transform=ax.transAxes,
        va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.set_xlabel('Number of Nodes', fontsize=12)
ax.set_ylabel('Avg Token Ratio',  fontsize=12)
ax.set_title('Token Ratio vs Graph Size', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# ── Panel 2: Head Gate vs Num Nodes (NEW) ────────────────────────────────────
ax = axes[0, 1]
if 'avg_head_gate' in df_filtered.columns:
    ax.scatter(df_filtered['num_nodes'], df_filtered['avg_head_gate'],
               alpha=0.6, s=50, color='royalblue')
    z2 = np.polyfit(df_filtered['num_nodes'], df_filtered['avg_head_gate'], 1)
    p2 = np.poly1d(z2)
    ax.plot(sx, p2(sx), 'r--', linewidth=2,
            label=f'Trend: y={z2[0]:.2e}x+{z2[1]:.2f}')
    rho2, pval2 = spearmanr(df_filtered['num_nodes'],
                             df_filtered['avg_head_gate'])
    ax.text(0.05, 0.95, f'ρ={rho2:+.3f}', transform=ax.transAxes,
            va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_xlabel('Number of Nodes',  fontsize=12)
    ax.set_ylabel('Avg Head Gate',    fontsize=12)
    ax.set_title('Head Gate vs Graph Size', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'avg_head_gate\nnot available',
            ha='center', va='center', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

# ── Panel 3: Token Ratio distribution ────────────────────────────────────────
ax = axes[1, 0]
ax.hist(df_filtered['avg_token_ratio'], bins=30, color='steelblue',
        alpha=0.7, edgecolor='black')
ax.axvline(df_filtered['avg_token_ratio'].mean(), color='red',
           linestyle='--', linewidth=2, label='Mean')
ax.axvline(df_filtered['avg_token_ratio'].median(), color='orange',
           linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('Avg Token Ratio', fontsize=12)
ax.set_ylabel('Frequency',       fontsize=12)
ax.set_title('Token Ratio Distribution', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# ── Panel 4: Head Gate distribution (NEW) ────────────────────────────────────
ax = axes[1, 1]
if 'avg_head_gate' in df_filtered.columns:
    ax.hist(df_filtered['avg_head_gate'], bins=30, color='royalblue',
            alpha=0.7, edgecolor='black')
    ax.axvline(df_filtered['avg_head_gate'].mean(), color='red',
               linestyle='--', linewidth=2, label='Mean')
    ax.axvline(df_filtered['avg_head_gate'].median(), color='orange',
               linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Avg Head Gate', fontsize=12)
    ax.set_ylabel('Frequency',     fontsize=12)
    ax.set_title('Head Gate Distribution', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, 'avg_head_gate\nnot available',
            ha='center', va='center', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig('ratio_and_headgate_vs_nodes.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Console summary ───────────────────────────────────────────────────────────
print(f"✓ Plotted {len(df_filtered)} graphs "
      f"(removed {len(df) - len(df_filtered)} outliers)")
print(f"  num_nodes range:       "
      f"{df_filtered['num_nodes'].min():.0f} – {df_filtered['num_nodes'].max():.0f}")
print(f"  avg_token_ratio range: "
      f"{df_filtered['avg_token_ratio'].min():.4f} – {df_filtered['avg_token_ratio'].max():.4f}")
if 'avg_head_gate' in df_filtered.columns:
    print(f"  avg_head_gate range:   "
          f"{df_filtered['avg_head_gate'].min():.4f} – {df_filtered['avg_head_gate'].max():.4f}")