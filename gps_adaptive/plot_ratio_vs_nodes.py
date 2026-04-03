import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

# Path to the graph_stats.csv file
csv_path = r"exps\DD_adaptive_cv3-04_03-16_30_00\fold_0\graph_stats.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Compute graph properties
df['log_num_nodes'] = np.log(df['num_nodes'] + 1)
df['degree_per_node'] = df['num_nodes'].apply(lambda x: 2 * 5 / x if x > 0 else 0)  # rough approx

# Compute correlations with token ratio
print("\n" + "="*70)
print("CORRELATIONS: Graph Properties vs Average Token Ratio")
print("="*70)
correlations = {}
for col in ['num_nodes', 'log_num_nodes']:
    if col in df.columns:
        rho, pval = spearmanr(df[col], df['avg_token_ratio'])
        correlations[col] = (rho, pval)
        print(f"{col:20s}: ρ = {rho:+.4f} (p={pval:.2e})")
print("="*70 + "\n")

# Remove outliers using IQR method
Q1_nodes = df['num_nodes'].quantile(0.25)
Q3_nodes = df['num_nodes'].quantile(0.75)
IQR_nodes = Q3_nodes - Q1_nodes

Q1_ratio = df['avg_token_ratio'].quantile(0.25)
Q3_ratio = df['avg_token_ratio'].quantile(0.75)
IQR_ratio = Q3_ratio - Q1_ratio

# Filter outliers
df_filtered = df[
    (df['num_nodes'] >= Q1_nodes - 1.5 * IQR_nodes) & 
    (df['num_nodes'] <= Q3_nodes + 1.5 * IQR_nodes) &
    (df['avg_token_ratio'] >= Q1_ratio - 1.5 * IQR_ratio) & 
    (df['avg_token_ratio'] <= Q3_ratio + 1.5 * IQR_ratio)
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['num_nodes'], df_filtered['avg_token_ratio'], alpha=0.6, s=50)

# Add trend line (using filtered data)
z = np.polyfit(df_filtered['num_nodes'], df_filtered['avg_token_ratio'], 1)
p = np.poly1d(z)
sorted_nodes = df_filtered['num_nodes'].sort_values()
plt.plot(sorted_nodes, p(sorted_nodes), 
         "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2f}')

plt.xlabel('Number of Nodes (num_nodes)', fontsize=12)
plt.ylabel('Average Token Ratio (avg_token_ratio)', fontsize=12)
plt.title('Token Ratio vs Graph Size', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig('token_ratio_vs_nodes.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Plotted {len(df_filtered)} graphs (removed {len(df) - len(df_filtered)} outliers)")
print(f"  num_nodes range: {df_filtered['num_nodes'].min():.0f} - {df_filtered['num_nodes'].max():.0f}")
print(f"  avg_token_ratio range: {df_filtered['avg_token_ratio'].min():.4f} - {df_filtered['avg_token_ratio'].max():.4f}")
