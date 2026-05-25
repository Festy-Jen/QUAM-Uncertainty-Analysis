#---------------LIBRARIES--------------------------------------
import os
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper_functions import calculate_uncertainty_setting_a, calculate_uncertainty_setting_b

#---------------ESTABLISHING-DIRECTORIES----------------------
DATASET_NAME = "cifar10"
RESULTS_DIR = "results"
N_SAMPLES = 1000
os.makedirs(RESULTS_DIR, exist_ok=True)

#---------------YOUR EXACT DATA PROCESSING----------------------
def get_uncertainties(algorithm_name, math_setting):
    input_file = f"data/{algorithm_name}_{DATASET_NAME}_preds.pkl"
    print(f"Loading {input_file} for Setting {math_setting}...")
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    avg_pred = torch.as_tensor(data['average_net_pred'])
    samples = torch.as_tensor(data['sample_preds'])

    if samples.dim() == 3:
        samples = samples.unsqueeze(2)

    if samples.min() < 0 or samples.max() > 1:
        samples = torch.softmax(samples, dim=-1)
        avg_pred = torch.softmax(avg_pred, dim=-1)
        
    samples = torch.clamp(samples, min=1e-10, max=1.0)
    avg_pred = torch.clamp(avg_pred, min=1e-10, max=1.0)

    samples_sliced = samples[:, :N_SAMPLES, :, :]
    
    shape_for_b = samples_sliced                      
    shape_for_a = samples_sliced.permute(1, 2, 0, 3)

    if math_setting == "A":
        math_results = calculate_uncertainty_setting_a(avg_pred, shape_for_a)
    elif math_setting == "B":
        math_results = calculate_uncertainty_setting_b(avg_pred, shape_for_b)

    aleatoric = np.clip(math_results['aleatoric'].numpy(), a_min=1e-14, a_max=None)
    epistemic = np.clip(math_results['epistemic'].numpy(), a_min=1e-14, a_max=None) 
    
    return aleatoric, epistemic

#---------------1. LOAD DATA & BUILD DATAFRAMES----------------------
print("\n--- Loading Setting A ---")
mcdo_al_A, mcdo_ep_A = get_uncertainties("mcdo", "A")
ens_al_A, ens_ep_A = get_uncertainties("ensembles", "A")
quam_al_A, quam_ep_A = get_uncertainties("quam", "A")

print("\n--- Loading Setting B ---")
mcdo_al_B, mcdo_ep_B = get_uncertainties("mcdo", "B")
ens_al_B, ens_ep_B = get_uncertainties("ensembles", "B")
quam_al_B, quam_ep_B = get_uncertainties("quam", "B")

# Build DataFrame for A
df_A = pd.DataFrame({
    'MCDO Aleatoric': mcdo_al_A, 'MCDO Epistemic': mcdo_ep_A,
    'Ens Aleatoric': ens_al_A, 'Ens Epistemic': ens_ep_A,
    'QUAM Aleatoric': quam_al_A, 'QUAM Epistemic': quam_ep_A,
    'Setting': 'Setting A' # Label for the Pairplot
})

# Build DataFrame for B
df_B = pd.DataFrame({
    'MCDO Aleatoric': mcdo_al_B, 'MCDO Epistemic': mcdo_ep_B,
    'Ens Aleatoric': ens_al_B, 'Ens Epistemic': ens_ep_B,
    'QUAM Aleatoric': quam_al_B, 'QUAM Epistemic': quam_ep_B,
    'Setting': 'Setting B' # Label for the Pairplot
})

# Combine them into one massive DataFrame
df_combined = pd.concat([df_A, df_B], ignore_index=True)

#---------------2. SIDE-BY-SIDE CORRELATION MATRICES----------------------
print("\nDrawing Side-by-Side Correlation Matrices...")

for corr_type in ['pearson', 'spearman']:
    # Create a wide figure (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    fig.suptitle(f'{DATASET_NAME.upper()}: {corr_type.capitalize()} Correlation Comparison (A vs B)', fontsize=20, fontweight='bold', y=1.02)

    # Calculate correlations based on the loop (drop the 'Setting' text column first)
    corr_A = df_A.drop(columns='Setting').corr(method=corr_type)
    corr_B = df_B.drop(columns='Setting').corr(method=corr_type)

    # Plot A on the left
    sns.heatmap(corr_A, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
                annot_kws={"size": 11}, ax=axes[0], cbar=False)
    axes[0].set_title('Setting A', fontsize=16, pad=10)
    # FIX: Use plt.setp to properly rotate and align the labels!
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Plot B on the right
    sns.heatmap(corr_B, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
                annot_kws={"size": 11}, ax=axes[1])
    axes[1].set_title('Setting B', fontsize=16, pad=10)
    # FIX: Use plt.setp to properly rotate and align the labels!
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    corr_path = os.path.join(RESULTS_DIR, f'correlation_comparison_A_vs_B_{corr_type}_{DATASET_NAME}.png')
    plt.savefig(corr_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {corr_type.capitalize()} comparison matrix!")

#---------------3. OVERLAID PAIRPLOT (FIXED SIZE & LEGEND)----------------------
print("Drawing Overlaid Pairplot (This might take a minute)...")

# Apply Log10 only to the numeric columns
numeric_cols = df_combined.columns.drop('Setting')
df_combined[numeric_cols] = np.log10(df_combined[numeric_cols])

# Temporarily boost font sizes for this specific plot
sns.set_context("notebook", rc={"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

# Plot with 'hue' to color by Setting
pairplot = sns.pairplot(
    df_combined, 
    hue='Setting', 
    palette={'Setting A': 'dodgerblue', 'Setting B': 'crimson'}, # Blue = Setting A, Red = Setting B
    diag_kind='kde',
    height=3, # <--- THIS MAKES THE SQUARES BIGGER! (6x6 grid = 18x18 inch image)
    plot_kws={'alpha': 0.15, 's': 5, 'edgecolor': 'none'},
    diag_kws={'linewidth': 2, 'fill': True, 'alpha': 0.3}
)

pairplot.fig.suptitle(f'{DATASET_NAME.upper()}: Log10 Overlaid Distributions (A vs B)', y=1.03, fontsize=24, fontweight='bold')

# FIX THE LEGEND: Move it to the right side, stack it vertically, and keep dots huge
sns.move_legend(
    pairplot, "center left",           # Anchor the left side of the legend...
    bbox_to_anchor=(1.02, 0.5),        # ...just slightly outside the right edge (x=1.02), centered vertically (y=0.5)
    ncol=1,                            # Stack the labels on top of each other
    title=None, frameon=True, 
    fontsize=18, markerscale=10
)

# Force the legend dots to be completely solid (not transparent like the data points)
for lh in pairplot._legend.legend_handles:
    lh.set_alpha(1.0)

pairplot_path = os.path.join(RESULTS_DIR, f'pairplot_A_vs_B_combined_{DATASET_NAME}.png')
pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
print(f"Done! Saved MASSIVE Pairplot to: {pairplot_path}")

# Reset seaborn context back to default so it doesn't mess up future plots
sns.reset_orig()