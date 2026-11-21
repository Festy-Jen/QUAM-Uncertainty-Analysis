import os
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper_functions import calculate_uncertainty_setting_a, calculate_uncertainty_setting_b

DATASET_NAME = "cifar10"
RESULTS_DIR = "results"
N_SAMPLES = 1000
os.makedirs(RESULTS_DIR, exist_ok=True)
#------
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

#loading data
print("\n--- Loading Setting A ---")
mcdo_al_A, mcdo_ep_A = get_uncertainties("mcdo", "A")
ens_al_A, ens_ep_A = get_uncertainties("ensembles", "A")
quam_al_A, quam_ep_A = get_uncertainties("quam", "A")

print("\n--- Loading Setting B ---")
mcdo_al_B, mcdo_ep_B = get_uncertainties("mcdo", "B")
ens_al_B, ens_ep_B = get_uncertainties("ensembles", "B")
quam_al_B, quam_ep_B = get_uncertainties("quam", "B")

# DataFrame for A
df_A = pd.DataFrame({
    'MCDO Aleatoric': mcdo_al_A, 'MCDO Epistemic': mcdo_ep_A,
    'Ens Aleatoric': ens_al_A, 'Ens Epistemic': ens_ep_A,
    'QUAM Aleatoric': quam_al_A, 'QUAM Epistemic': quam_ep_A,
    'Setting': 'Setting A' # Label for the Pairplot
})

# DataFrame for B
df_B = pd.DataFrame({
    'MCDO Aleatoric': mcdo_al_B, 'MCDO Epistemic': mcdo_ep_B,
    'Ens Aleatoric': ens_al_B, 'Ens Epistemic': ens_ep_B,
    'QUAM Aleatoric': quam_al_B, 'QUAM Epistemic': quam_ep_B,
    'Setting': 'Setting B' # Label for the Pairplot
})

df_combined = pd.concat([df_A, df_B], ignore_index=True)

# Correlation matrices. Produce both a comparison overview and individual
# high-resolution figures: the individual PDFs are best suited for the report.
print("\nDrawing correlation matrices...")

from matplotlib.colors import Normalize

CORR_CMAP = plt.get_cmap("coolwarm")
CORR_NORM = Normalize(vmin=-1, vmax=1)


def draw_correlation_heatmap(ax, matrix, title, colorbar=False, font_size=16):
    """Draw a labeled correlation matrix with contrast-aware annotations."""
    sns.heatmap(
        matrix,
        annot=False,
        cmap=CORR_CMAP,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.7,
        linecolor="white",
        cbar=colorbar,
        cbar_kws={"shrink": 0.72, "label": "Correlation"} if colorbar else None,
        ax=ax,
    )
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.tick_params(length=0)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = float(matrix.iloc[i, j])
            if not np.isfinite(value):
                label, color = "n/a", "#333333"
            else:
                # Select white or dark text based on the actual background color.
                red, green, blue, _ = CORR_CMAP(CORR_NORM(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "white" if luminance < 0.56 else "#171717"
                label = f"{value:.2f}"
            ax.text(
                j + 0.5,
                i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color=color,
            )


for corr_type in ("pearson", "spearman"):
    matrices = {
        "A": df_A.drop(columns="Setting").corr(method=corr_type),
        "B": df_B.drop(columns="Setting").corr(method=corr_type),
    }

    # Report version: one matrix per file, with large readable cell labels.
    for setting, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=200)
        draw_correlation_heatmap(
            ax, matrix, f"{DATASET_NAME.upper()} — {corr_type.title()} — Setting {setting}",
            colorbar=True, font_size=17,
        )
        fig.tight_layout(pad=1.2)
        stem = f"correlation_{corr_type}_{DATASET_NAME}_setting_{setting}"
        fig.savefig(os.path.join(RESULTS_DIR, stem + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(RESULTS_DIR, stem + ".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Overview version, retaining the existing filename for your report link.
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=200)
    fig.suptitle(
        f"{DATASET_NAME.upper()}: {corr_type.title()} Correlation Comparison (A vs B)",
        fontsize=22, fontweight="bold", y=1.02,
    )
    draw_correlation_heatmap(axes[0], matrices["A"], "Setting A", font_size=17)
    draw_correlation_heatmap(axes[1], matrices["B"], "Setting B", colorbar=True, font_size=17)
    fig.tight_layout(pad=1.2)
    stem = f"correlation_comparison_A_vs_B_{corr_type}_{DATASET_NAME}"
    fig.savefig(os.path.join(RESULTS_DIR, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, stem + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {corr_type} matrices as PNG and vector PDF.")

#pairplot
print("Drawing Overlaid Pairplot (This might take a minute)...")

numeric_cols = df_combined.columns.drop('Setting')
df_combined[numeric_cols] = np.log10(df_combined[numeric_cols])

sns.set_context("notebook", rc={"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

pairplot = sns.pairplot(
    df_combined, 
    hue='Setting', 
    palette={'Setting A': 'dodgerblue', 'Setting B': 'crimson'}, 
    diag_kind='kde',
    height=3, 
    plot_kws={'alpha': 0.15, 's': 5, 'edgecolor': 'none'},
    diag_kws={'linewidth': 2, 'fill': True, 'alpha': 0.3}
)

pairplot.fig.suptitle(f'{DATASET_NAME.upper()}: Log10 Overlaid Distributions (A vs B)', y=1.03, fontsize=24, fontweight='bold')

sns.move_legend(
    pairplot, "center left",         
    bbox_to_anchor=(1.02, 0.5),       
    ncol=1,                            
    title=None, frameon=True, 
    fontsize=18, markerscale=10
)

for lh in pairplot._legend.legend_handles:
    lh.set_alpha(1.0)

pairplot_path = os.path.join(RESULTS_DIR, f'pairplot_A_vs_B_combined_{DATASET_NAME}.png')
pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
print(f"Done! Saved MASSIVE Pairplot to: {pairplot_path}")
