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

#---------------1. LOAD DATA & BUILD DATAFRAME----------------------
print("Loading data...")
mcdo_al_B, mcdo_ep_B = get_uncertainties("mcdo", "B")
ens_al_B, ens_ep_B = get_uncertainties("ensembles", "B")

# We create a Pandas DataFrame (like an Excel table) for Setting B
# (You can easily copy this block for Setting A if he wants both)
df_B = pd.DataFrame({
    'MCDO Aleatoric': mcdo_al_B,
    'MCDO Epistemic': mcdo_ep_B,
    'Ens Aleatoric': ens_al_B,
    'Ens Epistemic': ens_ep_B
})

#---------------2. CORRELATION MATRICES (HEATMAPS)----------------------
print("Drawing Correlation Matrices...")

for corr_type in ['pearson', 'spearman']:
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Calculate the matrix based on the loop
    corr_matrix = df_B.corr(method=corr_type)
    
    # Draw the Heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 12})
    
    plt.title(f'{DATASET_NAME.upper()} Setting B: {corr_type.capitalize()} Correlation', fontsize=14, pad=15)
    plt.tight_layout()
    
    corr_path = os.path.join(RESULTS_DIR, f'correlation_matrix_B_{corr_type}_{DATASET_NAME}.png')
    plt.savefig(corr_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {corr_type.capitalize()} matrix!")

#---------------3. PAIRPLOT (DIAGONAL PLOT)----------------------
print("Drawing Pairplot...")

# Apply Log10 so the pairplot spreads out visually (ignoring log(0) warnings since we clamped to 1e-14)
df_B_log = np.log10(df_B)

# Draw the Pairplot
pairplot = sns.pairplot(
    df_B_log, 
    diag_kind='kde', # 'kde' gives those smooth mountain curves on the diagonal
    plot_kws={'alpha': 0.1, 's': 10, 'color': 'purple', 'edgecolor': 'none'},
    diag_kws={'color': 'purple', 'fill': True}
)

pairplot.fig.suptitle(f'{DATASET_NAME.upper()} Setting B: Log10 Pairplot', y=1.02, fontsize=16)

pairplot_path = os.path.join(RESULTS_DIR, f'pairplot_B_{DATASET_NAME}.png')
pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
print(f"Done! Saved to:\n- {corr_path}\n- {pairplot_path}")