#---------------LIBRARIES--------------------------------------
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from helper_functions import (
    calculate_uncertainty_setting_a, 
    calculate_uncertainty_setting_b
)

#---------------ESTABLISHING-DIRECTORIES----------------------
DATASET_NAME = "cifar10"
RESULTS_DIR = "results"
N_SAMPLES = 1000
os.makedirs(RESULTS_DIR, exist_ok=True)

# Adjusted to tightly fit the data you just showed me!
PLOT_LIMS = (1e-3, 1e1)

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

#---------------RUN THE SCRIPT FOR BOTH----------------------
print("Calculating uncertainties...")
mcdo_al_A, mcdo_ep_A = get_uncertainties("mcdo", "A")
ens_al_A, ens_ep_A = get_uncertainties("ensembles", "A")

mcdo_al_B, mcdo_ep_B = get_uncertainties("mcdo", "B")
ens_al_B, ens_ep_B = get_uncertainties("ensembles", "B")

#---------------DRAWING THE 4 QUADRANTS (PUBLICATION READY)----------------------
print("Drawing high-res plots with extreme points...")
fig, axs = plt.subplots(2, 2, figsize=(14, 14))
fig.suptitle(f'{DATASET_NAME.upper()}: Aleatoric vs Epistemic Uncertainty', fontsize=22, fontweight='bold', y=0.98)

def draw_quadrant(ax, aleatoric_pts, epistemic_pts, method_name, setting_name):
    # 1. Base Scatter (Slightly larger dots for high-res)
    ax.scatter(aleatoric_pts, epistemic_pts, alpha=0.3, s=10, c='purple', edgecolors='none')
    
    # 2. Limits, Scales, and Fonts (Publication sizes)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(PLOT_LIMS)
    ax.set_ylim(PLOT_LIMS)
    ax.set_xlabel('Aleatoric Uncertainty', fontsize=16)
    ax.set_ylabel('Epistemic Uncertainty', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Standard diagonal y=x line
    x_vals = np.array([PLOT_LIMS[0], PLOT_LIMS[1]])
    ax.plot(x_vals, x_vals, 'k--', alpha=0.5, linewidth=1.5, label='y=x')

    # 3. EXTREMUM POINTS (Mykyta's request)
    max_ep_idx = np.argmax(epistemic_pts)
    max_al_idx = np.argmax(aleatoric_pts)

    x_ep, y_ep = aleatoric_pts[max_ep_idx], epistemic_pts[max_ep_idx]
    x_al, y_al = aleatoric_pts[max_al_idx], epistemic_pts[max_al_idx]

    # Circle the extremes
    ax.scatter(x_ep, y_ep, facecolors='none', edgecolors='red', s=150, linewidth=2, zorder=5, label='Max Epistemic')
    ax.scatter(x_al, y_al, facecolors='none', edgecolors='blue', s=150, linewidth=2, zorder=5, label='Max Aleatoric')

    # Diagonal lines through extremes (y = c * x)
    c_ep = y_ep / x_ep
    c_al = y_al / x_al
    ax.plot(x_vals, x_vals * c_ep, 'r:', alpha=0.6, linewidth=1.5)
    ax.plot(x_vals, x_vals * c_al, 'b:', alpha=0.6, linewidth=1.5)

    # 4. Scipy Stats & Title
    p_stat, _ = pearsonr(aleatoric_pts, epistemic_pts)
    s_stat, _ = spearmanr(aleatoric_pts, epistemic_pts)
    full_title = f"{method_name} ({setting_name})\nPearson: {p_stat:.3f} | Spearman: {s_stat:.3f}"
    ax.set_title(full_title, fontsize=18, pad=12)
    
    # Add legend to just one plot to keep it clean (top right usually looks good)
    if method_name == 'Ensembles' and setting_name == 'Setting A':
        ax.legend(fontsize=12, loc='lower right')

# Quadrant 1 (Top-Left): MCDO Setting A
draw_quadrant(axs[0, 0], mcdo_al_A, mcdo_ep_A, 'MCDO', 'Setting A')
# Quadrant 2 (Top-Right): Ensembles Setting A
draw_quadrant(axs[0, 1], ens_al_A, ens_ep_A, 'Ensembles', 'Setting A')
# Quadrant 3 (Bottom-Left): MCDO Setting B
draw_quadrant(axs[1, 0], mcdo_al_B, mcdo_ep_B, 'MCDO', 'Setting B')
# Quadrant 4 (Bottom-Right): Ensembles Setting B
draw_quadrant(axs[1, 1], ens_al_B, ens_ep_B, 'Ensembles', 'Setting B')

# Tight layout prevents cutting off labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
save_path = os.path.join(RESULTS_DIR, f'quadrants_pubready_{DATASET_NAME}.png')

# dpi=300 makes it high-resolution! bbox_inches='tight' trims empty borders!
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Done! Publication-ready plot saved to: {save_path}")