import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import calculate_uncertainty_setting_a, calculate_uncertainty_setting_b

# --- CONFIGURATION ---
DATASET = "cifar10"
RESULTS_DIR = "results"
N_SAMPLES = 1000
os.makedirs(RESULTS_DIR, exist_ok=True)

x_lims = (1e-4, 5e1)
y_lims = (1e-4, 5e1)

def load_and_calculate(algo_name, math_setting):
    file_path = f"data/{algo_name}_{DATASET}_preds.pkl"
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
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
    
    if math_setting == "A":
        shape_for_a = samples_sliced.permute(1, 2, 0, 3)
        res = calculate_uncertainty_setting_a(avg_pred, shape_for_a)
    else:
        res = calculate_uncertainty_setting_b(avg_pred, samples_sliced)
        
    aleatoric = np.clip(res['aleatoric'].cpu().numpy(), a_min=1e-14, a_max=None)
    epistemic = np.clip(res['epistemic'].cpu().numpy(), a_min=1e-14, a_max=None)
    return aleatoric, epistemic

mcdo_ale, mcdo_epi = load_and_calculate("mcdo", "A")
quam_ale, quam_epi = load_and_calculate("quam", "B")

print("Drawing upgraded combined scatter plot...")
fig, ax = plt.subplots(figsize=(9, 9), dpi=150)

ax.scatter(mcdo_ale, mcdo_epi, c='dodgerblue', alpha=0.15, s=12, label='MCDO (Setting A)', edgecolors='none')
ax.scatter(quam_ale, quam_epi, c='crimson', alpha=0.15, s=12, label='QUAM (Setting B)', edgecolors='none')

mcdo_extreme_idx = np.argsort(mcdo_epi)[-50:]
quam_extreme_idx = np.argsort(quam_epi)[-50:]

ax.scatter(mcdo_ale[mcdo_extreme_idx], mcdo_epi[mcdo_extreme_idx], 
           facecolors='none', edgecolors='blue', s=80, linewidths=1.5, alpha=0.8, label="MCDO Extremes")
ax.scatter(quam_ale[quam_extreme_idx], quam_epi[quam_extreme_idx], 
           facecolors='none', edgecolors='darkred', s=80, linewidths=1.5, alpha=0.8, label="QUAM Extremes")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(x_lims)
ax.set_ylim(y_lims)

ax.grid(True, which="both", ls="--", alpha=0.2)

ax.set_xlabel('Aleatoric Uncertainty', fontsize=14, fontweight='bold')
ax.set_ylabel('Epistemic Uncertainty', fontsize=14, fontweight='bold')
ax.set_title(f'MCDO vs QUAM Uncertainties ({DATASET.upper()})', fontsize=16, pad=15)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

leg = ax.legend(fontsize=11, loc='upper left')
for lh in leg.legend_handles: 
    lh.set_alpha(1)

fig.tight_layout()
filename = f"{RESULTS_DIR}/combined_mcdo_quam_{DATASET}_upgraded.png"
fig.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Success! Saved as {filename}")