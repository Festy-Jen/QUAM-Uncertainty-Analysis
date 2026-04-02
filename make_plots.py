#---------------LIBRARIES--------------------------------------

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import (
    calculate_uncertainty_setting_a, 
    calculate_uncertainty_setting_b, 
    evaluate_missclass
)

#---------------ESTABLISHING-DIRECTORIES----------------------

DATASET_NAME = "emnist"
INPUT_FILE = f"data/{DATASET_NAME}_preds.pkl"
RESULTS_DIR = "results"

#---------------LOADING-DATA--------------------------------------

print(f"--- Loading: {DATASET_NAME.upper()} ---")

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

avg_pred = torch.as_tensor(data['average_net_pred']) #.to(device)
samples = torch.as_tensor(data['sample_preds']) #.to(device)
targets = torch.as_tensor(data['target']) #.to(device)

#----------------NECCESARY-CHECKS----------------------

#our functions expects 4d setting, but we have only 3d, 
#so we plug in 1 as second parameter

if samples.dim() == 3:
    samples = samples.unsqueeze(2)

# 2. Prevent NaNs and Negative Logits for BOTH tensors
if samples.min() < 0 or samples.max() > 1:
    samples = torch.softmax(samples, dim=-1)
    avg_pred = torch.softmax(avg_pred, dim=-1)

samples = torch.clamp(samples, min=1e-10, max=1.0)
avg_pred = torch.clamp(avg_pred, min=1e-10, max=1.0)

# 3. Shape Permutation! 
# Setting A needs: [Samples, Models, Points, Classes]
samples_a = samples.permute(1, 2, 0, 3)

#----------------RUNNING-SCRIPT----------------------

gal_results = calculate_uncertainty_setting_a(avg_pred, samples_a)

print("Data was successfully calculated...")

#we transform it so matplot lib can see numpy.arrays

#the data is messy
aleatoric_gal = gal_results['aleatoric'].numpy()
#the data was not learned
epistemic_gal = gal_results['epistemic'].numpy()

print("Data was successfully transformed...")

#----------------DRAWING-PLOT----------------------

def plot_uncertainty_correlation(
        aleatoric, epistemic, 
        title, filename,
        x_lims=(1e-13, 1e1), y_lims=(1e-14, 1e0)
):
    print(f"Generating {DATASET_NAME} Plot...")

    fig, ax = plt.subplots(figsize=(8,8), dpi=100)

    hb = ax.hexbin(
        aleatoric, epistemic,
        gridsize=100,
        cmap='plasma',
        bins='log',
        xscale='log',yscale='log',
        mincnt=1,
        edgecolors='none'
    )

    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)

    ax.set_xlabel('Aleatoric Uncertainty', color='royalblue', fontsize=12, fontweight='bold')
    ax.set_ylabel('Epistemic Uncertainty', color='firebrick', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=15)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # colorbar with the temperature at the side
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Log10', fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{filename}", bbox_inches='tight', dpi=300)
    print(f"Success! Saved as {filename}")
    plt.show()

plot_uncertainty_correlation(
    aleatoric_gal, epistemic_gal,
    f'MC Dropout: Entangled Uncertainties ({DATASET_NAME})',
    f'correlation_gal_{DATASET_NAME}.png',
)
