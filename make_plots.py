#---------------LIBRARIES--------------------------------------

import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import (
    calculate_uncertainty_setting_a, 
    calculate_uncertainty_setting_b, 
    evaluate_missclass,
    plot_uncertainty_correlation
)

#---------------ESTABLISHING-DIRECTORIES----------------------

#takes format: algoname_datasetname_preds.pkl
#example: mcdo_mnist_preds.pkl
#example: quam_emnist_preds.pkl
ALGORITHM_NAME = "mcdo"
DATASET_NAME = "cifar10"
INPUT_FILE = f"data/{ALGORITHM_NAME}_{DATASET_NAME}_preds.pkl"
RESULTS_DIR = "results"
MATH_SETTING = "A"
N_SAMPLES = 1000

PLOT_SETTINGS = {
    "mnist":   {"x_lim": (1e-13, 1e1), "y_lim": (1e-14, 1e0)},
    "emnist":  {"x_lim": (1e-13, 1e1), "y_lim": (1e-14, 1e0)},
    "cifar10": {"x_lim": (1e-6, 1e1),  "y_lim": (1e-6, 1e1)}
}
current_limits = PLOT_SETTINGS[DATASET_NAME]

#---------------LOADING-DATA--------------------------------------

print(f"--- Loading: {DATASET_NAME.upper()} ---")

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

avg_pred = torch.as_tensor(data['average_net_pred']) #.to(device)
samples = torch.as_tensor(data['sample_preds']) #.to(device)
targets = torch.as_tensor(data['target']) #.to(device)

#----------------NECCESARY-CHECKS---------------------

# Our functions expects 4D setting, but we have only 3D, 
# Hence we plug in 1 as second parameter, by telling to create 1 on index 2

if samples.dim() == 3:
    samples = samples.unsqueeze(2)

# we apply softmax to all negative and positive probs 
# that are x>0 and 1<x
if samples.min() < 0 or samples.max() > 1:
    print("We have logits so we apply Softmax...")
    samples = torch.softmax(samples, dim=-1)
    avg_pred = torch.softmax(avg_pred, dim=-1)
    
# make a border so even our softmaxed probs are not 0.000000000001 small
samples = torch.clamp(samples, min=1e-10, max=1.0)
avg_pred = torch.clamp(avg_pred, min=1e-10, max=1.0)

samples_sliced = samples[:, :N_SAMPLES, :, :]
samples_a = samples.permute(1, 2, 0, 3)

shape_for_b = samples_sliced                      
shape_for_a = samples_sliced.permute(1, 2, 0, 3)

#----------------RUNNING-SCRIPT----------------------

print(f"calculating math...")

if MATH_SETTING == "A":
    math_results = calculate_uncertainty_setting_a(avg_pred, shape_for_a)
elif MATH_SETTING == "B":
    math_results = calculate_uncertainty_setting_b(avg_pred, shape_for_b)

print("Data was successfully calculated...")

#we transform it so matplot lib can see numpy.arrays

aleatoric = np.clip(math_results['aleatoric'].numpy(), a_min=1e-14, a_max=None)
epistemic = np.clip(math_results['epistemic'].numpy(), a_min=1e-14, a_max=None) 
total_uncertainty = math_results['total'].numpy()

print("Data was successfully transformed...")

#----------------DRAWING-PLOT----------------------

plot_uncertainty_correlation(
    aleatoric=aleatoric, 
    epistemic=epistemic,
    title=f'{ALGORITHM_NAME.upper()} Entangled Uncertainties ({DATASET_NAME.upper()})',
    filename=os.path.join(RESULTS_DIR, f'correlation_{ALGORITHM_NAME.upper()}_{DATASET_NAME.upper()}_setting_{MATH_SETTING}.png'),
    x_lims=current_limits["x_lim"], 
    y_lims=current_limits["y_lim"]
)

#----------------EVALUATING-RETENTION-(MISCLASS)----------------------
print("\n--- Retention Metrics (Using EPISTEMIC) ---")

# We use the epistemic tensor to prove QUAM isolates the lack of knowledge!
metrics = evaluate_missclass(
    id_uncerts=math_results['epistemic'], 
    id_uncerts_target=targets, 
    id_uncerts_average_net_pred=avg_pred
)

print(f"AUROC: {metrics['AUROC']:.4f} (Higher is better)")
print(f"AUPR:  {metrics['AUPR']:.4f}  (Higher is better)")
print(f"FPR95: {metrics['FPR']:.4f}  (Lower is better)")
print("-------------------------\n")

