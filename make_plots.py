import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from helper_functions import calculate_uncertainty_setting_a, calculate_uncertainty_setting_b, evaluate_missclass

# Loading data
pkl_file_path = r"data/mnist_preds.pkl" 
print(f"Loading {pkl_file_path} ...")

try:
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
except Exception:
    data = torch.load(pkl_file_path, map_location='cpu')

# Extracting predictions
if isinstance(data, dict):
    preds = data.get('predictions', data.get('probs', data.get('logits')))
    targets = data.get('targets', data.get('labels', None))
else:
    preds = data
    targets = None

if not isinstance(preds, torch.Tensor):
    preds = torch.tensor(preds)

if len(preds.shape) == 3:
    preds = preds.unsqueeze(2) 
elif len(preds.shape) == 2:
    preds = preds.unsqueeze(1).unsqueeze(2)

average_net_pred = torch.mean(preds, dim=(1, 2))
# math
print("Calculating Yarin Gal's Decomposition (MC Dropout)...")
preds_a = preds.permute(1, 2, 0, 3) 
gal_results = calculate_uncertainty_setting_a(average_net_pred, preds_a)

# correlation plot
print("Generating Correlation Plot...")
plt.figure(figsize=(6, 6))

aleatoric_gal = gal_results['aleatoric'].numpy()
epistemic_gal = gal_results['epistemic'].numpy()

plt.hexbin(aleatoric_gal, epistemic_gal, gridsize=100, cmap='plasma', 
           bins='log', xscale='log', yscale='log', mincnt=1)

plt.xlabel('Aleatoric Uncertainty', color='blue', fontsize=12)
plt.ylabel('Epistemic Uncertainty', color='darkred', fontsize=12)
plt.title('MC Dropout: Entangled Uncertainties', fontsize=14)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("results/correlation_gal.png", dpi=300)
print("✅ Success! Saved as correlation_gal.png in your results folder.")
plt.show()