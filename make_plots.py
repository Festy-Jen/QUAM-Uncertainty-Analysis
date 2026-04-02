import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import calculate_uncertainty_setting_a, calculate_uncertainty_setting_b, evaluate_missclass

# ==========================================
# 1. LOAD DATA
# ==========================================
pkl_file_path = r"data/mnist_preds.pkl" 
print(f"Loading {pkl_file_path} ...")

with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

average_net_pred = data['average_net_pred']
sample_preds = data['sample_preds']

# Ensure they are PyTorch tensors
if not isinstance(average_net_pred, torch.Tensor):
    average_net_pred = torch.tensor(average_net_pred)
if not isinstance(sample_preds, torch.Tensor):
    sample_preds = torch.tensor(sample_preds)

print(f"Original sample_preds shape: {sample_preds.shape}")

# ==========================================
# 2. SANITIZE DATA (THE BUG FIXES)
# ==========================================
# Fix 1: Shape Check. Mykyta's setting_a needs [samples, models, points, classes]
# If the first dimension is massive (e.g., 10000 points), it's in the wrong order.
if sample_preds.shape[0] >= 1000:
    if len(sample_preds.shape) == 3: # [points, samples, classes] -> [samples, 1, points, classes]
        sample_preds = sample_preds.permute(1, 0, 2).unsqueeze(1)
        average_net_pred = average_net_pred.unsqueeze(0).unsqueeze(1) # [1, 1, points, classes]
    elif len(sample_preds.shape) == 4: # [points, samples, models, classes] -> [samples, models, points, classes]
        sample_preds = sample_preds.permute(1, 2, 0, 3)

print(f"Corrected sample_preds shape: {sample_preds.shape}")

# Fix 2: If they are raw logits (contain negatives), convert to probabilities
if sample_preds.min() < 0.0 or sample_preds.max() > 1.0:
    print("Detected raw logits! Applying softmax...")
    sample_preds = torch.softmax(sample_preds, dim=-1)
    average_net_pred = torch.softmax(average_net_pred, dim=-1)

# Fix 3: Prevent log(0) NaNs by clamping perfectly zero probabilities
eps = 1e-10
sample_preds = torch.clamp(sample_preds, min=eps, max=1.0)
average_net_pred = torch.clamp(average_net_pred, min=eps, max=1.0)

# ==========================================
# 3. RUN MYKYTA'S MATH
# ==========================================
print("Calculating Yarin Gal's Decomposition...")
gal_results = calculate_uncertainty_setting_a(average_net_pred, sample_preds)

aleatoric_gal = gal_results['aleatoric'].numpy()
epistemic_gal = gal_results['epistemic'].numpy()

# Debug: Check if NaNs survived
print(f"NaNs in Aleatoric: {np.isnan(aleatoric_gal).sum()}")
print(f"NaNs in Epistemic: {np.isnan(epistemic_gal).sum()}")

# ==========================================
# 4. DRAW CORRELATION PLOT
# ==========================================
print("Generating Correlation Plot...")
plt.figure(figsize=(6, 6))

# Added explicit limits to force Matplotlib to look at the right area
plt.hexbin(aleatoric_gal, epistemic_gal, gridsize=100, cmap='plasma', 
           bins='log', xscale='log', yscale='log', mincnt=1)

# Setting limits explicitly to match the paper
plt.xlim(10**-13, 10**1)
plt.ylim(10**-14, 10**0)

plt.xlabel('Aleatoric Uncertainty', color='blue', fontsize=12)
plt.ylabel('Epistemic Uncertainty', color='darkred', fontsize=12)
plt.title('MC Dropout: Entangled Uncertainties (MNIST)', fontsize=14)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("results/correlation_gal_mnist.png", dpi=300)
print("✅ Success! Saved as correlation_gal_mnist.png")
plt.show()