"""Shared loading and uncertainty post-processing for saved QUAM-format predictions.

The transformations below intentionally retain the numerical conventions of the
initial analysis scripts; they are not a revised QUAM estimator.
Files are Python pickle objects and must come from trusted sources.
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import torch

from helper_functions import calculate_uncertainty_setting_a, calculate_uncertainty_setting_b

METHODS = ("mcdo", "ensembles", "quam")
DATASETS = ("mnist", "emnist", "cifar10")


def load_predictions(data_dir, algorithm, dataset, n_samples=1000):
    """Load reference predictions and sampled models, retaining original preprocessing.

    `n_samples` limits axis 1 (trajectory/sample positions), *not* test inputs.
    For example, [N, 1, 10, C] ensembles always retain their ten members.
    """
    if n_samples < 1:
        raise ValueError("--n-samples must be positive")
    path = Path(data_dir) / f"{algorithm}_{dataset}_preds.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; see README.md for prediction file layout")

    with path.open("rb") as file:
        data = pickle.load(file)  # Only use prediction files from trusted sources.

    for key in ("average_net_pred", "sample_preds", "target"):
        if key not in data:
            raise KeyError(f"{path}: expected key {key!r}")

    avg_pred = torch.as_tensor(data["average_net_pred"])
    samples = torch.as_tensor(data["sample_preds"])
    targets = torch.as_tensor(data["target"])
    del data

    if samples.ndim == 3:
        samples = samples.unsqueeze(2)
    if avg_pred.ndim != 2 or samples.ndim != 4:
        raise ValueError(f"Unexpected shapes: reference {tuple(avg_pred.shape)}, samples {tuple(samples.shape)}")
    if samples.shape[0] != avg_pred.shape[0] or samples.shape[-1] != avg_pred.shape[-1]:
        raise ValueError("Mismatched test inputs or class dimensions")
    if targets.shape[0] != avg_pred.shape[0]:
        raise ValueError("Mismatched label count")

    # Historical convention: apply softmax only if sample values lie outside [0,1].
    # This condition is not a full probability-normalization test.
    if bool((samples.min() < 0) or (samples.max() > 1)):
        samples = torch.softmax(samples, dim=-1)
        avg_pred = torch.softmax(avg_pred, dim=-1)

    # Preserve the original lower bound and do not silently renormalize.
    samples = torch.clamp(samples, min=1e-10, max=1.0)
    avg_pred = torch.clamp(avg_pred, min=1e-10, max=1.0)

    if not bool(torch.isfinite(avg_pred).all()):
        warnings.warn(f"{path.name}: nonfinite reference predictions", stacklevel=2)
    if not bool(torch.isfinite(samples).all()):
        warnings.warn(f"{path.name}: nonfinite sampled predictions", stacklevel=2)

    # Original scripts slice only axis 1, not images, models, or classes.
    return avg_pred, samples[:, :n_samples, :, :], targets


def calculate_uncertainties(avg_pred, sample_preds, setting, clip=1e-14):
    """Calculate A/B using original helpers; optionally clip for log plots.

    Clipping was also used before Pearson/Spearman correlations in the original
    analysis. Keep that convention for matching the original reported figures.
    """
    setting = setting.upper()
    if setting == "A":
        res = calculate_uncertainty_setting_a(avg_pred, sample_preds.permute(1, 2, 0, 3))
    elif setting == "B":
        res = calculate_uncertainty_setting_b(avg_pred, sample_preds)
    else:
        raise ValueError("Setting must be A or B")

    al = res["aleatoric"].detach().cpu().numpy()
    ep = res["epistemic"].detach().cpu().numpy()
    if clip is not None:
        al = np.clip(al, a_min=clip, a_max=None)
        ep = np.clip(ep, a_min=clip, a_max=None)
    return al, ep
