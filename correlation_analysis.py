import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from helper_functions import (
    calculate_uncertainty_setting_a,
    calculate_uncertainty_setting_b,
    plot_uncertainty_correlation,
)


# -----------------------------------------------------------------------------
# MAIN PRACTICAL CONFIGURATION
# -----------------------------------------------------------------------------
# Edit this block for your usual single-run workflow.
BASE_CONFIG = {
    "algorithm_name": "ensembles",
    "dataset_name": "mnist",
    "math_setting": "A",
    "results_dir": "results",
    "n_samples": 1000,
    "value_clip_floor": 1e-14,
    "probability_floor": 1e-10,
}

# Plot limits can differ by dataset and practical setup.
PLOT_SETTINGS = {
    "mnist": {"x_lim": (1e-15, 1e6), "y_lim": (1e-15, 1e6)},
    "emnist": {"x_lim": (1e-13, 1e1), "y_lim": (1e-14, 1e0)},
    "cifar10": {"x_lim": (1e-4, 5e1), "y_lim": (1e-5, 5e1)},
}

# Add scenarios when you want to compare adjustments in one execution.
EXPERIMENT_SCENARIOS = [
    {"name": "default", "overrides": {}},
    {"name": "setting_b", "overrides": {"math_setting": "B"}},
]


def merge_config(base: Dict, overrides: Dict) -> Dict:
    merged = dict(base)
    merged.update(overrides)
    return merged


def load_prediction_tensors(dataset_name: str, algorithm_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    input_file = f"data/{algorithm_name}_{dataset_name}_preds.pkl"
    print(f"Loading file: {input_file}")
    with open(input_file, "rb") as file_obj:
        data = pickle.load(file_obj)
    avg_pred = torch.as_tensor(data["average_net_pred"])
    samples = torch.as_tensor(data["sample_preds"])
    return avg_pred, samples


def normalize_probability_tensors(
    avg_pred: torch.Tensor,
    samples: torch.Tensor,
    probability_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Uncertainty functions expect:
    # [n_points, n_posterior_samples, n_models, n_classes]
    if samples.dim() == 3:
        samples = samples.unsqueeze(2)

    # Convert logits to probabilities if needed.
    if samples.min() < 0 or samples.max() > 1:
        print("Input appears to be logits. Applying softmax.")
        samples = torch.softmax(samples, dim=-1)
        avg_pred = torch.softmax(avg_pred, dim=-1)

    # Clamp to avoid log(0).
    samples = torch.clamp(samples, min=probability_floor, max=1.0)
    avg_pred = torch.clamp(avg_pred, min=probability_floor, max=1.0)
    return avg_pred, samples


def compute_uncertainties(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    avg_pred, samples = load_prediction_tensors(config["dataset_name"], config["algorithm_name"])
    avg_pred, samples = normalize_probability_tensors(
        avg_pred=avg_pred,
        samples=samples,
        probability_floor=config["probability_floor"],
    )

    samples_sliced = samples[:, : config["n_samples"], :, :]
    shape_for_b = samples_sliced
    shape_for_a = samples_sliced.permute(1, 2, 0, 3)

    if config["math_setting"] == "A":
        math_results = calculate_uncertainty_setting_a(avg_pred, shape_for_a)
    elif config["math_setting"] == "B":
        math_results = calculate_uncertainty_setting_b(avg_pred, shape_for_b)
    else:
        raise ValueError(f"Unsupported math setting: {config['math_setting']}")

    aleatoric = np.clip(
        math_results["aleatoric"].numpy(),
        a_min=config["value_clip_floor"],
        a_max=None,
    )
    epistemic = np.clip(
        math_results["epistemic"].numpy(),
        a_min=config["value_clip_floor"],
        a_max=None,
    )
    return aleatoric, epistemic


def summarize_results(aleatoric: np.ndarray, epistemic: np.ndarray) -> Dict[str, float]:
    pearson_value, _ = pearsonr(aleatoric, epistemic)
    spearman_value, _ = spearmanr(aleatoric, epistemic)
    return {
        "pearson": float(pearson_value),
        "spearman": float(spearman_value),
        "aleatoric_mean": float(np.mean(aleatoric)),
        "epistemic_mean": float(np.mean(epistemic)),
        "aleatoric_max": float(np.max(aleatoric)),
        "epistemic_max": float(np.max(epistemic)),
    }


def run_scenario(scenario: Dict) -> Dict:
    config = merge_config(BASE_CONFIG, scenario["overrides"])
    os.makedirs(config["results_dir"], exist_ok=True)

    print(f"\n=== Running scenario: {scenario['name']} ===")
    print(f"Parameters: {config}")

    aleatoric, epistemic = compute_uncertainties(config)
    limits = PLOT_SETTINGS[config["dataset_name"]]

    figure_path = os.path.join(
        config["results_dir"],
        f"correlation_{config['algorithm_name'].upper()}_{config['dataset_name'].upper()}_setting_{config['math_setting']}_{scenario['name']}.png",
    )
    plot_uncertainty_correlation(
        aleatoric=aleatoric,
        epistemic=epistemic,
        title=f"{config['algorithm_name'].upper()} Entangled Uncertainties ({config['dataset_name'].upper()}, Setting {config['math_setting']})",
        filename=figure_path,
        x_lims=limits["x_lim"],
        y_lims=limits["y_lim"],
    )

    summary = summarize_results(aleatoric, epistemic)
    return {
        "scenario": scenario["name"],
        "algorithm": config["algorithm_name"],
        "dataset": config["dataset_name"],
        "math_setting": config["math_setting"],
        "n_samples": config["n_samples"],
        "x_lim_min": limits["x_lim"][0],
        "x_lim_max": limits["x_lim"][1],
        "y_lim_min": limits["y_lim"][0],
        "y_lim_max": limits["y_lim"][1],
        "plot_file": figure_path,
        **summary,
    }


def main() -> None:
    all_rows: List[Dict] = []
    for scenario in EXPERIMENT_SCENARIOS:
        all_rows.append(run_scenario(scenario))

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(BASE_CONFIG["results_dir"], f"correlation_run_summary_{BASE_CONFIG['dataset_name']}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary table: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
