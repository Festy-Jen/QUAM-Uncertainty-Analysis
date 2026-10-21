import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from helper_functions import calculate_uncertainty_setting_a, calculate_uncertainty_setting_b


# -----------------------------------------------------------------------------
# MAIN PRACTICAL CONFIGURATION
# -----------------------------------------------------------------------------
# Edit this block when you want to test different practical adjustments.
BASE_CONFIG = {
    "dataset_name": "cifar10",
    "results_dir": "results",
    "algorithms": ["mcdo", "ensembles"],
    "math_settings": ["A", "B"],
    "n_samples": 1000,
    "plot_lims": (1e-3, 1e1),
    "value_clip_floor": 1e-14,
    "probability_floor": 1e-10,
}

# Add or edit scenarios here to compare practical adjustments in one run.
# Each scenario can override any key from BASE_CONFIG.
EXPERIMENT_SCENARIOS = [
    {"name": "default", "overrides": {}},
    {"name": "fewer_samples_500", "overrides": {"n_samples": 500}},
]


def merge_config(base: Dict, overrides: Dict) -> Dict:
    merged = dict(base)
    merged.update(overrides)
    return merged


def load_prediction_tensors(dataset_name: str, algorithm_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    input_file = f"data/{algorithm_name}_{dataset_name}_preds.pkl"
    print(f"Loading {input_file}...")
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
    # Our uncertainty functions expect 4D samples:
    # [n_points, n_posterior_samples, n_models, n_classes]
    if samples.dim() == 3:
        samples = samples.unsqueeze(2)

    # Some saved files contain logits instead of probabilities.
    if samples.min() < 0 or samples.max() > 1:
        samples = torch.softmax(samples, dim=-1)
        avg_pred = torch.softmax(avg_pred, dim=-1)

    # Clamp to avoid log(0) in entropy formulas.
    samples = torch.clamp(samples, min=probability_floor, max=1.0)
    avg_pred = torch.clamp(avg_pred, min=probability_floor, max=1.0)
    return avg_pred, samples


def compute_uncertainties(
    algorithm_name: str,
    math_setting: str,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    avg_pred, samples = load_prediction_tensors(config["dataset_name"], algorithm_name)
    avg_pred, samples = normalize_probability_tensors(
        avg_pred=avg_pred,
        samples=samples,
        probability_floor=config["probability_floor"],
    )

    samples_sliced = samples[:, : config["n_samples"], :, :]
    shape_for_b = samples_sliced
    shape_for_a = samples_sliced.permute(1, 2, 0, 3)

    if math_setting == "A":
        math_results = calculate_uncertainty_setting_a(avg_pred, shape_for_a)
    elif math_setting == "B":
        math_results = calculate_uncertainty_setting_b(avg_pred, shape_for_b)
    else:
        raise ValueError(f"Unsupported math setting: {math_setting}")

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


def summarize_uncertainty_pair(aleatoric: np.ndarray, epistemic: np.ndarray) -> Dict[str, float]:
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


def draw_quadrant(
    ax: plt.Axes,
    aleatoric_pts: np.ndarray,
    epistemic_pts: np.ndarray,
    method_name: str,
    setting_name: str,
    plot_lims: Tuple[float, float],
) -> None:
    ax.scatter(aleatoric_pts, epistemic_pts, alpha=0.3, s=10, c="purple", edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(plot_lims)
    ax.set_ylim(plot_lims)
    ax.set_xlabel("Aleatoric Uncertainty", fontsize=16)
    ax.set_ylabel("Epistemic Uncertainty", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)

    x_vals = np.array([plot_lims[0], plot_lims[1]])
    ax.plot(x_vals, x_vals, "k--", alpha=0.5, linewidth=1.5, label="y=x")

    max_ep_idx = np.argmax(epistemic_pts)
    max_al_idx = np.argmax(aleatoric_pts)

    x_ep, y_ep = aleatoric_pts[max_ep_idx], epistemic_pts[max_ep_idx]
    x_al, y_al = aleatoric_pts[max_al_idx], epistemic_pts[max_al_idx]

    ax.scatter(x_ep, y_ep, facecolors="none", edgecolors="red", s=150, linewidth=2, zorder=5, label="Max Epistemic")
    ax.scatter(x_al, y_al, facecolors="none", edgecolors="blue", s=150, linewidth=2, zorder=5, label="Max Aleatoric")

    c_ep = y_ep / x_ep
    c_al = y_al / x_al
    ax.plot(x_vals, x_vals * c_ep, "r:", alpha=0.6, linewidth=1.5)
    ax.plot(x_vals, x_vals * c_al, "b:", alpha=0.6, linewidth=1.5)

    p_stat, _ = pearsonr(aleatoric_pts, epistemic_pts)
    s_stat, _ = spearmanr(aleatoric_pts, epistemic_pts)
    title = f"{method_name} ({setting_name})\nPearson: {p_stat:.3f} | Spearman: {s_stat:.3f}"
    ax.set_title(title, fontsize=18, pad=12)


def save_quadrant_plot(
    scenario_name: str,
    dataset_name: str,
    results_dir: str,
    plot_lims: Tuple[float, float],
    results_by_setting: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(
        f"{dataset_name.upper()} [{scenario_name}]: Aleatoric vs Epistemic Uncertainty",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )

    plot_order = [
        ("A", "mcdo", axs[0, 0], "MCDO"),
        ("A", "ensembles", axs[0, 1], "Ensembles"),
        ("B", "mcdo", axs[1, 0], "MCDO"),
        ("B", "ensembles", axs[1, 1], "Ensembles"),
    ]

    for setting_name, algorithm_name, axis, display_name in plot_order:
        aleatoric_pts, epistemic_pts = results_by_setting[setting_name][algorithm_name]
        draw_quadrant(
            ax=axis,
            aleatoric_pts=aleatoric_pts,
            epistemic_pts=epistemic_pts,
            method_name=display_name,
            setting_name=f"Setting {setting_name}",
            plot_lims=plot_lims,
        )

    axs[0, 1].legend(fontsize=12, loc="lower right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(results_dir, f"quadrants_{dataset_name}_{scenario_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved quadrant plot: {output_path}")


def run_scenario(scenario: Dict) -> List[Dict]:
    scenario_name = scenario["name"]
    scenario_config = merge_config(BASE_CONFIG, scenario["overrides"])
    os.makedirs(scenario_config["results_dir"], exist_ok=True)

    print(f"\n=== Running scenario: {scenario_name} ===")
    print(f"Parameters: {scenario_config}")

    results_by_setting: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {"A": {}, "B": {}}
    summary_rows: List[Dict] = []

    for math_setting in scenario_config["math_settings"]:
        for algorithm_name in scenario_config["algorithms"]:
            aleatoric, epistemic = compute_uncertainties(
                algorithm_name=algorithm_name,
                math_setting=math_setting,
                config=scenario_config,
            )
            results_by_setting[math_setting][algorithm_name] = (aleatoric, epistemic)
            summary = summarize_uncertainty_pair(aleatoric, epistemic)
            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "dataset": scenario_config["dataset_name"],
                    "algorithm": algorithm_name,
                    "math_setting": math_setting,
                    "n_samples": scenario_config["n_samples"],
                    "plot_lim_min": scenario_config["plot_lims"][0],
                    "plot_lim_max": scenario_config["plot_lims"][1],
                    **summary,
                }
            )

    save_quadrant_plot(
        scenario_name=scenario_name,
        dataset_name=scenario_config["dataset_name"],
        results_dir=scenario_config["results_dir"],
        plot_lims=scenario_config["plot_lims"],
        results_by_setting=results_by_setting,
    )
    return summary_rows


def main() -> None:
    all_summary_rows: List[Dict] = []
    for scenario in EXPERIMENT_SCENARIOS:
        all_summary_rows.extend(run_scenario(scenario))

    summary_df = pd.DataFrame(all_summary_rows)
    output_csv = os.path.join(BASE_CONFIG["results_dir"], f"run_summary_{BASE_CONFIG['dataset_name']}.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSaved scenario summary table: {output_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
