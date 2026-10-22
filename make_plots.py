import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr

from helper_functions import calculate_uncertainty_setting_a, calculate_uncertainty_setting_b


# -----------------------------------------------------------------------------
# MAIN PRACTICAL CONFIGURATION
# -----------------------------------------------------------------------------
# Edit this block for your regular workflow.
BASE_CONFIG = {
    "dataset_name": "emnist",
    "results_dir": "results",
    "algorithms": ["mcdo", "ensembles", "quam"],
    "math_settings": ["A", "B"],
    "n_samples": 1000,
    "value_clip_floor": 1e-14,
    "probability_floor": 1e-10,
}

# Add scenarios here to compare adjustments in one run.
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
    if samples.dim() == 3:
        samples = samples.unsqueeze(2)

    if samples.min() < 0 or samples.max() > 1:
        samples = torch.softmax(samples, dim=-1)
        avg_pred = torch.softmax(avg_pred, dim=-1)

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

    aleatoric = np.clip(math_results["aleatoric"].numpy(), a_min=config["value_clip_floor"], a_max=None)
    epistemic = np.clip(math_results["epistemic"].numpy(), a_min=config["value_clip_floor"], a_max=None)
    return aleatoric, epistemic


def build_setting_dataframe(
    setting_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    setting_name: str,
) -> pd.DataFrame:
    data_dict = {"Setting": setting_name}
    for algorithm_name, (aleatoric, epistemic) in setting_results.items():
        short_name = "Ens" if algorithm_name == "ensembles" else algorithm_name.upper()
        data_dict[f"{short_name} Aleatoric"] = aleatoric
        data_dict[f"{short_name} Epistemic"] = epistemic
    return pd.DataFrame(data_dict)


def save_correlation_matrices(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    dataset_name: str,
    scenario_name: str,
    results_dir: str,
) -> None:
    print("Drawing side-by-side correlation matrices...")
    for corr_type in ["pearson", "spearman"]:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
        fig.suptitle(
            f"{dataset_name.upper()} [{scenario_name}]: {corr_type.capitalize()} Correlation (A vs B)",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )

        corr_a = df_a.drop(columns="Setting").corr(method=corr_type)
        corr_b = df_b.drop(columns="Setting").corr(method=corr_type)

        sns.heatmap(
            corr_a,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            annot_kws={"size": 11},
            ax=axes[0],
            cbar=False,
        )
        axes[0].set_title("Setting A", fontsize=16, pad=10)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        sns.heatmap(
            corr_b,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            annot_kws={"size": 11},
            ax=axes[1],
        )
        axes[1].set_title("Setting B", fontsize=16, pad=10)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        plt.tight_layout()
        output_path = os.path.join(
            results_dir,
            f"correlation_comparison_A_vs_B_{corr_type}_{dataset_name}_{scenario_name}.png",
        )
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {corr_type} matrix: {output_path}")


def save_pairplot(
    df_combined: pd.DataFrame,
    dataset_name: str,
    scenario_name: str,
    results_dir: str,
) -> None:
    print("Drawing overlaid pairplot...")
    numeric_cols = df_combined.columns.drop("Setting")
    df_for_plot = df_combined.copy()
    df_for_plot[numeric_cols] = np.log10(df_for_plot[numeric_cols])

    sns.set_context("notebook", rc={"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})
    pairplot = sns.pairplot(
        df_for_plot,
        hue="Setting",
        palette={"Setting A": "dodgerblue", "Setting B": "crimson"},
        diag_kind="kde",
        height=3,
        plot_kws={"alpha": 0.15, "s": 5, "edgecolor": "none"},
        diag_kws={"linewidth": 2, "fill": True, "alpha": 0.3},
    )
    pairplot.fig.suptitle(
        f"{dataset_name.upper()} [{scenario_name}]: Log10 Overlaid Distributions (A vs B)",
        y=1.03,
        fontsize=24,
        fontweight="bold",
    )
    sns.move_legend(
        pairplot,
        "center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        title=None,
        frameon=True,
        fontsize=18,
        markerscale=10,
    )
    for legend_handle in pairplot._legend.legend_handles:
        legend_handle.set_alpha(1.0)

    output_path = os.path.join(results_dir, f"pairplot_A_vs_B_combined_{dataset_name}_{scenario_name}.png")
    pairplot.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pairplot: {output_path}")


def summarize_algorithm_setting(
    scenario_name: str,
    dataset_name: str,
    algorithm_name: str,
    math_setting: str,
    n_samples: int,
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
) -> Dict[str, float]:
    pearson_value, _ = pearsonr(aleatoric, epistemic)
    spearman_value, _ = spearmanr(aleatoric, epistemic)
    return {
        "scenario": scenario_name,
        "dataset": dataset_name,
        "algorithm": algorithm_name,
        "math_setting": math_setting,
        "n_samples": n_samples,
        "pearson_aleatoric_vs_epistemic": float(pearson_value),
        "spearman_aleatoric_vs_epistemic": float(spearman_value),
        "aleatoric_mean": float(np.mean(aleatoric)),
        "epistemic_mean": float(np.mean(epistemic)),
        "aleatoric_max": float(np.max(aleatoric)),
        "epistemic_max": float(np.max(epistemic)),
    }


def run_scenario(scenario: Dict) -> List[Dict]:
    scenario_config = merge_config(BASE_CONFIG, scenario["overrides"])
    scenario_name = scenario["name"]
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
            summary_rows.append(
                summarize_algorithm_setting(
                    scenario_name=scenario_name,
                    dataset_name=scenario_config["dataset_name"],
                    algorithm_name=algorithm_name,
                    math_setting=math_setting,
                    n_samples=scenario_config["n_samples"],
                    aleatoric=aleatoric,
                    epistemic=epistemic,
                )
            )

    df_a = build_setting_dataframe(results_by_setting["A"], "Setting A")
    df_b = build_setting_dataframe(results_by_setting["B"], "Setting B")
    df_combined = pd.concat([df_a, df_b], ignore_index=True)

    save_correlation_matrices(
        df_a=df_a,
        df_b=df_b,
        dataset_name=scenario_config["dataset_name"],
        scenario_name=scenario_name,
        results_dir=scenario_config["results_dir"],
    )
    save_pairplot(
        df_combined=df_combined,
        dataset_name=scenario_config["dataset_name"],
        scenario_name=scenario_name,
        results_dir=scenario_config["results_dir"],
    )
    return summary_rows


def main() -> None:
    all_summary_rows: List[Dict] = []
    for scenario in EXPERIMENT_SCENARIOS:
        all_summary_rows.extend(run_scenario(scenario))

    summary_df = pd.DataFrame(all_summary_rows)
    output_csv = os.path.join(BASE_CONFIG["results_dir"], f"make_plots_run_summary_{BASE_CONFIG['dataset_name']}.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSaved run summary table: {output_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
