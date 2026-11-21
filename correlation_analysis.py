"""Correlation comparisons between Settings A and B across three methods.

Scientific computations are kept identical to the original analysis. In
particular, uncertainty is clipped at 1e-14 before calculating correlations.
See README for implications and the upstream QUAM attribution.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save files without requiring an interactive desktop.
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_utils import DATASETS, METHODS, calculate_uncertainties, load_predictions

LABELS = {"mcdo": "MCDO", "ensembles": "Ens", "quam": "QUAM"}
CMAP = plt.get_cmap("coolwarm")
NORM = Normalize(vmin=-1, vmax=1)


def plot_heatmap(ax, matrix, title, show_colorbar=False, font_size=16):
    """Use high-contrast, bold values legible in an exported PDF."""
    sns.heatmap(
        matrix, ax=ax, annot=False, cmap=CMAP, vmin=-1, vmax=1,
        cbar=show_colorbar, square=True, linewidths=0.7, linecolor="white",
        cbar_kws={"shrink": 0.72, "label": "Correlation"} if show_colorbar else None,
    )
    ax.set_title(title, fontsize=17, fontweight="bold", pad=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if not np.isfinite(value):
                color, label = "#161616", "n/a"
            else:
                red, green, blue, _ = CMAP(NORM(value))
                def linear(c):
                    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                lum = (0.2126 * linear(red) + 0.7152 * linear(green)
                       + 0.0722 * linear(blue))
                color = "white" if lum < 0.18 else "#141414"
                label = f"{value:.2f}"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
                    fontsize=font_size, fontweight="bold", color=color)


def render_matrices(frames, output_dir, dataset):
    """Save vector PDF, high-resolution PNG, and full-precision coefficients."""
    for kind in ("pearson", "spearman"):
        matrices = {s: frame.corr(method=kind) for s, frame in frames.items()}
        for setting, matrix in matrices.items():
            matrix.to_csv(output_dir / f"correlation_{kind}_{dataset}_setting_{setting}.csv")
            fig, ax = plt.subplots(figsize=(9, 7.5))
            plot_heatmap(ax, matrix, f"{dataset.upper()} — {kind.title()} — Setting {setting}",
                         show_colorbar=True, font_size=18)
            fig.tight_layout(pad=1.2)
            stem = output_dir / f"correlation_{kind}_{dataset}_setting_{setting}"
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"{dataset.upper()}: {kind.title()} Correlation Comparison (A vs B)",
                     fontsize=21, fontweight="bold", y=1.01)
        plot_heatmap(axs[0], matrices["A"], "Setting A", font_size=17)
        plot_heatmap(axs[1], matrices["B"], "Setting B", show_colorbar=True, font_size=17)
        fig.tight_layout(pad=1.1)
        stem = output_dir / f"correlation_comparison_A_vs_B_{kind}_{dataset}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {kind} comparisons to {output_dir}")


def render_pairplot(frames, output_dir, dataset, limit=10000):
    """Optional exploratory plot; omitted from the default command for speed."""
    items = []
    for setting, frame in frames.items():
        subset = frame.iloc[:limit].copy()
        subset["Setting"] = f"Setting {setting}"
        items.append(subset)
    frame = pd.concat(items, ignore_index=True)
    cols = list(frames["A"].columns)
    frame[cols] = np.log10(frame[cols])
    sns.set_context("notebook", rc={"axes.labelsize": 13, "xtick.labelsize": 11,
                                     "ytick.labelsize": 11})
    pair = sns.pairplot(
        frame, hue="Setting", palette={"Setting A": "dodgerblue", "Setting B": "crimson"},
        diag_kind="kde", height=2.7,
        plot_kws={"alpha": 0.15, "s": 5, "edgecolor": "none"},
        diag_kws={"linewidth": 2, "fill": True, "alpha": 0.25},
    )
    pair.fig.suptitle(f"{dataset.upper()}: log10 score distributions (A vs B)",
                      fontsize=19, fontweight="bold", y=1.02)
    pair.savefig(output_dir / f"pairplot_A_vs_B_combined_{dataset}.png", dpi=200,
                 bbox_inches="tight")
    plt.close(pair.fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Maximum trajectory/sample positions (not test images)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--pairplot", action="store_true", help="Also generate large pairplot")
    parser.add_argument("--pairplot-points", type=int, default=10000)
    args = parser.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    col_data = {"A": {}, "B": {}}
    reference_targets = None
    for method in METHODS:
        avg, samples, targets = load_predictions(args.data_dir, method, args.dataset,
                                                  args.n_samples)
        if reference_targets is None:
            reference_targets = targets
        elif not np.array_equal(targets.cpu().numpy(), reference_targets.cpu().numpy()):
            raise ValueError(f"Targets differ for {method}: test-index alignment must be fixed")
        for setting in ("A", "B"):
            aleatoric, epistemic = calculate_uncertainties(avg, samples, setting)
            col_data[setting][f"{LABELS[method]} Aleatoric"] = aleatoric
            col_data[setting][f"{LABELS[method]} Epistemic"] = epistemic
        del avg, samples, targets
    frames = {s: pd.DataFrame(cols) for s, cols in col_data.items()}
    render_matrices(frames, args.results_dir, args.dataset)
    if args.pairplot:
        render_pairplot(frames, args.results_dir, args.dataset, args.pairplot_points)
    print("Note: matching target labels is necessary but not sufficient to establish image alignment.")


if __name__ == "__main__":
    main()
