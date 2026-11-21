"""Four-panel comparison of MCDO and Ensembles under Settings A and B."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from analysis_utils import DATASETS, calculate_uncertainties, load_predictions


def draw_quadrant(ax, aleatoric, epistemic, method, setting, limits):
    """Scatter two uncertainty components; highlight extreme observations."""
    ax.scatter(aleatoric, epistemic, alpha=0.3, s=10,
               c="purple", edgecolors="none")
    ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
           xlabel="Aleatoric uncertainty", ylabel="Epistemic uncertainty")
    ax.tick_params(axis="both", which="major", labelsize=12)

    x = np.asarray(limits)
    ax.plot(x, x, "k--", alpha=0.5, lw=1.4, label="y = x")
    ep_idx = int(np.argmax(epistemic))
    al_idx = int(np.argmax(aleatoric))
    x_ep, y_ep = aleatoric[ep_idx], epistemic[ep_idx]
    x_al, y_al = aleatoric[al_idx], epistemic[al_idx]
    ax.scatter(x_ep, y_ep, facecolors="none", edgecolors="red", s=145, lw=2,
               zorder=5, label="Max epistemic")
    ax.scatter(x_al, y_al, facecolors="none", edgecolors="blue", s=145, lw=2,
               zorder=5, label="Max aleatoric")
    ax.plot(x, x * y_ep / x_ep, "r:", alpha=0.6)
    ax.plot(x, x * y_al / x_al, "b:", alpha=0.6)
    r = pearsonr(aleatoric, epistemic).statistic
    rho = spearmanr(aleatoric, epistemic).statistic
    ax.set_title(f"{method} (Setting {setting})\nPearson: {r:.3f} | Spearman: {rho:.3f}",
                 fontsize=16, pad=11)
    if method == "Ensembles" and setting == "A":
        ax.legend(fontsize=10, loc="lower right")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--limits", nargs=2, type=float, default=(1e-3, 1e1),
                        metavar=("MIN", "MAX"))
    args = parser.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    targets_ref = None
    for method in ("mcdo", "ensembles"):
        avg, samples, targets = load_predictions(args.data_dir, method,
                                                   args.dataset, args.n_samples)
        if targets_ref is None:
            targets_ref = targets
        elif not np.array_equal(targets_ref.numpy(), targets.numpy()):
            raise ValueError("Method target sequences differ")
        for setting in ("A", "B"):
            data[(method, setting)] = calculate_uncertainties(avg, samples, setting)
        del avg, samples, targets

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f"{args.dataset.upper()}: Aleatoric vs Epistemic Uncertainty",
                 fontsize=21, fontweight="bold", y=0.98)
    for ax, method, setting in zip(axs.flatten(),
                                    ("mcdo", "ensembles", "mcdo", "ensembles"),
                                    ("A", "A", "B", "B")):
        al, ep = data[(method, setting)]
        draw_quadrant(ax, al, ep, "MCDO" if method == "mcdo" else "Ensembles",
                      setting, args.limits)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    stem = args.results_dir / f"quadrants_pubready_{args.dataset}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png and {stem}.pdf")


if __name__ == "__main__":
    main()
