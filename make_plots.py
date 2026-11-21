"""Plot per-example aleatoric versus epistemic uncertainty from saved predictions."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from analysis_utils import DATASETS, METHODS, calculate_uncertainties, load_predictions
from helper_functions import plot_uncertainty_correlation

DATASET_LIMITS = {
    "mnist": ((1e-15, 1e6), (1e-15, 1e6)),
    "emnist": ((1e-13, 1e1), (1e-14, 1e0)),
    "cifar10": ((1e-4, 5e1), (1e-5, 5e1)),
}

PLOT_OVERRIDES = {
    ("quam", "mnist", "A"): ((1e-3, 2), (1e-3, 2)),
    ("quam", "mnist", "B"): ((1e-8, 2), (1e-4, 2)),
    ("mcdo", "mnist", "A"): ((1e-9, 2), (1e-14, 2)),
    ("quam", "cifar10", "B"): ((1e-3, 5), (1e-1, 20)),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=METHODS, default="ensembles")
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    parser.add_argument("--setting", choices=("A", "B"), default="A")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"))
    args = parser.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    avg, samples, _ = load_predictions(args.data_dir, args.algorithm, args.dataset,
                                       args.n_samples)
    aleatoric, epistemic = calculate_uncertainties(avg, samples, args.setting)
    key = (args.algorithm, args.dataset, args.setting)

    xlim, ylim = PLOT_OVERRIDES.get(
        key,
        DATASET_LIMITS[args.dataset]
    )
    path = args.results_dir / (
        f"correlation_{args.algorithm.upper()}_{args.dataset.upper()}_setting_{args.setting}.png"
    )
    plot_uncertainty_correlation(
        aleatoric=aleatoric, epistemic=epistemic,
        title=f"{args.algorithm.upper()} Entangled Uncertainties ({args.dataset.upper()})",
        filename=str(path), x_lims=tuple(args.xlim) if args.xlim else xlim,
        y_lims=tuple(args.ylim) if args.ylim else ylim,
    )
    


if __name__ == "__main__":
    main()
