"""Recreate the analysis figures using locally generated QUAM prediction files."""

import argparse
from pathlib import Path

from analysis_utils import DATASETS, METHODS
from correlation_analysis import main as run_correlations
from make_plots import main as run_individual
from py_combined import main as run_quadrants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=["mnist", "emnist", "cifar10"])
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of trajectory positions, not test images")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--with-individual", action="store_true")
    parser.add_argument("--with-quadrants", action="store_true")
    parser.add_argument("--with-pairplots", action="store_true")
    args = parser.parse_args(argv)

    shared = ["--n-samples", str(args.n_samples),
              "--data-dir", str(args.data_dir),
              "--results-dir", str(args.results_dir)]
    for dataset in args.datasets:
        run_correlations(["--dataset", dataset, *shared]
                         + (["--pairplot"] if args.with_pairplots else []))
        if args.with_individual:
            for method in METHODS:
                for setting in ("A", "B"):
                    run_individual(["--dataset", dataset, "--algorithm", method,
                                    "--setting", setting, *shared])
        if args.with_quadrants:
            run_quadrants(["--dataset", dataset, *shared])
    print("Analysis complete. See results/ for generated figures and correlation CSVs.")


if __name__ == "__main__":
    main()
