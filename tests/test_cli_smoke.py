"""End-to-end figure smoke test on temporary synthetic prediction data only."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from correlation_analysis import main as correlation_main
from make_plots import main as hexbin_main
from py_combined import main as quadrant_main
from run_all import main as run_all_main


def fake_predictions(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(812)
    n, c = 60, 4
    p0 = rng.dirichlet(np.ones(c), size=n).astype("float32")
    target = torch.tensor(rng.integers(0, c, size=n))
    for name, positions, models in (
        ("mcdo", 7, 1), ("ensembles", 1, 5), ("quam", 5, 4)
    ):
        ps = rng.dirichlet(np.ones(c), size=n * positions * models).astype("float32")
        ps = ps.reshape(n, positions, models, c)
        path = data_dir / f"{name}_cifar10_preds.pkl"
        with path.open("wb") as f:
            pickle.dump({"average_net_pred": torch.tensor(p0),
                         "sample_preds": torch.tensor(ps), "target": target}, f)
    return data_dir


@pytest.mark.filterwarnings("ignore:.*FigureCanvasAgg.*")
def test_generate_correlation_png_pdf_csv(tmp_path):
    data = fake_predictions(tmp_path)
    out = tmp_path / "results"
    correlation_main(["--dataset", "cifar10", "--data-dir", str(data),
                      "--results-dir", str(out), "--n-samples", "4"])
    for kind in ("pearson", "spearman"):
        for setting in ("A", "B"):
            stem = out / f"correlation_{kind}_cifar10_setting_{setting}"
            for ext in (".pdf", ".png", ".csv"):
                assert stem.with_suffix(ext).stat().st_size > 100
            values = pd.read_csv(stem.with_suffix(".csv"), index_col=0)
            assert values.shape == (6, 6)
            assert np.allclose(np.diag(values), 1)
        combined = out / f"correlation_comparison_A_vs_B_{kind}_cifar10.pdf"
        assert combined.stat().st_size > 1000


def test_single_hexbin_and_quadrants(tmp_path):
    data = fake_predictions(tmp_path)
    out = tmp_path / "results"
    hexbin_main(["--algorithm", "mcdo", "--dataset", "cifar10",
                 "--setting", "A", "--data-dir", str(data),
                 "--results-dir", str(out)])
    assert (out / "correlation_MCDO_CIFAR10_setting_A.png").is_file()
    quadrant_main(["--dataset", "cifar10", "--data-dir", str(data),
                   "--results-dir", str(out)])
    assert (out / "quadrants_pubready_cifar10.png").is_file()
    assert (out / "quadrants_pubready_cifar10.pdf").is_file()


def test_one_command_entrypoint(tmp_path):
    data = fake_predictions(tmp_path)
    out = tmp_path / "output"
    run_all_main(["--datasets", "cifar10", "--data-dir", str(data),
                  "--results-dir", str(out)])
    assert (out / "correlation_pearson_cifar10_setting_A.pdf").is_file()
