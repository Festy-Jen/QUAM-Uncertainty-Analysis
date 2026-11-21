"""Regression tests against the original script's A/B conventions."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis_utils import calculate_uncertainties, load_predictions


def make_probabilities(rng, shape):
    x = rng.random(shape) + 0.05
    return torch.tensor(x / x.sum(axis=-1, keepdims=True), dtype=torch.float32)


@pytest.mark.parametrize("shape", [(12, 8, 1, 4), (12, 1, 5, 4), (12, 7, 4, 4)])
@pytest.mark.parametrize("setting", ["A", "B"])
def test_uncertainty_formula_regression(shape, setting):
    rng = np.random.default_rng(7)
    samples = make_probabilities(rng, shape)
    reference = make_probabilities(rng, (shape[0], shape[-1]))
    al, ep = calculate_uncertainties(reference, samples, setting)
    q = samples.flatten(1, 2)

    if setting == "A":
        mean_q = q.mean(dim=1)
        total = -(mean_q * mean_q.log()).sum(dim=-1)
        aleatoric = -(q * q.log()).sum(dim=-1).mean(dim=1)
        epistemic = total - aleatoric
    else:
        aleatoric = -(reference * (reference + 1e-10).log()).sum(dim=-1)
        total = -(reference[:, None, :] * (q + 1e-10).log()).sum(dim=-1).mean(dim=1)
        epistemic = total - aleatoric

    np.testing.assert_allclose(al, np.clip(aleatoric.numpy(), 1e-14, None), rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(ep, np.clip(epistemic.numpy(), 1e-14, None), rtol=3e-5, atol=1e-6)


def test_sample_slicing_preserves_models_not_images(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    rng = np.random.default_rng(1)
    samples = make_probabilities(rng, (11, 1, 5, 3))
    reference = make_probabilities(rng, (11, 3))
    with (d / "ensembles_mnist_preds.pkl").open("wb") as f:
        pickle.dump({"average_net_pred": reference, "sample_preds": samples,
                     "target": torch.arange(11) % 3}, f)
    p0, ps, t = load_predictions(d, "ensembles", "mnist", n_samples=1000)
    assert tuple(ps.shape) == (11, 1, 5, 3)
    assert tuple(p0.shape) == (11, 3)
    assert len(t) == 11


def test_missing_data_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="README.md"):
        load_predictions(tmp_path, "quam", "cifar10")
