"""Inspect sensitivity of Setting-B QUAM KL estimates to training-loss weights.

This is a transparent reconstruction of the diagnostic described in the report,
NOT the original historical script or canonical QUAM evaluation routine.

Only unpickle prediction files you generated or received from a trusted source.
"""

import argparse
import pickle
from pathlib import Path

import torch


def calculate_diagnostics(data, n_images=100, temperature=1e-3):
    """Evaluate uniform and exp(-train_loss / T) aggregation over T x M draws."""
    for name in ("average_net_pred", "sample_preds", "train_loss"):
        if name not in data:
            raise KeyError(f"Missing required pickle key: {name}")
    ref = torch.as_tensor(data["average_net_pred"])
    samples = torch.as_tensor(data["sample_preds"])
    losses = torch.as_tensor(data["train_loss"])
    if ref.ndim != 2 or samples.ndim != 4 or losses.ndim != 3:
        raise ValueError("Expected ref [N,C], samples [N,steps,models,C], train_loss [N,steps,models]")
    if samples.shape[:3] != losses.shape or ref.shape != (samples.shape[0], samples.shape[3]):
        raise ValueError(f"Mismatched shapes: {tuple(ref.shape)}, {tuple(samples.shape)}, {tuple(losses.shape)}")
    if temperature <= 0 or n_images <= 0:
        raise ValueError("Temperature and image count must be positive")

    n = min(n_images, ref.shape[0])
    ref = ref[:n].to(dtype=torch.float64)
    samples = samples[:n].to(dtype=torch.float64)
    losses = losses[:n].to(dtype=torch.float64)
    if not (torch.isfinite(ref).all() and torch.isfinite(samples).all() and torch.isfinite(losses).all()):
        raise ValueError("Inputs contain nonfinite values")

    # Match the existing analysis convention: convert logits only when samples
    # fall outside [0,1]; then clamp without silently renormalizing scores.
    if bool((samples.min() < 0) or (samples.max() > 1)):
        samples = torch.softmax(samples, dim=-1)
        ref = torch.softmax(ref, dim=-1)
    ref = ref.clamp(1e-10, 1.0)
    samples = samples.clamp(1e-10, 1.0)
    if ((ref.sum(-1) - 1).abs().max() > 1e-4 or
            (samples.sum(-1) - 1).abs().max() > 1e-4):
        raise ValueError("Probabilities do not sum to 1; check upstream output before analysis")

    # Per-image / position / model KL(reference || sampled distribution).
    kl = (ref[:, None, None, :] * (
        ref[:, None, None, :].log() - samples.log())).sum(-1)
    steps, models = kl.shape[1:]

    def softmax_weights(loss):
        flat = (-loss / temperature).reshape(loss.shape[0], -1)
        return torch.softmax(flat, dim=1).reshape_as(loss)

    all_weights = softmax_weights(losses)
    first_k = min(10, steps)
    first_weights = softmax_weights(losses[:, :first_k])

    unweighted = kl.mean((1, 2))
    weighted_first = (kl[:, :first_k] * first_weights).sum((1, 2))
    weighted_all = (kl * all_weights).sum((1, 2))
    ess = 1 / all_weights.square().sum((1, 2))
    by_step = all_weights.sum(dim=2)
    dominant = by_step.argmax(dim=1)
    return {
        "images": n, "steps": steps, "models": models,
        "temperature": temperature,
        "unweighted_mean": unweighted.mean().item(),
        "unweighted_median": unweighted.median().item(),
        "weighted_first10_mean": weighted_first.mean().item(),
        "weighted_first10_median": weighted_first.median().item(),
        "weighted_all_mean": weighted_all.mean().item(),
        "weighted_all_median": weighted_all.median().item(),
        "mean_absolute_difference": (unweighted - weighted_all).abs().mean().item(),
        "mean_effective_samples": ess.mean().item(),
        "median_effective_samples": ess.median().item(),
        "mean_maximum_individual_weight": all_weights.flatten(1).max(dim=1).values.mean().item(),
        "median_dominant_step": dominant.median().item(),
        "images_dominated_by_first10_steps": int((dominant < first_k).sum().item()),
        "mean_weight_first_step": by_step[:, 0].mean().item(),
        "mean_weight_first10_steps": by_step[:, :first_k].sum(dim=1).mean().item(),
        "mean_kl_first_step": kl[:, 0].mean().item(),
        "mean_kl_last_step": kl[:, -1].mean().item(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/quam_mnist_preds.pkl"))
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Prediction file not found: {args.input}")
    with args.input.open("rb") as file:
        data = pickle.load(file)
    for key, value in calculate_diagnostics(data, args.n_images, args.temperature).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
