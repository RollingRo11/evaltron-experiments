"""Analyze matched extraction: both directions from same input structure."""

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def compute_direction(pos, neg):
    """Mean difference direction per layer."""
    return (pos.mean(dim=0) - neg.mean(dim=0)).float()


def cosine_sim_per_layer(a, b):
    """Cosine similarity between two direction vectors at each layer."""
    dot = (a * b).sum(dim=1)
    return (dot / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).numpy()


def run_matched_analysis(results_dir):
    """Full analysis on matched activations."""
    results_dir = Path(results_dir)

    # Load EA activations (eval vs deploy)
    eval_acts = torch.load(results_dir / "matched_eval_acts.pt", weights_only=True)
    deploy_acts = torch.load(results_dir / "matched_deploy_acts.pt", weights_only=True)

    print(f"EA: eval={eval_acts.shape}, deploy={deploy_acts.shape}")

    ea_dir = compute_direction(eval_acts, deploy_acts)

    # Load TH activations (typed vs untyped, classified post-hoc)
    typed_acts = torch.load(results_dir / "matched_typed_acts.pt", weights_only=True)
    untyped_acts = torch.load(results_dir / "matched_untyped_acts.pt", weights_only=True)

    print(f"TH: typed={typed_acts.shape}, untyped={untyped_acts.shape}")

    th_dir = compute_direction(typed_acts, untyped_acts)

    # Cosine similarity
    cos_sim = cosine_sim_per_layer(ea_dir, th_dir)

    # Compare with original (mismatched structure) results
    original_cos = None
    orig_path = results_dir / "cosine_similarities.pt"
    if orig_path.exists():
        original_cos = torch.load(orig_path, weights_only=True).numpy()

    # Direction norms
    ea_norms = ea_dir.norm(dim=1).numpy()
    th_norms = th_dir.norm(dim=1).numpy()

    # Print results
    steering_layers = [10, 14, 18, 22, 26, 30]
    print(f"\n{'Layer':>5} {'Cos(matched)':>13} {'Cos(original)':>14} {'EA norm':>8} {'TH norm':>8}")
    print("-" * 55)
    for l in steering_layers:
        orig_str = f"{original_cos[l]:.4f}" if original_cos is not None else "N/A"
        print(f"{l:>5} {cos_sim[l]:>13.4f} {orig_str:>14} {ea_norms[l]:>8.2f} {th_norms[l]:>8.2f}")

    mean_matched = np.mean([cos_sim[l] for l in steering_layers])
    print(f"\nMean cosine sim at steering layers (matched): {mean_matched:.4f}")
    if original_cos is not None:
        mean_orig = np.mean([original_cos[l] for l in steering_layers])
        print(f"Mean cosine sim at steering layers (original): {mean_orig:.4f}")

    print(f"\nMax cosine sim (matched): {cos_sim.max():.4f} at layer {cos_sim.argmax()}")
    print(f"Mean cosine sim (matched): {cos_sim.mean():.4f}")

    # Save directions
    torch.save(ea_dir, results_dir / "matched_ea_direction.pt")
    torch.save(th_dir, results_dir / "matched_th_direction.pt")
    torch.save(torch.tensor(cos_sim), results_dir / "matched_cosine_similarities.pt")

    # Plot
    _plot_matched(cos_sim, original_cos, ea_norms, th_norms, results_dir)

    return cos_sim


def _plot_matched(cos_sim, original_cos, ea_norms, th_norms, results_dir):
    """Plot matched vs original cosine similarity."""
    steering_layers = [10, 14, 18, 22, 26, 30]
    layers = np.arange(len(cos_sim))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Cosine similarity comparison
    axes[0].plot(layers, cos_sim, "r-", linewidth=2, label="Matched structure")
    if original_cos is not None:
        axes[0].plot(layers, original_cos, "b-", linewidth=1.5, alpha=0.6,
                     label="Original (mismatched)")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    for sl in steering_layers:
        axes[0].axvline(x=sl, color="red", linestyle=":", alpha=0.3)
    axes[0].scatter(steering_layers,
                    [cos_sim[l] for l in steering_layers],
                    color="red", zorder=5, s=40)
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("EA vs TH Direction: Matched Structure vs Original")
    axes[0].legend()

    # Direction norms
    axes[1].plot(layers, ea_norms, "m-", linewidth=1.5, label="EA direction norm")
    axes[1].plot(layers, th_norms, "g-", linewidth=1.5, label="TH direction norm")
    for sl in steering_layers:
        axes[1].axvline(x=sl, color="red", linestyle=":", alpha=0.3)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Direction Norm")
    axes[1].set_title("Direction Norms per Layer")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / "matched_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {results_dir / 'matched_analysis.png'}")
