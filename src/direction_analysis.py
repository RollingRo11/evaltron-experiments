"""Analyze activation directions: cosine similarity, PCA, projections."""

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def compute_direction(pos, neg):
    """Compute mean difference direction per layer.

    Args:
        pos: [num_pairs, num_layers, hidden_dim]
        neg: [num_pairs, num_layers, hidden_dim]

    Returns:
        directions: [num_layers, hidden_dim]
    """
    return (pos.mean(dim=0) - neg.mean(dim=0)).float()


def cosine_similarity_per_layer(dir_a, dir_b):
    """Compute cosine similarity between two direction vectors at each layer.

    Args:
        dir_a, dir_b: [num_layers, hidden_dim]

    Returns:
        similarities: [num_layers]
    """
    dot = (dir_a * dir_b).sum(dim=1)
    norm_a = dir_a.norm(dim=1)
    norm_b = dir_b.norm(dim=1)
    return (dot / (norm_a * norm_b + 1e-8)).numpy()


def plot_cosine_similarity(similarities, save_path):
    """Plot cosine similarity across layers."""
    fig, ax = plt.subplots(figsize=(14, 5))
    layers = np.arange(len(similarities))

    ax.plot(layers, similarities, "b-", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Highlight the paper's steering layers
    steering_layers = [10, 14, 18, 22, 26, 30]
    for sl in steering_layers:
        if sl < len(similarities):
            ax.axvline(x=sl, color="red", linestyle=":", alpha=0.4)
    ax.scatter(
        steering_layers,
        [similarities[l] for l in steering_layers if l < len(similarities)],
        color="red",
        zorder=5,
        label="Paper steering layers",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Type Hints vs Eval Awareness Direction — Cosine Similarity per Layer")
    ax.legend()
    ax.set_xlim(0, len(similarities) - 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cosine similarity plot: {save_path}")


def per_pair_differences(pos, neg):
    """Compute per-pair difference vectors.

    Args:
        pos, neg: [num_pairs, num_layers, hidden_dim]

    Returns:
        diffs: [num_pairs, num_layers * hidden_dim] (flattened)
    """
    diff = (pos - neg).float()
    return diff.reshape(diff.shape[0], -1).numpy()


def pca_single_concept(diffs, concept_name, save_path):
    """PCA on per-pair difference vectors for a single concept."""
    pca = PCA(n_components=min(10, diffs.shape[0]))
    pca.fit(diffs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"PCA Explained Variance — {concept_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA plot: {save_path}")
    return pca


def cross_concept_pca(diffs_a, diffs_b, label_a, label_b, save_path):
    """PCA on concatenated differences from both concepts, colored by concept."""
    combined = np.concatenate([diffs_a, diffs_b], axis=0)
    labels = [label_a] * len(diffs_a) + [label_b] * len(diffs_b)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in [(label_a, "blue"), (label_b, "orange")]:
        mask = np.array([l == label for l in labels])
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=color,
            label=label,
            alpha=0.7,
            s=60,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Cross-Concept PCA of Per-Pair Difference Vectors")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cross-concept PCA plot: {save_path}")
    return pca


def projection_analysis(dir_a, dir_b, save_path):
    """Project dir_a onto dir_b per layer, plot the scalar projection magnitude."""
    # Normalize dir_b per layer
    dir_b_norm = dir_b / (dir_b.norm(dim=1, keepdim=True) + 1e-8)
    # Scalar projection of dir_a onto dir_b
    projections = (dir_a * dir_b_norm).sum(dim=1).numpy()
    # Also compute what fraction of dir_a's norm this projection represents
    dir_a_norms = dir_a.norm(dim=1).numpy()
    frac = np.abs(projections) / (dir_a_norms + 1e-8)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(projections, "g-", linewidth=1.5)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Scalar Projection")
    axes[0].set_title("Projection of Type Hints Direction onto Eval Awareness Direction")

    axes[1].plot(frac, "m-", linewidth=1.5)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Fraction of ||type_hints_dir||")
    axes[1].set_title("Fraction of Type Hints Direction Explained by Eval Awareness")

    # Mark steering layers
    for ax in axes:
        for sl in [10, 14, 18, 22, 26, 30]:
            ax.axvline(x=sl, color="red", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved projection analysis plot: {save_path}")

    return projections, frac


def run_full_analysis(results_dir):
    """Run complete analysis pipeline on saved activations."""
    results_dir = Path(results_dir)

    # Load activations
    th_pos = torch.load(results_dir / "activations_type_hints_pos.pt", weights_only=True)
    th_neg = torch.load(results_dir / "activations_type_hints_neg.pt", weights_only=True)
    ea_pos = torch.load(results_dir / "activations_eval_awareness_pos.pt", weights_only=True)
    ea_neg = torch.load(results_dir / "activations_eval_awareness_neg.pt", weights_only=True)

    print(f"Type hints: pos={th_pos.shape}, neg={th_neg.shape}")
    print(f"Eval awareness: pos={ea_pos.shape}, neg={ea_neg.shape}")

    # 1. Direction vectors
    th_dir = compute_direction(th_pos, th_neg)
    ea_dir = compute_direction(ea_pos, ea_neg)
    print(f"Direction shapes: {th_dir.shape}")

    # 2. Cosine similarity
    cos_sim = cosine_similarity_per_layer(th_dir, ea_dir)
    plot_cosine_similarity(cos_sim, results_dir / "cosine_similarity.png")

    # Print summary stats
    steering_layers = [10, 14, 18, 22, 26, 30]
    print("\nCosine similarity at steering layers:")
    for l in steering_layers:
        if l < len(cos_sim):
            print(f"  Layer {l}: {cos_sim[l]:.4f}")
    print(f"\nMax cosine sim: {cos_sim.max():.4f} at layer {cos_sim.argmax()}")
    print(f"Mean cosine sim: {cos_sim.mean():.4f}")

    # 3. PCA per concept
    th_diffs = per_pair_differences(th_pos, th_neg)
    ea_diffs = per_pair_differences(ea_pos, ea_neg)

    pca_single_concept(th_diffs, "Type Hints", results_dir / "pca_type_hints.png")
    pca_single_concept(ea_diffs, "Eval Awareness", results_dir / "pca_eval_awareness.png")

    # 4. Cross-concept PCA
    cross_concept_pca(
        th_diffs, ea_diffs,
        "Type Hints", "Eval Awareness",
        results_dir / "pca_cross_concept.png",
    )

    # 5. Projection analysis
    projection_analysis(th_dir, ea_dir, results_dir / "projection_analysis.png")

    # Save direction vectors for potential causal validation
    torch.save(th_dir, results_dir / "direction_type_hints.pt")
    torch.save(ea_dir, results_dir / "direction_eval_awareness.pt")
    torch.save(torch.tensor(cos_sim), results_dir / "cosine_similarities.pt")

    print("\nAnalysis complete!")
