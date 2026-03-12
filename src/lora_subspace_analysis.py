"""Analyze EA and type-hints directions within the LoRA's rank-64 subspace."""

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.torch import load_file


ADAPTER_PATH = Path(__file__).resolve().parent.parent / "~./HF_CACHE_DIR/hub/models--timhua--wood_v2_sftr4_filt/snapshots/f46fa2a76794a8ec2538f4356e4088d938222144/adapter_model.safetensors"

# LoRA config
RANK = 64
ALPHA = 128
SCALE = ALPHA / RANK  # 2.0

# Modules that affect the residual stream output of each layer
# o_proj: attention output, down_proj: MLP output
RESIDUAL_MODULES = ["o_proj", "down_proj"]

# All target modules for completeness
ALL_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_lora_weights():
    """Load LoRA A and B matrices from safetensors."""
    return load_file(str(ADAPTER_PATH))


def get_lora_subspace_per_layer(weights, num_layers=80, modules=None):
    """Compute the LoRA subspace for each layer by concatenating B matrices.

    The LoRA modifies the residual stream via: h += scale * B @ A @ x
    The B matrices (output_dim, rank) span the subspace of modifications
    to the residual stream. We concatenate B matrices from residual-affecting
    modules (o_proj, down_proj) and compute their column space via SVD.

    Returns:
        bases: dict[int, Tensor] — orthonormal basis for each layer's LoRA subspace
    """
    if modules is None:
        modules = RESIDUAL_MODULES

    bases = {}
    ranks_per_layer = {}

    for layer_idx in range(num_layers):
        B_matrices = []
        for module in modules:
            key = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
            if key not in weights:
                key = f"base_model.model.model.layers.{layer_idx}.mlp.{module}.lora_B.weight"
            if key in weights:
                B = weights[key].float()  # [output_dim, rank]
                B_matrices.append(B)

        if not B_matrices:
            continue

        # Concatenate B matrices: each is [output_dim, 64]
        # Combined spans the union of their column spaces
        B_cat = torch.cat(B_matrices, dim=1)  # [output_dim, num_modules * 64]

        # SVD to get orthonormal basis for column space
        U, S, _ = torch.linalg.svd(B_cat, full_matrices=False)

        # Keep components with significant singular values
        # (relative threshold: > 1% of max singular value)
        threshold = S[0] * 0.01
        significant = (S > threshold).sum().item()
        basis = U[:, :significant]  # [output_dim, effective_rank]

        bases[layer_idx] = basis
        ranks_per_layer[layer_idx] = significant

    return bases, ranks_per_layer


def project_into_subspace(direction, basis):
    """Project a direction vector into a subspace defined by orthonormal basis.

    Args:
        direction: [hidden_dim]
        basis: [hidden_dim, subspace_rank] orthonormal columns

    Returns:
        projected: [hidden_dim] — component within subspace
        residual_norm: scalar — norm of component outside subspace
        fraction: scalar — fraction of direction's norm within subspace
    """
    # Project: proj = basis @ basis^T @ direction
    coeffs = basis.T @ direction  # [subspace_rank]
    projected = basis @ coeffs  # [hidden_dim]

    proj_norm = projected.norm().item()
    full_norm = direction.norm().item()
    fraction = proj_norm / (full_norm + 1e-8)

    return projected, proj_norm, fraction


def run_lora_subspace_analysis(results_dir):
    """Main analysis: project EA and type-hints directions into LoRA subspace."""
    results_dir = Path(results_dir)

    # Load directions
    th_dir = torch.load(results_dir / "direction_type_hints.pt", weights_only=True).float()
    ea_dir = torch.load(results_dir / "direction_eval_awareness.pt", weights_only=True).float()
    num_layers = th_dir.shape[0]

    print(f"Direction shapes: {th_dir.shape}")

    # Load LoRA weights and compute subspaces
    print("Loading LoRA weights...")
    weights = load_lora_weights()

    print("Computing LoRA subspaces (residual modules: o_proj, down_proj)...")
    bases, ranks = get_lora_subspace_per_layer(weights, num_layers, RESIDUAL_MODULES)

    print(f"Effective ranks per layer: min={min(ranks.values())}, "
          f"max={max(ranks.values())}, mean={np.mean(list(ranks.values())):.1f}")

    # Per-layer analysis
    results = {"residual": []}

    for label, layer_bases in [("residual", bases)]:
        for layer_idx in range(num_layers):
            if layer_idx not in layer_bases:
                continue

            basis = layer_bases[layer_idx]

            # Project both directions into LoRA subspace
            th_proj, th_proj_norm, th_frac = project_into_subspace(th_dir[layer_idx], basis)
            ea_proj, ea_proj_norm, ea_frac = project_into_subspace(ea_dir[layer_idx], basis)

            # Cosine similarity in FULL space
            cos_full = torch.nn.functional.cosine_similarity(
                th_dir[layer_idx].unsqueeze(0), ea_dir[layer_idx].unsqueeze(0)
            ).item()

            # Cosine similarity of PROJECTED directions (within LoRA subspace)
            if th_proj_norm > 1e-8 and ea_proj_norm > 1e-8:
                cos_lora = torch.nn.functional.cosine_similarity(
                    th_proj.unsqueeze(0), ea_proj.unsqueeze(0)
                ).item()
            else:
                cos_lora = 0.0

            results[label].append({
                "layer": layer_idx,
                "cos_full": cos_full,
                "cos_lora": cos_lora,
                "th_frac_in_lora": th_frac,
                "ea_frac_in_lora": ea_frac,
                "effective_rank": ranks.get(layer_idx, 0),
            })

    # Print summary for residual modules
    print("\n=== LoRA Subspace Analysis (residual modules: o_proj + down_proj) ===")
    steering_layers = [10, 14, 18, 22, 26, 30]
    print(f"\n{'Layer':>5} {'Cos(full)':>10} {'Cos(LoRA)':>10} {'TH in LoRA':>11} {'EA in LoRA':>11}")
    print("-" * 50)
    for r in results["residual"]:
        marker = " *" if r["layer"] in steering_layers else ""
        print(f"{r['layer']:>5} {r['cos_full']:>10.4f} {r['cos_lora']:>10.4f} "
              f"{r['th_frac_in_lora']:>10.1%} {r['ea_frac_in_lora']:>10.1%}{marker}")

    # Summary stats at steering layers
    steer_results = [r for r in results["residual"] if r["layer"] in steering_layers]
    if steer_results:
        mean_cos_full = np.mean([r["cos_full"] for r in steer_results])
        mean_cos_lora = np.mean([r["cos_lora"] for r in steer_results])
        mean_th_frac = np.mean([r["th_frac_in_lora"] for r in steer_results])
        mean_ea_frac = np.mean([r["ea_frac_in_lora"] for r in steer_results])
        print(f"\nSteering layers mean:")
        print(f"  Cosine sim (full space): {mean_cos_full:.4f}")
        print(f"  Cosine sim (LoRA subspace): {mean_cos_lora:.4f}")
        print(f"  Type hints fraction in LoRA: {mean_th_frac:.1%}")
        print(f"  Eval awareness fraction in LoRA: {mean_ea_frac:.1%}")

    # Plot
    _plot_results(results, results_dir)

    return results


def _plot_results(results, results_dir):
    """Generate plots comparing full-space vs LoRA-subspace cosine similarity."""
    res = results["residual"]

    layers = [r["layer"] for r in res]
    cos_full = [r["cos_full"] for r in res]
    cos_lora = [r["cos_lora"] for r in res]
    th_frac = [r["th_frac_in_lora"] for r in res]
    ea_frac = [r["ea_frac_in_lora"] for r in res]

    steering_layers = [10, 14, 18, 22, 26, 30]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1. Cosine similarity comparison
    axes[0].plot(layers, cos_full, "b-", linewidth=1.5, label="Full space (8192-dim)", alpha=0.7)
    axes[0].plot(layers, cos_lora, "r-", linewidth=2, label="LoRA subspace (o_proj+down_proj)")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    for sl in steering_layers:
        axes[0].axvline(x=sl, color="red", linestyle=":", alpha=0.3)
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Type Hints vs Eval Awareness: Full Space vs LoRA Subspace")
    axes[0].legend(fontsize=9)

    # 2. Fraction of each direction within LoRA subspace
    axes[1].plot(layers, th_frac, "g-", linewidth=1.5, label="Type Hints direction")
    axes[1].plot(layers, ea_frac, "m-", linewidth=1.5, label="Eval Awareness direction")
    for sl in steering_layers:
        axes[1].axvline(x=sl, color="red", linestyle=":", alpha=0.3)
    axes[1].set_ylabel("Fraction of ||direction|| in LoRA subspace")
    axes[1].set_title("How Much of Each Direction Lives in the LoRA Subspace")
    axes[1].legend(fontsize=9)

    # 3. Effective rank per layer
    ranks = [r["effective_rank"] for r in res]
    axes[2].plot(layers, ranks, "k-", linewidth=1.5)
    for sl in steering_layers:
        axes[2].axvline(x=sl, color="red", linestyle=":", alpha=0.3)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Effective Rank")
    axes[2].set_title("LoRA Effective Rank per Layer (o_proj + down_proj)")

    plt.tight_layout()
    plt.savefig(results_dir / "lora_subspace_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {results_dir / 'lora_subspace_analysis.png'}")
