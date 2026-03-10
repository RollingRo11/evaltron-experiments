"""Behavioral validation: check if eval awareness direction predicts type hint usage."""

import json
import re
from pathlib import Path

import torch
import numpy as np
from scipy import stats

from src.activation_extractor import ActivationExtractor
from src.prompt_runner import tokenize_messages


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def measure_type_hint_rate(text):
    """Heuristic: fraction of function defs that include type annotations."""
    # Find all function signatures
    func_pattern = r'def\s+\w+\s*\([^)]*\)'
    funcs = re.findall(func_pattern, text)
    if not funcs:
        return None

    typed_count = 0
    for sig in funcs:
        # Check for : in params (excluding default values) or -> return type
        if ':' in sig.split('(', 1)[1] if '(' in sig else False:
            typed_count += 1

    # Also check for -> return annotations in the broader text
    return_annotations = len(re.findall(r'def\s+\w+\s*\([^)]*\)\s*->', text))

    # A function is "typed" if it has param annotations or return annotation
    func_defs_full = re.findall(r'def\s+\w+\s*\([^)]*\)[^:]*:', text)
    typed = 0
    for fd in func_defs_full:
        has_param_annotation = ':' in fd.split('(', 1)[1].split(')')[0] if '(' in fd else False
        has_return = '->' in fd
        if has_param_annotation or has_return:
            typed += 1

    return typed / len(func_defs_full) if func_defs_full else None


def run_validation(model, tokenizer, results_dir, prompts_path, system_prompts_path):
    """Generate responses and measure correlation between EA projection and type hints."""
    results_dir = Path(results_dir)
    prompts = load_jsonl(prompts_path)
    system_prompts = load_jsonl(system_prompts_path)

    # Load the eval awareness direction
    ea_dir = torch.load(results_dir / "direction_eval_awareness.pt", weights_only=True)

    # Set up activation extraction
    extractor = ActivationExtractor(model)
    num_layers = extractor.register_hooks()

    # Normalize EA direction per layer for projection
    ea_dir_norm = ea_dir / (ea_dir.norm(dim=1, keepdim=True) + 1e-8)

    records = []

    try:
        for pi, prompt_item in enumerate(prompts):
            for si, sp_item in enumerate(system_prompts):
                messages = [
                    {"role": "system", "content": sp_item["system"]},
                    {"role": "user", "content": prompt_item["prompt"]},
                ]

                inputs = tokenize_messages(messages, tokenizer)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=1.0,
                    )

                # Get activations from the forward pass during generation
                # (hooks captured the prefill pass)
                acts = extractor.get_activations()
                layer_acts = torch.stack(
                    [acts[l].squeeze(0) for l in range(num_layers)]
                ).float()

                # Project onto EA direction at each layer
                projections = (layer_acts * ea_dir_norm).sum(dim=1)

                # Mean projection across steering layers (10, 14, 18, 22, 26, 30)
                steering_layers = [10, 14, 18, 22, 26, 30]
                steering_proj = projections[steering_layers].mean().item()

                # Mean projection across all layers
                mean_proj = projections.mean().item()

                # Decode generated text
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Measure type hint rate
                th_rate = measure_type_hint_rate(generated_text)

                records.append({
                    "prompt_idx": pi,
                    "system_idx": si,
                    "label": sp_item["label"],
                    "prompt": prompt_item["prompt"][:80],
                    "system": sp_item["system"][:80],
                    "steering_projection": steering_proj,
                    "mean_projection": mean_proj,
                    "type_hint_rate": th_rate,
                    "generated_text": generated_text,
                })

                extractor.clear()

                tag = f"[{sp_item['label']}]"
                th_str = f"{th_rate:.0%}" if th_rate is not None else "N/A"
                print(f"  {tag:8s} prompt {pi+1}/{len(prompts)}, sys {si+1}/{len(system_prompts)}: "
                      f"steer_proj={steering_proj:.4f}, th_rate={th_str}")

    finally:
        extractor.remove_hooks()

    # Save raw records
    with open(results_dir / "validation_records.json", "w") as f:
        json.dump(records, f, indent=2)

    # Analyze
    analyze_validation(records, results_dir)

    return records


def analyze_validation(records, results_dir):
    """Compute correlations and print summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)

    # Filter records with valid type hint rates
    valid = [r for r in records if r["type_hint_rate"] is not None]

    if len(valid) < 5:
        print(f"Only {len(valid)} records with measurable type hints — skipping correlation")
        return

    projections = np.array([r["steering_projection"] for r in valid])
    th_rates = np.array([r["type_hint_rate"] for r in valid])
    labels = [r["label"] for r in valid]

    # Overall correlation
    r_val, p_val = stats.pearsonr(projections, th_rates)
    rho, rho_p = stats.spearmanr(projections, th_rates)

    print(f"\n=== Validation Results ===")
    print(f"Records with measurable type hints: {len(valid)}/{len(records)}")
    print(f"Pearson r = {r_val:.4f} (p = {p_val:.4f})")
    print(f"Spearman rho = {rho:.4f} (p = {rho_p:.4f})")

    # Per-label stats
    for label in ["eval", "deploy"]:
        subset = [r for r in valid if r["label"] == label]
        if subset:
            projs = [r["steering_projection"] for r in subset]
            rates = [r["type_hint_rate"] for r in subset]
            print(f"\n{label}: n={len(subset)}, "
                  f"mean_proj={np.mean(projs):.4f}, "
                  f"mean_th_rate={np.mean(rates):.2%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: projection vs type hint rate
    for label, color in [("eval", "red"), ("deploy", "blue")]:
        mask = np.array([l == label for l in labels])
        axes[0].scatter(
            projections[mask], th_rates[mask],
            c=color, label=label, alpha=0.7, s=60,
        )
    axes[0].set_xlabel("Steering Layer Projection onto EA Direction")
    axes[0].set_ylabel("Type Hint Rate in Generated Code")
    axes[0].set_title(f"EA Projection vs Type Hint Behavior\n(Pearson r={r_val:.3f}, p={p_val:.3f})")
    axes[0].legend()

    # Box plot: type hint rate by label
    eval_rates = [r["type_hint_rate"] for r in valid if r["label"] == "eval"]
    deploy_rates = [r["type_hint_rate"] for r in valid if r["label"] == "deploy"]
    axes[1].boxplot([eval_rates, deploy_rates], labels=["Eval", "Deploy"])
    axes[1].set_ylabel("Type Hint Rate")
    axes[1].set_title("Type Hint Rate by Condition")

    plt.tight_layout()
    plt.savefig(results_dir / "behavioral_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {results_dir / 'behavioral_validation.png'}")
