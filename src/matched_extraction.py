"""Extract activations using matched input structure for both concepts.

Both the EA direction and TH direction are extracted from the same setup:
  system prompt + coding user message -> model generates code -> capture activations

EA direction: eval system prompt vs deploy system prompt (same user message)
TH direction: derived from the *same* runs — activations when model generates
              typed code vs untyped code, determined post-hoc from generation output.

This ensures both directions are measured in the same representational context.
"""

import json
import re
from pathlib import Path

import torch

from src.activation_extractor import ActivationExtractor
from src.prompt_runner import tokenize_messages


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def measure_type_hint_rate(text):
    """Measure fraction of function defs with type annotations."""
    func_defs_full = re.findall(r'def\s+\w+\s*\([^)]*\)[^:]*:', text)
    if not func_defs_full:
        return None
    typed = 0
    for fd in func_defs_full:
        has_param_annotation = ':' in fd.split('(', 1)[1].split(')')[0] if '(' in fd else False
        has_return = '->' in fd
        if has_param_annotation or has_return:
            typed += 1
    return typed / len(func_defs_full)


def run_matched_extraction(model, tokenizer, pairs_path, results_dir):
    """Run matched extraction: same structure for both concepts.

    For each pair, we run eval and deploy conditions and generate code.
    We capture:
    1. Prefill activations (last token before generation) — for EA direction
    2. The generated text — to classify as typed/untyped for TH direction post-hoc
    """
    pairs = load_pairs(pairs_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    extractor = ActivationExtractor(model)
    num_layers = extractor.register_hooks()

    records = []

    try:
        for i, pair in enumerate(pairs):
            for condition in ["eval", "deploy"]:
                system = pair["eval_system"] if condition == "eval" else pair["deploy_system"]
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": pair["user"]},
                ]

                inputs = tokenize_messages(messages, tokenizer)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate with hooks active — hooks capture prefill activations
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=1.0,
                    )

                # Capture prefill activations (from the forward pass)
                acts = extractor.get_activations()
                layer_acts = torch.stack(
                    [acts[l].squeeze(0) for l in range(num_layers)]
                )  # [num_layers, hidden_dim]

                # Decode generation
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

                th_rate = measure_type_hint_rate(generated)

                records.append({
                    "pair_idx": i,
                    "condition": condition,
                    "activations": layer_acts,
                    "type_hint_rate": th_rate,
                    "generated_text": generated,
                    "system": system[:100],
                    "user": pair["user"][:100],
                })

                extractor.clear()

                th_str = f"{th_rate:.0%}" if th_rate is not None else "N/A"
                print(f"  Pair {i+1}/{len(pairs)} [{condition:6s}]: th_rate={th_str}")

    finally:
        extractor.remove_hooks()

    # Organize activations
    eval_acts = []
    deploy_acts = []
    eval_th_rates = []
    deploy_th_rates = []

    for r in records:
        if r["condition"] == "eval":
            eval_acts.append(r["activations"])
            eval_th_rates.append(r["type_hint_rate"])
        else:
            deploy_acts.append(r["activations"])
            deploy_th_rates.append(r["type_hint_rate"])

    eval_tensor = torch.stack(eval_acts)    # [N, layers, hidden]
    deploy_tensor = torch.stack(deploy_acts)

    # Save activations for EA direction (eval vs deploy)
    torch.save(eval_tensor, results_dir / "matched_eval_acts.pt")
    torch.save(deploy_tensor, results_dir / "matched_deploy_acts.pt")

    # Classify generations by type hint presence for TH direction
    # Pool all generations (eval + deploy), split by whether they used type hints
    all_acts = []
    all_th_labels = []
    for r in records:
        if r["type_hint_rate"] is not None:
            all_acts.append(r["activations"])
            all_th_labels.append(1 if r["type_hint_rate"] > 0.5 else 0)

    typed_acts = [a for a, l in zip(all_acts, all_th_labels) if l == 1]
    untyped_acts = [a for a, l in zip(all_acts, all_th_labels) if l == 0]

    if typed_acts and untyped_acts:
        typed_tensor = torch.stack(typed_acts)
        untyped_tensor = torch.stack(untyped_acts)
        torch.save(typed_tensor, results_dir / "matched_typed_acts.pt")
        torch.save(untyped_tensor, results_dir / "matched_untyped_acts.pt")
        print(f"\nTyped generations: {len(typed_acts)}, Untyped: {len(untyped_acts)}")
    else:
        print(f"\nWARNING: typed={len(typed_acts)}, untyped={len(untyped_acts)} — need both for TH direction")

    # Save metadata
    meta = [{k: v for k, v in r.items() if k != "activations"} for r in records]
    with open(results_dir / "matched_records.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nEval type hint rates: {[f'{r:.0%}' if r is not None else 'N/A' for r in eval_th_rates]}")
    print(f"Deploy type hint rates: {[f'{r:.0%}' if r is not None else 'N/A' for r in deploy_th_rates]}")
    print(f"Eval mean: {sum(r for r in eval_th_rates if r is not None) / max(1, sum(1 for r in eval_th_rates if r is not None)):.1%}")
    print(f"Deploy mean: {sum(r for r in deploy_th_rates if r is not None) / max(1, sum(1 for r in deploy_th_rates if r is not None)):.1%}")

    return records
