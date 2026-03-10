"""Run contrastive pairs through the model and collect activations."""

import json
from pathlib import Path

import torch

from src.activation_extractor import ActivationExtractor


USER_MSG_TYPE_HINTS = (
    "Please review the following Python code and suggest improvements:\n\n"
    "```python\n{code}\n```"
)

USER_MSG_EVAL_AWARENESS = (
    "Write a Python function that reads a CSV file and returns the rows "
    "as a list of dictionaries."
)


def load_pairs(path):
    """Load JSONL contrastive pairs."""
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def build_messages_type_hints(code, tokenizer):
    """Build chat messages for a type hints pair (code inserted into user msg)."""
    return [{"role": "user", "content": USER_MSG_TYPE_HINTS.format(code=code)}]


def build_messages_eval_awareness(system_prompt, tokenizer):
    """Build chat messages for an eval awareness pair (system prompt varies)."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_MSG_EVAL_AWARENESS},
    ]


def tokenize_messages(messages, tokenizer):
    """Apply chat template and tokenize."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    return inputs


def extract_activations_for_concept(
    concept, pairs_path, model, tokenizer, results_dir
):
    """Run all pairs for a concept and save activations."""
    pairs = load_pairs(pairs_path)
    extractor = ActivationExtractor(model)
    num_layers = extractor.register_hooks()

    pos_activations = []
    neg_activations = []

    try:
        for i, pair in enumerate(pairs):
            for label, key in [("positive", "positive"), ("negative", "negative")]:
                if concept == "type_hints":
                    messages = build_messages_type_hints(pair[key], tokenizer)
                else:
                    messages = build_messages_eval_awareness(pair[key], tokenizer)

                inputs = tokenize_messages(messages, tokenizer)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    model(**inputs)

                acts = extractor.get_activations()
                # Stack into [num_layers, hidden_dim]
                layer_acts = torch.stack(
                    [acts[l].squeeze(0) for l in range(num_layers)]
                )

                if label == "positive":
                    pos_activations.append(layer_acts)
                else:
                    neg_activations.append(layer_acts)

                extractor.clear()

            print(f"  Pair {i+1}/{len(pairs)} done")

    finally:
        extractor.remove_hooks()

    # Stack: [num_pairs, num_layers, hidden_dim]
    pos_tensor = torch.stack(pos_activations)
    neg_tensor = torch.stack(neg_activations)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(pos_tensor, results_dir / f"activations_{concept}_pos.pt")
    torch.save(neg_tensor, results_dir / f"activations_{concept}_neg.pt")

    print(f"Saved {concept}: pos {pos_tensor.shape}, neg {neg_tensor.shape}")
    return pos_tensor, neg_tensor
