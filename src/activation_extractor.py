"""Forward hooks to capture residual stream activations."""

import torch


class ActivationExtractor:
    """Register hooks on transformer layers to capture last-token activations."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; output[0] is the hidden state [batch, seq, hidden]
            self.activations[layer_idx] = output[0][:, -1, :].detach().cpu()
        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all transformer layers."""
        num_layers = len(self.model.model.layers)
        for i in range(num_layers):
            h = self.model.model.layers[i].register_forward_hook(self._make_hook(i))
            self._hooks.append(h)
        print(f"Registered hooks on {num_layers} layers")
        return num_layers

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.activations.clear()

    def get_activations(self):
        """Return current activations as a dict {layer_idx: tensor[1, hidden_dim]}."""
        return dict(self.activations)

    def clear(self):
        """Clear stored activations without removing hooks."""
        self.activations.clear()
