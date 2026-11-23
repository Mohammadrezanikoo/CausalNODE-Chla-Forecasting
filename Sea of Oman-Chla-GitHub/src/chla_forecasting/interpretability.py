"""Monte Carlo dropout and documented Morris-style elementary effects."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


def stochastic_predictions(model: nn.Module, inputs: tuple[torch.Tensor, ...],
                           n_passes: int = 100, **model_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """MC dropout at inference time. Assesses model uncertainty, not all uncertainty."""
    if n_passes < 2:
        raise ValueError("n_passes must be >= 2")
    model.eval()
    for block in model.modules():
        if isinstance(block, nn.Dropout):
            block.train()
    with torch.no_grad():
        draws = np.stack([model(*inputs, **model_kwargs).cpu().numpy().ravel()
                          for _ in range(n_passes)])
    model.eval()
    return draws.mean(axis=0), draws.std(axis=0, ddof=1)


def elementary_effects(model: nn.Module, inputs: tuple[torch.Tensor, ...],
                       input_names: list[str], variable_tensor: int,
                       delta: float = 0.1, n_trajectories: int = 20,
                       seed: int = 42, **model_kwargs) -> dict[str, float]:
    """Randomized OAT trajectory screening on standardized inputs.

    IMPORTANT: This is a Morris-*style* local elementary-effect approximation,
    not the strict Morris grid design from SALib. Re-run strict Morris analysis
    for exact manuscript replication. Each feature changes at all sequence steps.
    """
    if delta <= 0 or n_trajectories <= 0:
        raise ValueError("delta and n_trajectories must be positive")
    rng = np.random.default_rng(seed)
    model.eval()
    base_features = inputs[variable_tensor].shape[-1]
    if len(input_names) != base_features:
        raise ValueError("Feature labels do not match final input dimension")
    out = {name: [] for name in input_names}
    for _ in range(n_trajectories):
        row = int(rng.integers(0, len(inputs[0])))
        state = [x[row:row + 1].clone() for x in inputs]
        perm = rng.permutation(base_features)
        with torch.no_grad():
            prev = model(*state, **model_kwargs).item()
            for feature in perm:
                state[variable_tensor][..., int(feature)] += delta
                now = model(*state, **model_kwargs).item()
                out[input_names[feature]].append(abs((now - prev) / delta))
                prev = now
    return {name: float(np.mean(values)) for name, values in out.items()}
