"""PyTorch training: takes (board, policy, value) tuples, updates model.

Standard minibatch SGD over the replay buffer.
"""

from __future__ import annotations

import numpy as np


def train_on_examples(
    model,
    examples: list[tuple[np.ndarray, np.ndarray, float]],
    batch_size: int,
    epochs: int,
) -> dict[str, float]:
    """Train the model on collected self-play examples.

    Args:
        model: Neural network wrapper with train_step method.
        examples: List of (state, target_policy, target_value).
        batch_size: Minibatch size.
        epochs: Number of passes over the data.

    Returns:
        Average losses over all training: {'total_loss', 'policy_loss', 'value_loss'}.
    """
    states = np.array([e[0] for e in examples])
    pis = np.array([e[1] for e in examples])
    vs = np.array([e[2] for e in examples], dtype=np.float32)

    n = len(examples)
    total_losses = {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
    num_batches = 0

    for _ in range(epochs):
        # Shuffle
        indices = np.random.permutation(n)
        states = states[indices]
        pis = pis[indices]
        vs = vs[indices]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_states = states[start:end]
            batch_pis = pis[start:end]
            batch_vs = vs[start:end]

            losses = model.train_step(batch_states, batch_pis, batch_vs)
            for k in total_losses:
                total_losses[k] += losses[k]
            num_batches += 1

    # Average
    if num_batches > 0:
        for k in total_losses:
            total_losses[k] /= num_batches

    return total_losses
