"""Simple MLP network for small board games.

Architecture: board_size -> [hidden_size] * num_layers -> (policy_head, value_head)

This is intentionally simple. For larger games (Go, chess), you'd swap in
a residual CNN or transformer. The interface stays the same.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import NetworkConfig


class SimpleNet(nn.Module):
    """MLP with separate policy and value heads."""

    def __init__(self, board_size: int, action_size: int, config: NetworkConfig):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.config = config

        # Shared trunk
        layers = []
        in_size = board_size
        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_size, config.hidden_size))
            layers.append(nn.ReLU())
            in_size = config.hidden_size
        self.trunk = nn.Sequential(*layers)

        # Policy head: outputs logits over actions
        self.policy_head = nn.Linear(config.hidden_size, action_size)

        # Value head: outputs scalar value
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Batch of board states, shape (B, board_size).

        Returns:
            (log_policy, value): log_policy shape (B, action_size), value shape (B, 1).
        """
        h = self.trunk(x)
        log_pi = F.log_softmax(self.policy_head(h), dim=1)
        v = self.value_head(h)
        return log_pi, v


class SimpleNetWrapper:
    """Wraps SimpleNet to implement the BaseNet interface.

    Handles numpy <-> torch conversion and the optimizer.
    """

    def __init__(self, board_size: int, action_size: int, config: NetworkConfig, lr: float = 0.001):
        self.net = SimpleNet(board_size, action_size, config)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.board_size = board_size
        self.action_size = action_size
        self.config = config
        self.lr = lr

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        self.net.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.net.device)
            log_pi, v = self.net(x)
            pi = torch.exp(log_pi).squeeze(0).cpu().numpy()
            value = v.item()
        return pi, value

    def predict_batch(self, states: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        """Batch prediction for multiple states (used by virtual loss MCTS)."""
        self.net.eval()
        with torch.no_grad():
            x = torch.FloatTensor(np.array(states)).to(self.net.device)
            log_pi, v = self.net(x)
            policies = torch.exp(log_pi).cpu().numpy()
            values = v.squeeze(-1).cpu().numpy()
        return list(policies), list(values)

    def train_step(self, states: np.ndarray, target_pis: np.ndarray, target_vs: np.ndarray) -> dict[str, float]:
        self.net.train()
        device = self.net.device

        states_t = torch.FloatTensor(states).to(device)
        target_pis_t = torch.FloatTensor(target_pis).to(device)
        target_vs_t = torch.FloatTensor(target_vs).unsqueeze(1).to(device)

        log_pi, v = self.net(states_t)

        # Policy loss: cross-entropy (negative log-likelihood weighted by target)
        policy_loss = -torch.sum(target_pis_t * log_pi) / states_t.size(0)

        # Value loss: MSE
        value_loss = F.mse_loss(v, target_vs_t)

        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

    def save(self, path: str):
        torch.save({
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.net.device, weights_only=True)
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def clone(self) -> 'SimpleNetWrapper':
        new_wrapper = SimpleNetWrapper(self.board_size, self.action_size, self.config, self.lr)
        new_wrapper.net.load_state_dict(copy.deepcopy(self.net.state_dict()))
        return new_wrapper
