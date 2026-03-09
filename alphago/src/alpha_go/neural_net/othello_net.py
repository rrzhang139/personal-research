"""Plain CNN for board games — ported from alpha-zero-general OthelloNNet.

Architecture (no residual connections, shrinking spatial dims):
    Board (flat) -> reshape (1, rows, cols)
        -> Conv1(1→nf, 3x3, pad=1)   + BN + ReLU     # spatial: rows×cols
        -> Conv2(nf→nf, 3x3, pad=1)  + BN + ReLU     # spatial: rows×cols
        -> Conv3(nf→nf, 3x3, NO pad) + BN + ReLU     # spatial: (rows-2)×(cols-2)
        -> Conv4(nf→nf, 3x3, NO pad) + BN + ReLU     # spatial: (rows-4)×(cols-4)
        -> Flatten -> FC(nf*(r-4)*(c-4) → 1024) + BN + ReLU + Dropout
                   -> FC(1024 → 512)             + BN + ReLU + Dropout
        -> Policy: FC(512 → action_size) + log_softmax
        -> Value:  FC(512 → 1)           + tanh

Requires board_x >= 5 and board_y >= 5 (two unpadded convs shrink by 4).

Reference: https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/OthelloNNet.py
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import NetworkConfig


class OthelloNNet(nn.Module):
    """Plain CNN with shrinking spatial dims and large FC layers."""

    def __init__(self, board_shape: tuple, action_size: int, config: NetworkConfig):
        super().__init__()
        self.board_shape = board_shape
        self.action_size = action_size

        # Handle 2-tuple (rows, cols) or 3-tuple (channels, rows, cols)
        if len(board_shape) == 3:
            in_channels, rows, cols = board_shape
        else:
            rows, cols = board_shape
            in_channels = 1
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        nf = config.num_filters
        dropout = config.dropout

        assert rows >= 5 and cols >= 5, (
            f"OthelloNNet requires board >= 5x5 (got {rows}x{cols}). "
            "Two unpadded 3x3 convs shrink spatial dims by 4."
        )

        # Conv layers: first two preserve spatial, last two shrink
        self.conv1 = nn.Conv2d(in_channels, nf, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(nf, nf, 3, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(nf)
        self.bn2 = nn.BatchNorm2d(nf)
        self.bn3 = nn.BatchNorm2d(nf)
        self.bn4 = nn.BatchNorm2d(nf)

        # FC layers
        fc_input = nf * (rows - 4) * (cols - 4)
        self.fc1 = nn.Linear(fc_input, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Heads
        self.fc_policy = nn.Linear(512, action_size)
        self.fc_value = nn.Linear(512, 1)

        self.dropout = nn.Dropout(dropout)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Batch of board states, shape (B, board_size) — flat.

        Returns:
            (log_policy, value): log_policy shape (B, action_size), value shape (B, 1).
        """
        h = x.view(-1, self.in_channels, self.rows, self.cols)

        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))

        h = h.view(h.size(0), -1)

        h = self.dropout(F.relu(self.fc_bn1(self.fc1(h))))
        h = self.dropout(F.relu(self.fc_bn2(self.fc2(h))))

        log_pi = F.log_softmax(self.fc_policy(h), dim=1)
        v = torch.tanh(self.fc_value(h))

        return log_pi, v


class OthelloNetWrapper:
    """Wraps OthelloNNet with the same interface as ConvNetWrapper/SimpleNetWrapper."""

    def __init__(self, board_size: int, action_size: int, config: NetworkConfig,
                 lr: float = 0.001, board_shape: tuple[int, int] = (6, 6)):
        self.net = OthelloNNet(board_shape, action_size, config)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.board_size = board_size
        self.board_shape = board_shape
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

        policy_loss = -torch.sum(target_pis_t * log_pi) / states_t.size(0)
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

    def clone(self) -> 'OthelloNetWrapper':
        new_wrapper = OthelloNetWrapper(
            self.board_size, self.action_size, self.config,
            self.lr, self.board_shape,
        )
        new_wrapper.net.load_state_dict(copy.deepcopy(self.net.state_dict()))
        return new_wrapper
