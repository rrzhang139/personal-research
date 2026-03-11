"""Residual CNN for board games (AlphaZero-style).

Architecture:
    Board (flat) -> reshape (1, rows, cols)
        -> initial conv (3x3, num_filters) + BN + ReLU
        -> N residual blocks (two 3x3 convs with BN + skip)
        -> policy head: 1x1 conv -> BN -> ReLU -> flatten -> FC(action_size)
        -> value head:  1x1 conv -> BN -> ReLU -> flatten -> FC(64) -> ReLU -> FC(1) -> tanh

Captures spatial patterns (corners, edges, flanking) that an MLP cannot exploit.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import NetworkConfig


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel-wise attention (Leela Zero, KataGo)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        se = x.mean(dim=(2, 3))  # (B, C)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """Residual block: conv -> [BN] -> ReLU -> conv -> [BN] -> [SE] -> skip -> ReLU."""

    def __init__(self, num_filters: int, use_bn: bool = True, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(num_filters) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(num_filters) if use_bn else nn.Identity()
        self.se = SEBlock(num_filters) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        out = F.relu(out + residual)
        return out


class ConvNet(nn.Module):
    """Residual CNN with separate policy and value heads."""

    def __init__(self, board_shape: tuple, action_size: int, config: NetworkConfig):
        super().__init__()
        self.action_size = action_size
        self.config = config

        # Handle 2-tuple (rows, cols) or 3-tuple (channels, rows, cols)
        if len(board_shape) == 3:
            in_channels, rows, cols = board_shape
        else:
            rows, cols = board_shape
            in_channels = 1
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.board_shape = board_shape
        nf = config.num_filters
        use_bn = getattr(config, 'use_batch_norm', True)

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, nf, 3, padding=1, bias=not use_bn)
        self.initial_bn = nn.BatchNorm2d(nf) if use_bn else nn.Identity()

        # Residual tower
        use_se = getattr(config, 'use_se', False)
        self.res_blocks = nn.Sequential(
            *[ResBlock(nf, use_bn=use_bn, use_se=use_se) for _ in range(config.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(nf, 2, 1, bias=not use_bn)
        self.policy_bn = nn.BatchNorm2d(2) if use_bn else nn.Identity()
        self.policy_fc = nn.Linear(2 * rows * cols, action_size)

        # Value head
        self.value_conv = nn.Conv2d(nf, 1, 1, bias=not use_bn)
        self.value_bn = nn.BatchNorm2d(1) if use_bn else nn.Identity()
        self.global_pool_value = getattr(config, 'global_pool_value', False)
        value_input_size = rows * cols + (2 * nf if self.global_pool_value else 0)
        self.value_fc1 = nn.Linear(value_input_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Dropout (applied in policy and value heads)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Batch of board states, shape (B, board_size) — flat.

        Returns:
            (log_policy, value): log_policy shape (B, action_size), value shape (B, 1).
        """
        # Reshape flat input to (B, C, rows, cols)
        h = x.view(-1, self.in_channels, self.rows, self.cols)

        # Initial conv
        h = F.relu(self.initial_bn(self.initial_conv(h)))

        # Residual tower
        h = self.res_blocks(h)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)
        p = self.dropout(p)
        log_pi = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head (optional global pooling)
        v_spatial = F.relu(self.value_bn(self.value_conv(h)))
        v_spatial = v_spatial.view(v_spatial.size(0), -1)
        if self.global_pool_value:
            v_avg = h.mean(dim=(2, 3))  # (B, nf)
            v_max = h.amax(dim=(2, 3))  # (B, nf)
            v = torch.cat([v_spatial, v_avg, v_max], dim=1)
        else:
            v = v_spatial
        v = self.dropout(v)
        v = F.relu(self.value_fc1(v))
        v = self.dropout(v)
        v = torch.tanh(self.value_fc2(v))

        return log_pi, v


class ConvNetWrapper:
    """Wraps ConvNet with the same interface as SimpleNetWrapper."""

    def __init__(self, board_size: int, action_size: int, config: NetworkConfig,
                 lr: float = 0.001, board_shape: tuple[int, int] = (6, 6)):
        self.net = ConvNet(board_shape, action_size, config)
        self.board_size = board_size
        self.board_shape = board_shape
        self.action_size = action_size
        self.config = config
        self.lr = lr
        self.weight_decay = 0.0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=self.weight_decay)

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        if self.net.training:
            self.net.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.net.device)
            log_pi, v = self.net(x)
            pi = torch.exp(log_pi).squeeze(0).cpu().numpy()
            value = v.item()
        return pi, value

    def predict_batch(self, states: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        """Batch prediction for multiple states (used by virtual loss MCTS)."""
        if self.net.training:
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

    def clone(self) -> 'ConvNetWrapper':
        new_wrapper = ConvNetWrapper(
            self.board_size, self.action_size, self.config,
            self.lr, self.board_shape,
        )
        new_wrapper.net.load_state_dict(copy.deepcopy(self.net.state_dict()))
        if self.weight_decay > 0:
            new_wrapper.weight_decay = self.weight_decay
            new_wrapper.optimizer = torch.optim.Adam(
                new_wrapper.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        return new_wrapper
