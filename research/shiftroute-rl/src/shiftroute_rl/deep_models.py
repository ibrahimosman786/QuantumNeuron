"""Small trainable neural models. Pure PyTorch, CPU-compatible."""
from __future__ import annotations
import torch
from torch import nn
from .deep_env import OBS_DIM

class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 6)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.core(x)
        return self.actor(h), self.critic(h).squeeze(-1)

class RiskMonitor(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)
