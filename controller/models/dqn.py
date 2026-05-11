"""
KRSI Framework — DQN Model Definition

Separating the PyTorch model from the training loop is essential for:
  - Independent model testing (architecture changes don't require a full train run)
  - Model serialisation/loading without coupling to agent state
  - Future architecture swaps (e.g., dueling DQN, transformer-based policy)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(
    input_dim: int, hidden_layers: list[int], output_dim: int, activation: str
) -> nn.Sequential:
    """Build an MLP from a list of hidden layer sizes.

    WHY a factory function instead of hardcoded layers:
      The architecture is now fully configurable from the YAML config,
      enabling hyperparameter sweeps without code changes.
    """
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    act_cls = act_map.get(activation.lower(), nn.ReLU)

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class DQN(nn.Module):
    """Deep Q-Network policy approximator.

    Input:  state vector of shape (state_dim,)
    Output: Q-value vector of shape (action_dim,)

    Architecture is fully configurable via hidden_layers and activation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = _build_mlp(
            input_dim=state_dim,
            hidden_layers=hidden_layers or [64, 64],
            output_dim=action_dim,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, state: torch.Tensor) -> int:
        """Greedy action selection (no gradient)."""
        with torch.no_grad():
            return int(torch.argmax(self.forward(state)).item())

    @classmethod
    def from_config(cls, state_dim: int, action_dim: int, agent_cfg: object) -> DQN:
        """Construct a DQN from an AgentConfig object."""
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=agent_cfg.hidden_layers,
            activation=agent_cfg.activation,
        )
