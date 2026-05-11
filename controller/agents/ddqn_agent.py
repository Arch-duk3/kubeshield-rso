"""
KRSI Framework — Double DQN Agent

Implements the full Double DQN training loop with:
  - Experience replay (prioritisable buffer)
  - Double DQN target computation (reduces overestimation bias)
  - Gradient clipping (prevents training instability)
  - Batch reward normalisation (stabilises Q-value scale)
  - Structured logging at every training event
  - Heuristic safety fallback (critical resilience floor)
"""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from controller.models.dqn import DQN
from controller.utils.config import AgentConfig
from controller.utils.logger import KRSILogger, EventType


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-size circular replay buffer for experience replay.

    WHY deque over list: O(1) append/pop from both ends, automatic
    eviction of oldest experiences when capacity is reached.
    """

    def __init__(self, capacity: int) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


class DDQNAgent:
    """Double DQN agent with heuristic safety override.

    Architecture:
      - Online network: updated every step via SGD
      - Target network: updated every N epochs (hard copy)
      - Heuristic fallback: activated when R < r_critical

    WHY Double DQN (§3 in van Hasselt et al. 2016):
      Standard DQN overestimates Q-values because argmax and evaluation
      use the same (noisy) target network. Double DQN decouples these:
        a* = argmax_a Q_online(s', a)   ← online selects best action
        y  = r + γ * Q_target(s', a*)   ← target evaluates that action
      This reduces overestimation bias and improves policy stability.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: AgentConfig,
        log: KRSILogger,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.log = log
        self.action_dim = action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = DQN.from_config(state_dim, action_dim, cfg).to(self.device)
        self.target_net = DQN.from_config(state_dim, action_dim, cfg).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(cfg.replay_buffer_size)

        self.epsilon = cfg.epsilon_start
        self.gamma = cfg.gamma
        self._step_count = 0

        log.info(EventType.TRAIN_START, "DDQN Agent initialised", data={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "device": str(self.device),
            "architecture": cfg.hidden_layers,
            "lr": cfg.learning_rate,
            "buffer_size": cfg.replay_buffer_size,
        })

    def select_action(self, state: np.ndarray, r_critical: float,
                      R: float, heuristic_fn: Any) -> Tuple[int, str]:
        """Select action using ε-greedy policy with heuristic safety override.

        Returns:
            (action, reason) where reason is one of:
            'heuristic', 'exploration', 'exploitation'
        """
        # Safety override: when resilience is critically low, heuristic takes control
        if R < r_critical:
            action = heuristic_fn(state, R)
            return action, "heuristic"

        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            self.log.debug(EventType.EXPLORATION_ACTION, "Exploration action",
                           data={"action": action, "epsilon": round(self.epsilon, 4)})
            return action, "exploration"

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.online_net.get_action(state_t)
        self.log.debug(EventType.EXPLOITATION_ACTION, "Exploitation action",
                       data={"action": action, "epsilon": round(self.epsilon, 4)})
        return action, "exploitation"

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """Sample a batch and perform one gradient update.

        Returns:
            Loss value (float) or None if buffer is insufficient.
        """
        if len(self.buffer) < self.cfg.batch_size:
            return None

        batch = self.buffer.sample(self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Batch reward normalisation: prevents Q-value scale from drifting
        # across episodes with different reward magnitudes
        rewards_mean = rewards_t.mean()
        rewards_std = rewards_t.std() + 1e-5
        rewards_t = (rewards_t - rewards_mean) / rewards_std

        # Current Q values
        q_vals = self.online_net(states_t).gather(1, actions_t)

        # Double DQN target computation
        with torch.no_grad():
            a_star = self.online_net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, a_star)
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.MSELoss()(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: prevents destabilising updates from outlier batches
        grad_norm = nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.cfg.grad_clip_norm
        )

        self.optimizer.step()
        self._step_count += 1

        loss_val = float(loss.item())
        self.log.debug(EventType.LOSS_COMPUTED, "Training update", data={
            "loss": round(loss_val, 6),
            "grad_norm": round(float(grad_norm), 4),
            "buffer_size": len(self.buffer),
        })

        return loss_val

    def decay_epsilon(self) -> None:
        """Decay ε and log the transition."""
        prev = self.epsilon
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
        self.log.debug(EventType.EPSILON_DECAY, "Epsilon decayed",
                       data={"from": round(prev, 4), "to": round(self.epsilon, 4)})

    def sync_target_network(self) -> None:
        """Hard copy online weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.log.info(EventType.TARGET_NET_UPDATE, "Target network synchronised",
                      data={"step": self._step_count})

    def save(self, path: str | Path) -> None:
        """Save online network weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), path)
        self.log.info(EventType.RESULTS_SAVED, f"Model saved to {path}", data={"path": str(path)})

    def load(self, path: str | Path) -> None:
        """Load online network weights (for inference or fine-tuning)."""
        state_dict = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
