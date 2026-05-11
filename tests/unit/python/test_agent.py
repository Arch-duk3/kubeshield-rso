"""
KRSI Framework — Unit Tests for DDQN Agent

Tests the RL agent independently of the environment.
Covers: action selection modes, replay buffer, gradient updates,
epsilon decay, target network sync, and model save/load.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
import torch

from controller.agents.ddqn_agent import DDQNAgent, ReplayBuffer
from controller.utils.config import AgentConfig
from controller.utils.logger import KRSILogger


@pytest.fixture
def agent() -> DDQNAgent:
    cfg = AgentConfig()
    log = KRSILogger("test.agent", level="ERROR")
    return DDQNAgent(state_dim=12, action_dim=9, cfg=cfg, log=log, device="cpu")


@pytest.fixture
def filled_agent(agent: DDQNAgent) -> DDQNAgent:
    """Agent with a replay buffer full enough for updates."""
    rng = np.random.RandomState(42)
    for _ in range(100):
        s = rng.randn(12).astype(np.float32)
        a = rng.randint(0, 9)
        r = rng.randn()
        ns = rng.randn(12).astype(np.float32)
        d = rng.random() < 0.1
        agent.store(s, a, r, ns, d)
    return agent


# ─── Replay Buffer ───────────────────────────────────────────────────────────


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.push(np.zeros(12), 0, 1.0, np.zeros(12), False)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(np.zeros(12), 0, float(i), np.zeros(12), False)
        assert len(buf) == 10

    def test_sample_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(np.zeros(12), 0, float(i), np.zeros(12), False)
        batch = buf.sample(10)
        assert len(batch) == 10

    def test_sample_insufficient_raises(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(np.zeros(12), 0, 1.0, np.zeros(12), False)
        with pytest.raises(ValueError):
            buf.sample(10)


# ─── Action Selection ────────────────────────────────────────────────────────


class TestActionSelection:
    def test_action_in_valid_range(self, agent: DDQNAgent):
        state = np.random.randn(12).astype(np.float32)
        for _ in range(50):
            action, reason = agent.select_action(
                state, r_critical=0.3, R=0.5, heuristic_fn=lambda s, r: 1
            )
            assert 0 <= action <= 8

    def test_heuristic_override_below_critical(self, agent: DDQNAgent):
        state = np.random.randn(12).astype(np.float32)
        action, reason = agent.select_action(
            state,
            r_critical=0.3,
            R=0.1,  # R < r_critical
            heuristic_fn=lambda s, r: 7,
        )
        assert action == 7
        assert reason == "heuristic"

    def test_exploration_at_epsilon_1(self, agent: DDQNAgent):
        agent.epsilon = 1.0  # always explore
        state = np.random.randn(12).astype(np.float32)
        _, reason = agent.select_action(state, r_critical=0.3, R=0.5, heuristic_fn=lambda s, r: 0)
        assert reason == "exploration"

    def test_exploitation_at_epsilon_0(self, agent: DDQNAgent):
        agent.epsilon = 0.0  # always exploit
        state = np.random.randn(12).astype(np.float32)
        _, reason = agent.select_action(state, r_critical=0.3, R=0.5, heuristic_fn=lambda s, r: 0)
        assert reason == "exploitation"


# ─── Training Update ─────────────────────────────────────────────────────────


class TestTrainingUpdate:
    def test_update_returns_none_when_buffer_insufficient(self, agent: DDQNAgent):
        loss = agent.update()
        assert loss is None

    def test_update_returns_float_loss(self, filled_agent: DDQNAgent):
        loss = filled_agent.update()
        assert loss is not None
        assert isinstance(loss, float)
        assert math.isfinite(loss)

    def test_update_modifies_weights(self, filled_agent: DDQNAgent):
        params_before = [p.clone() for p in filled_agent.online_net.parameters()]
        filled_agent.update()
        params_after = list(filled_agent.online_net.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        assert changed, "Weights should change after an update"


# ─── Epsilon Decay ───────────────────────────────────────────────────────────


class TestEpsilonDecay:
    def test_decay_reduces_epsilon(self, agent: DDQNAgent):
        initial = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < initial

    def test_decay_respects_minimum(self, agent: DDQNAgent):
        for _ in range(1000):
            agent.decay_epsilon()
        assert agent.epsilon >= agent.cfg.epsilon_min


# ─── Target Network ─────────────────────────────────────────────────────────


class TestTargetNetwork:
    def test_sync_copies_weights(self, filled_agent: DDQNAgent):
        # Change online net via an update
        filled_agent.update()
        # Now sync
        filled_agent.sync_target_network()
        for op, tp in zip(
            filled_agent.online_net.parameters(), filled_agent.target_net.parameters()
        ):
            assert torch.equal(op, tp), "Target should match online after sync"


# ─── Save / Load ─────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, agent: DDQNAgent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            agent.save(path)
            assert os.path.exists(path)

            new_agent = DDQNAgent(
                state_dim=12,
                action_dim=9,
                cfg=AgentConfig(),
                log=KRSILogger("test", level="ERROR"),
                device="cpu",
            )
            new_agent.load(path)

            state = torch.randn(1, 12)
            with torch.no_grad():
                q1 = agent.online_net(state)
                q2 = new_agent.online_net(state)
            assert torch.allclose(q1, q2), "Loaded model should produce same Q-values"
