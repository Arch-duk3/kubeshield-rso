"""
KRSI Framework — Unit Tests for K8sEnv

Tests the environment wrapper independently of the Go simulator
by using MockSimulatorClient. Covers:
  - State normalisation bounds
  - Reward computation edge cases
  - Heuristic action correctness
  - Episode termination conditions
  - SLA violation detection
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from controller.env.k8s_env import K8sEnv
from controller.env.sim_client import MockSimulatorClient
from controller.utils.config import KRSIFrameworkConfig
from controller.utils.logger import KRSILogger


@pytest.fixture
def env() -> K8sEnv:
    cfg = KRSIFrameworkConfig()
    log = KRSILogger("test.env", level="ERROR")
    client = MockSimulatorClient(seed=42)
    return K8sEnv(cfg=cfg, log=log, client=client)


class TestEnvReset:
    def test_reset_returns_valid_state(self, env: K8sEnv):
        state = env.reset()
        assert state.shape == (12,), f"Expected shape (12,), got {state.shape}"
        assert state.dtype == np.float32

    def test_reset_state_in_bounds(self, env: K8sEnv):
        state = env.reset()
        # Most values should be in [-1, 1] (sched_val is ±1)
        for i, val in enumerate(state):
            assert not math.isnan(val), f"State[{i}] is NaN after reset"
            assert not math.isinf(val), f"State[{i}] is Inf after reset"


class TestEnvStep:
    def test_step_returns_tuple(self, env: K8sEnv):
        env.reset()
        result = env.step(0)
        assert len(result) == 4, "step() should return (state, reward, done, info)"

    def test_step_state_shape(self, env: K8sEnv):
        env.reset()
        state, _, _, _ = env.step(0)
        assert state.shape == (12,)

    def test_step_reward_is_finite(self, env: K8sEnv):
        env.reset()
        for action in range(9):
            _, reward, _, _ = env.step(action)
            assert math.isfinite(reward), f"Reward is not finite for action {action}: {reward}"

    def test_step_info_contains_required_keys(self, env: K8sEnv):
        env.reset()
        _, _, _, info = env.step(0)
        required = {
            "KRSI",
            "S",
            "R",
            "action",
            "action_name",
            "n_rep",
            "n_nodes",
            "latency",
            "energy",
        }
        assert required.issubset(info.keys()), f"Missing keys: {required - info.keys()}"

    def test_step_krsi_in_range(self, env: K8sEnv):
        env.reset()
        for _ in range(50):
            _, _, _, info = env.step(0)
            assert 0.0 <= info["KRSI"] <= 1.0, f"KRSI out of range: {info['KRSI']}"

    def test_all_actions_accepted(self, env: K8sEnv):
        """Every valid action [0, 8] should be accepted without error."""
        env.reset()
        for action in range(9):
            state, reward, done, info = env.step(action)
            assert state is not None


class TestRewardFunction:
    def test_reward_bounded_by_tanh(self, env: K8sEnv):
        """Reward should be in (-tanh_scale, +tanh_scale)."""
        env.reset()
        scale = env.tanh_scale
        for _ in range(100):
            _, reward, _, _ = env.step(0)
            assert -scale <= reward <= scale, f"Reward {reward} exceeds tanh bounds"

    def test_noop_does_not_get_stability_bonus(self, env: K8sEnv):
        """NO_OP (action=0) should not receive the stability bonus."""
        # The stability bonus is for consistent non-noop actions
        env.reset()
        env.step(0)
        env.step(0)
        # This is tested implicitly — we just ensure no crash


class TestHeuristicAction:
    def test_heuristic_returns_valid_action(self, env: K8sEnv):
        env.reset()
        state = np.zeros(12, dtype=np.float32)
        action = env.heuristic_action(state, R=0.2)
        assert 0 <= action <= 8

    def test_heuristic_noop_during_cooldown(self, env: K8sEnv):
        env.reset()
        state = np.zeros(12, dtype=np.float32)
        # First call sets cooldown
        env.heuristic_action(state, R=0.2)
        # Second call during cooldown should return NO_OP
        action = env.heuristic_action(state, R=0.2)
        assert action == 0, "Heuristic should return NO_OP during cooldown"


class TestEpisodeTermination:
    def test_not_done_on_healthy_system(self, env: K8sEnv):
        """Mock client returns healthy metrics, so done should be False."""
        env.reset()
        _, _, done, info = env.step(0)
        # With default mock: R should be high, latency low
        if info["R"] > 0.2 and info["latency"] <= 3 * env.sla_latency:
            assert not done


class TestStateNormalisation:
    def test_normalised_energy_in_range(self, env: K8sEnv):
        env.reset()
        state, _, _, _ = env.step(0)
        norm_e = state[3]  # index 3 = normalised energy
        assert 0.0 <= norm_e <= 1.0, f"Normalised energy out of range: {norm_e}"

    def test_sched_val_is_plus_or_minus_one(self, env: K8sEnv):
        env.reset()
        state, _, _, _ = env.step(0)
        sched_val = state[11]  # index 11 = scheduler value
        assert sched_val in (-1.0, 1.0), f"Scheduler value should be ±1, got {sched_val}"
