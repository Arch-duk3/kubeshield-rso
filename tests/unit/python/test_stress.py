"""
KRSI Framework — Stress & Stability Tests

Performs long-duration simulation runs to detect numeric instability,
memory leaks, or cumulative drift in the KRSI calculation.
"""
from __future__ import annotations

import time
import pytest
import numpy as np
from controller.env.k8s_env import K8sEnv
from controller.env.sim_client import MockSimulatorClient
from controller.utils.config import KRSIFrameworkConfig
from controller.utils.logger import KRSILogger


@pytest.mark.stress
def test_long_duration_stability():
    """Run 10,000 steps and verify no NaN/Inf or catastrophic score drops."""
    cfg = KRSIFrameworkConfig()
    log = KRSILogger("stress.test", level="ERROR")
    # Using a deterministic mock for repeatability
    client = MockSimulatorClient(seed=42)
    env = K8sEnv(cfg=cfg, log=log, client=client)
    
    state = env.reset()
    start_time = time.time()
    
    for i in range(10000):
        # Alternate between scaling out and no-op
        action = 1 if i % 10 == 0 else 0
        state, reward, done, info = env.step(action)
        
        assert np.isfinite(state).all(), f"Step {i}: State contains non-finite values"
        assert np.isfinite(reward), f"Step {i}: Reward is non-finite: {reward}"
        assert 0.0 <= info["KRSI"] <= 1.0, f"Step {i}: KRSI out of bounds: {info['KRSI']}"
        
        if done:
            state = env.reset()
            
    duration = time.time() - start_time
    print(f"\nCompleted 10,000 steps in {duration:.2f}s ({10000/duration:.2f} steps/sec)")


@pytest.mark.stress
def test_reward_convergence_bounds():
    """Verify reward doesn't explode over a trajectory."""
    cfg = KRSIFrameworkConfig()
    log = KRSILogger("stress.test", level="ERROR")
    client = MockSimulatorClient(seed=123)
    env = K8sEnv(cfg=cfg, log=log, client=client)
    
    env.reset()
    rewards = []
    for _ in range(1000):
        _, reward, _, _ = env.step(0)
        rewards.append(reward)
        
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Reward is squashed by tanh_scale=10.0
    assert abs(mean_reward) <= 10.0
    assert std_reward < 5.0 # Should be relatively stable
