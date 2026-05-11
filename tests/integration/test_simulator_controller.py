"""
KRSI Framework — Integration Tests

Tests the full simulator ↔ controller communication pipeline.
REQUIRES the Go simulator running on localhost:8080.

Run with: make run-simulator & make test-integration
"""

from __future__ import annotations

import pytest
import requests

SIMULATOR_URL = "http://localhost:8080"


def simulator_is_running() -> bool:
    try:
        r = requests.get(f"{SIMULATOR_URL}/health", timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not simulator_is_running(), reason="Go simulator not running on localhost:8080"
)


class TestSimulatorAPI:
    """Test the raw HTTP API contract of the simulator."""

    def test_health_endpoint(self):
        r = requests.get(f"{SIMULATOR_URL}/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_reset_endpoint(self):
        r = requests.get(f"{SIMULATOR_URL}/reset")
        assert r.status_code == 200

    def test_step_returns_valid_json(self):
        requests.get(f"{SIMULATOR_URL}/reset")
        r = requests.post(f"{SIMULATOR_URL}/step", json={"action": 0})
        assert r.status_code == 200
        data = r.json()
        assert "sustainability" in data
        assert "resilience" in data
        assert "state" in data

    def test_step_all_actions(self):
        """Every action [0, 8] should return 200."""
        requests.get(f"{SIMULATOR_URL}/reset")
        for action in range(9):
            r = requests.post(f"{SIMULATOR_URL}/step", json={"action": action})
            assert r.status_code == 200, f"Action {action} failed with {r.status_code}"

    def test_step_invalid_action_rejected(self):
        r = requests.post(f"{SIMULATOR_URL}/step", json={"action": 99})
        assert r.status_code == 400

    def test_step_invalid_body_rejected(self):
        r = requests.post(
            f"{SIMULATOR_URL}/step", data="not json", headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 400

    def test_state_endpoint(self):
        requests.get(f"{SIMULATOR_URL}/reset")
        r = requests.get(f"{SIMULATOR_URL}/state")
        assert r.status_code == 200
        state = r.json()
        assert "NRep" in state or "TimeStep" in state


class TestSimulatorControllerPipeline:
    """Test the full data flow from action → metrics → KRSI computation."""

    def test_full_episode_pipeline(self):
        """Run a short episode and verify KRSI is computable from responses."""
        from controller.models.krsi_calculator import KRSICalculator
        from controller.utils.config import KRSIConfig

        requests.get(f"{SIMULATOR_URL}/reset")
        calc = KRSICalculator(KRSIConfig())

        for step in range(20):
            action = step % 9
            r = requests.post(f"{SIMULATOR_URL}/step", json={"action": action})
            assert r.status_code == 200
            data = r.json()

            krsi, s, res = calc.compute(data)
            assert 0.0 <= krsi <= 1.0, f"Step {step}: KRSI={krsi}"
            assert 0.0 <= s <= 1.0, f"Step {step}: S={s}"
            assert 0.0 <= res <= 1.0, f"Step {step}: R={res}"

    def test_reset_clears_state(self):
        """After reset, state should return to initial values."""
        requests.get(f"{SIMULATOR_URL}/reset")
        # Run some steps to change state
        for _ in range(10):
            requests.post(f"{SIMULATOR_URL}/step", json={"action": 1})

        # Reset
        requests.get(f"{SIMULATOR_URL}/reset")
        r = requests.post(f"{SIMULATOR_URL}/step", json={"action": 0})
        data = r.json()
        # After reset + 1 step, replicas should be near initial (3)
        assert data["state"]["n_rep"] <= 5, "State not properly reset"

    def test_metrics_pipeline_consistency(self):
        """Sustainability and resilience metrics should be internally consistent."""
        requests.get(f"{SIMULATOR_URL}/reset")
        r = requests.post(f"{SIMULATOR_URL}/step", json={"action": 0})
        data = r.json()

        sus = data["sustainability"]
        assert sus["e_renewable"] <= sus["e_total"], "Renewable energy cannot exceed total"
        assert 0.0 <= sus["u_cpu"] <= 1.0
        assert 0.0 <= sus["u_mem"] <= 1.0

        res = data["resilience"]
        assert res["t_up"] <= res["t_obs"], "Uptime cannot exceed observed time"
        assert res["n_affected"] <= res["n_total"], "Affected nodes cannot exceed total"


class TestAttackDefenseFlow:
    """Test adversarial event injection and detection metrics."""

    def test_adversarial_events_appear_over_time(self):
        """Over many steps, adversarial events should eventually occur."""
        requests.get(f"{SIMULATOR_URL}/reset")
        saw_adversarial = False
        for _ in range(200):
            r = requests.post(f"{SIMULATOR_URL}/step", json={"action": 0})
            data = r.json()
            if data["resilience"]["n_affected"] > 0:
                saw_adversarial = True
                break
        assert saw_adversarial, "Expected at least one adversarial event in 200 steps"

    def test_failure_recovery_occurs(self):
        """Failures should eventually recover."""
        requests.get(f"{SIMULATOR_URL}/reset")
        saw_failure = False
        saw_recovery = False
        prev_failures = 0

        for _ in range(200):
            r = requests.post(f"{SIMULATOR_URL}/step", json={"action": 0})
            data = r.json()
            cur_failures = data["resilience"]["n_f"]
            if cur_failures > 0:
                saw_failure = True
            if saw_failure and cur_failures < prev_failures:
                saw_recovery = True
                break
            prev_failures = cur_failures

        # It's stochastic, so we only assert if we saw a failure
        if saw_failure:
            assert saw_recovery, "Expected recovery after failure within 200 steps"
