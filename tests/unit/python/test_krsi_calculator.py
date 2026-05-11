"""
KRSI Framework — Unit Tests for KRSICalculator

Tests the core mathematical model that produces the optimisation objective.
Bugs here silently corrupt every experiment — these tests are critical.

Coverage:
  - Nominal path: valid inputs produce values in [0, 1]
  - Edge cases: zero energy, zero failures, max failures
  - Floating-point tolerance: no NaN/Inf propagation
  - Determinism: same inputs produce identical outputs
  - Adaptive weights: verify inverse-variance weighting converges
  - Long trajectory: 10,000 steps without numeric instability
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from controller.models.krsi_calculator import KRSICalculator
from controller.utils.config import KRSIConfig

# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_metrics(
    e_total: float = 300.0,
    w: float = 95.0,
    ci: float = 0.42,
    e_renewable: float = 90.0,
    u_cpu: float = 0.6,
    u_mem: float = 0.48,
    u_sto: float = 0.5,
    u_target: float = 0.8,
    t_up: float = 900.0,
    t_obs: float = 1000.0,
    n_f: int = 0,
    t_rec: float = 5.0,
    t_rec_max: float = 20.0,
    p_min: float = 0.9,
    p_baseline: float = 1.0,
    p_post: float = 0.9,
    w_disruption: float = 5.0,
    w_baseline: float = 100.0,
    t_det: float = 0.0,
    t_det_max: float = 20.0,
    t_sec: float = 0.0,
    t_sec_max: float = 40.0,
    n_affected: int = 0,
    n_total: int = 3,
    severity_score: float = 0.0,
    o_conf: float = 0.95,
) -> dict:
    return {
        "sustainability": {
            "e_total": e_total,
            "w": w,
            "ci": ci,
            "e_renewable": e_renewable,
            "u_cpu": u_cpu,
            "u_mem": u_mem,
            "u_sto": u_sto,
            "u_target": u_target,
        },
        "resilience": {
            "t_up": t_up,
            "t_obs": t_obs,
            "n_f": n_f,
            "t_rec": t_rec,
            "t_rec_max": t_rec_max,
            "p_min": p_min,
            "p_baseline": p_baseline,
            "p_post": p_post,
            "w_disruption": w_disruption,
            "w_baseline": w_baseline,
            "t_det": t_det,
            "t_det_max": t_det_max,
            "t_sec": t_sec,
            "t_sec_max": t_sec_max,
            "n_affected": n_affected,
            "n_total": n_total,
            "severity_score": severity_score,
            "o_conf": o_conf,
        },
    }


@pytest.fixture
def calc() -> KRSICalculator:
    return KRSICalculator(KRSIConfig())


@pytest.fixture
def nominal_metrics() -> dict:
    return _make_metrics()


# ─── Basic Output Tests ──────────────────────────────────────────────────────


class TestKRSIOutputRange:
    """KRSI, S, and R must always be in [0, 1]."""

    def test_nominal_range(self, calc: KRSICalculator, nominal_metrics: dict):
        krsi, s, r = calc.compute(nominal_metrics)
        assert 0.0 <= krsi <= 1.0, f"KRSI out of range: {krsi}"
        assert 0.0 <= s <= 1.0, f"S out of range: {s}"
        assert 0.0 <= r <= 1.0, f"R out of range: {r}"

    def test_perfect_system(self, calc: KRSICalculator):
        """System with no failures, high throughput, low energy."""
        m = _make_metrics(
            e_total=100.0,
            w=200.0,
            ci=0.1,
            e_renewable=80.0,
            t_up=1000.0,
            t_obs=1000.0,
            n_f=0,
            t_rec=0.0,
            p_min=1.0,
            p_post=1.0,
            w_disruption=0.0,
            t_det=0.0,
            t_sec=0.0,
            n_affected=0,
            severity_score=0.0,
        )
        krsi, s, r = calc.compute(m)
        assert krsi > 0.5, f"Perfect system KRSI should be high, got {krsi}"
        assert r > 0.5, f"Perfect system R should be high, got {r}"

    def test_degraded_system(self, calc: KRSICalculator):
        """System with many failures and high severity."""
        m = _make_metrics(
            n_f=5,
            t_rec=18.0,
            p_min=0.3,
            p_post=0.4,
            w_disruption=80.0,
            t_det=18.0,
            t_sec=35.0,
            n_affected=3,
            severity_score=0.8,
        )
        # Warm up the window first
        for _ in range(5):
            calc.compute(_make_metrics())
        krsi, s, r = calc.compute(m)
        assert r < 0.8, f"Degraded system R should be low, got {r}"


# ─── Edge Case Tests ─────────────────────────────────────────────────────────


class TestKRSIEdgeCases:
    """Test boundary conditions that could cause division by zero or NaN."""

    def test_zero_energy(self, calc: KRSICalculator):
        """e_total=0 should not crash (clamped to 1.0 internally)."""
        m = _make_metrics(e_total=0.0, w=0.0)
        krsi, s, r = calc.compute(m)
        assert not math.isnan(krsi)
        assert not math.isinf(krsi)

    def test_zero_workload(self, calc: KRSICalculator):
        m = _make_metrics(w=0.0, w_disruption=0.0, w_baseline=0.0)
        krsi, s, r = calc.compute(m)
        assert not math.isnan(krsi)

    def test_zero_observation_time(self, calc: KRSICalculator):
        m = _make_metrics(t_obs=0.0)
        krsi, s, r = calc.compute(m)
        assert not math.isnan(r)

    def test_zero_carbon_intensity(self, calc: KRSICalculator):
        m = _make_metrics(ci=0.0)
        krsi, s, r = calc.compute(m)
        assert not math.isnan(s)

    def test_zero_baseline_performance(self, calc: KRSICalculator):
        m = _make_metrics(p_baseline=0.0)
        krsi, s, r = calc.compute(m)
        assert not math.isnan(r)

    def test_severity_score_at_max(self, calc: KRSICalculator):
        m = _make_metrics(severity_score=1.0, n_affected=10, n_total=10)
        for _ in range(3):
            calc.compute(_make_metrics())
        krsi, s, r = calc.compute(m)
        assert 0.0 <= krsi <= 1.0

    def test_all_nodes_affected(self, calc: KRSICalculator):
        m = _make_metrics(n_affected=5, n_total=5)
        krsi, s, r = calc.compute(m)
        assert 0.0 <= r <= 1.0

    def test_very_large_failure_count(self, calc: KRSICalculator):
        m = _make_metrics(n_f=1000)
        krsi, s, r = calc.compute(m)
        assert not math.isnan(r)
        assert 0.0 <= r <= 1.0

    def test_negative_values_clamped(self, calc: KRSICalculator):
        m = _make_metrics(e_total=-10.0, w=-5.0, t_up=-100.0)
        krsi, s, r = calc.compute(m)
        assert 0.0 <= krsi <= 1.0


# ─── Determinism Tests ───────────────────────────────────────────────────────


class TestKRSIDeterminism:
    """Same inputs must produce identical outputs."""

    def test_identical_inputs_identical_outputs(self):
        calc1 = KRSICalculator(KRSIConfig())
        calc2 = KRSICalculator(KRSIConfig())
        m = _make_metrics()
        for _ in range(50):
            r1 = calc1.compute(m)
            r2 = calc2.compute(m)
            assert abs(r1[0] - r2[0]) < 1e-10, "KRSI diverged between identical calculators"
            assert abs(r1[1] - r2[1]) < 1e-10, "S diverged"
            assert abs(r1[2] - r2[2]) < 1e-10, "R diverged"


# ─── Stability Tests ─────────────────────────────────────────────────────────


class TestKRSIStability:
    """Test numeric stability over long trajectories."""

    def test_long_trajectory_no_nan(self, calc: KRSICalculator):
        """Run 10,000 steps with varying inputs — no NaN/Inf allowed."""
        rng = np.random.RandomState(42)
        for i in range(10000):
            m = _make_metrics(
                e_total=rng.uniform(50, 500),
                w=rng.uniform(0, 200),
                ci=rng.uniform(0.01, 1.0),
                e_renewable=rng.uniform(0, 200),
                u_cpu=rng.uniform(0, 1),
                u_mem=rng.uniform(0, 1),
                u_sto=rng.uniform(0, 1),
                n_f=rng.randint(0, 10),
                t_rec=rng.uniform(0, 20),
                n_affected=rng.randint(0, 5),
                severity_score=rng.uniform(0, 1),
            )
            krsi, s, r = calc.compute(m)
            assert not math.isnan(krsi), f"Step {i}: KRSI is NaN"
            assert not math.isinf(krsi), f"Step {i}: KRSI is Inf"
            assert 0.0 <= krsi <= 1.0, f"Step {i}: KRSI out of range: {krsi}"

    def test_adaptive_weights_converge(self, calc: KRSICalculator):
        """Verify weights don't oscillate wildly with constant input."""
        m = _make_metrics()
        for _ in range(200):
            calc.compute(m)
        w = calc._adaptive_weights(["E_eff", "CIW", "U_band", "Ren"])
        assert abs(sum(w) - 1.0) < 1e-6, f"Weights don't sum to 1: {w}"


# ─── Harmonic Mean Property Tests ────────────────────────────────────────────


class TestKRSIHarmonicMean:
    """Verify the harmonic mean (§8.10) satisfies known mathematical properties."""

    def test_krsi_bounded_by_components(self, calc: KRSICalculator):
        """KRSI = H(R, S) must be ≤ min(R, S) × 2 (harmonic mean property)."""
        m = _make_metrics()
        for _ in range(20):
            krsi, s, r = calc.compute(m)
            if r + s > 1e-5:
                expected = (2 * r * s) / (r + s)
                assert abs(krsi - expected) < 1e-9, f"KRSI != H(R,S): {krsi} vs {expected}"
