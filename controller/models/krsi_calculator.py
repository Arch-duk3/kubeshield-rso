"""
KubeShield Framework — Generic Resilience Calculator (Public)
"""

from typing import Any


class KRSICalculator:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def compute(self, metrics: dict[str, Any]) -> tuple[float, float, float]:
        u_cpu = metrics["sustainability"]["u_cpu"]
        u_mem = metrics["sustainability"]["u_mem"]
        r_up = metrics["resilience"]["t_up"] / max(metrics["resilience"]["t_obs"], 1.0)
        S = 1.0 - (u_cpu + u_mem) / 2.0
        R = r_up
        KRSI = (R + S) / 2.0
        return float(KRSI), float(S), float(R)
