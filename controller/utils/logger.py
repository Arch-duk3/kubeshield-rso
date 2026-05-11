"""
KRSI Framework — Structured Logger (Python)

Provides JSON-structured, semantically-typed logging for the controller,
RL agent, environment, and experiment runner.

WHY semantic logging:
  Raw print() statements and unstructured log lines are useless for:
  - post-hoc RL debugging (why did the agent choose action X at step Y?)
  - experiment reproducibility (what config produced this KRSI trajectory?)
  - LLM-assisted log analysis
  - Prometheus/Grafana log-based alerting

  Every log record carries: timestamp, level, component, event_type,
  simulation context (cycle_id, step), and a typed data payload.
  This makes logs machine-parseable without regex brittle-ness.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class EventType(StrEnum):
    """Canonical event taxonomy for the controller/RL layer."""

    # Training lifecycle
    TRAIN_START = "TRAIN_START"
    TRAIN_END = "TRAIN_END"
    EPOCH_START = "EPOCH_START"
    EPOCH_END = "EPOCH_END"
    EPISODE_START = "EPISODE_START"
    EPISODE_END = "EPISODE_END"

    # RL decisions
    RL_ACTION_SELECTED = "RL_ACTION_SELECTED"
    HEURISTIC_ACTION_SELECTED = "HEURISTIC_ACTION_SELECTED"
    EXPLORATION_ACTION = "EXPLORATION_ACTION"
    EXPLOITATION_ACTION = "EXPLOITATION_ACTION"
    EPSILON_DECAY = "EPSILON_DECAY"

    # KRSI metrics
    KRSI_COMPUTED = "KRSI_COMPUTED"
    RESILIENCE_TRANSITION = "RESILIENCE_TRANSITION"
    RESILIENCE_CRITICAL = "RESILIENCE_CRITICAL"
    SUSTAINABILITY_UPDATE = "SUSTAINABILITY_UPDATE"

    # Reward
    REWARD_COMPUTED = "REWARD_COMPUTED"
    SLA_VIOLATION = "SLA_VIOLATION"
    RESILIENCE_PENALTY = "RESILIENCE_PENALTY"

    # Training updates
    LOSS_COMPUTED = "LOSS_COMPUTED"
    GRADIENT_CLIP = "GRADIENT_CLIP"
    TARGET_NET_UPDATE = "TARGET_NET_UPDATE"
    REPLAY_BUFFER_SAMPLE = "REPLAY_BUFFER_SAMPLE"

    # Simulator communication
    SIM_CONNECT_RETRY = "SIM_CONNECT_RETRY"
    SIM_CONNECTED = "SIM_CONNECTED"
    SIM_STEP_ERROR = "SIM_STEP_ERROR"

    # Experiment
    EXPERIMENT_CONFIG = "EXPERIMENT_CONFIG"
    SEED_SET = "SEED_SET"
    RESULTS_SAVED = "RESULTS_SAVED"

    # Anomaly / security
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    RESOURCE_PRESSURE = "RESOURCE_PRESSURE"


@dataclass
class LogRecord:
    """The canonical structured log record schema.

    Every field is typed and documented so that log consumers (dashboards,
    LLMs, alerting rules) can operate on schema-validated data.
    """

    timestamp: str
    level: str
    component: str
    event_type: str
    message: str
    experiment_id: str | None = None
    sim_cycle_id: int | None = None
    step: int | None = None
    epoch: int | None = None
    seed: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


class KRSILogger:
    """Structured JSON logger for the KRSI controller stack.

    Usage:
        log = KRSILogger("controller.agent", level="INFO")
        log.info(EventType.RL_ACTION_SELECTED, "Action chosen by policy",
                 step=42, data={"action": 1, "q_values": [0.1, 0.9, ...]})
    """

    ACTION_NAMES = {
        0: "NO_OP",
        1: "SCALE_OUT_REPLICA",
        2: "SCALE_IN_REPLICA",
        3: "ADD_NODE",
        4: "REMOVE_NODE",
        5: "SET_BINPACK",
        6: "SET_SPREAD",
        7: "SCALE_THRESHOLD_UP",
        8: "SCALE_THRESHOLD_DOWN",
    }

    def __init__(
        self,
        component: str,
        level: str = "INFO",
        output: str = "stdout",
        log_dir: str | None = None,
        experiment_id: str | None = None,
    ):
        self.component = component
        self.experiment_id = experiment_id or str(uuid.uuid4())[:8]
        self._level = getattr(logging, level.upper(), logging.INFO)

        # Context state — set via set_context()
        self._sim_cycle_id: int | None = None
        self._seed: int | None = None
        self._epoch: int | None = None

        # Configure handlers
        self._handlers: list[Any] = []
        if output in ("stdout", "both"):
            self._handlers.append(sys.stdout)
        if output in ("file", "both") and log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(log_dir) / f"{component.replace('.', '_')}_{self.experiment_id}.jsonl"
            self._handlers.append(open(log_path, "a", encoding="utf-8"))  # noqa: WPS515

    def set_context(
        self,
        sim_cycle_id: int | None = None,
        seed: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Set persistent context fields that are attached to every subsequent log record."""
        if sim_cycle_id is not None:
            self._sim_cycle_id = sim_cycle_id
        if seed is not None:
            self._seed = seed
        if epoch is not None:
            self._epoch = epoch

    def _emit(
        self,
        level: str,
        event_type: EventType | str,
        message: str,
        step: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        lvl_num = getattr(logging, level.upper(), logging.INFO)
        if lvl_num < self._level:
            return

        record = LogRecord(
            timestamp=datetime.now(UTC).isoformat(),
            level=level.upper(),
            component=self.component,
            event_type=str(event_type),
            message=message,
            experiment_id=self.experiment_id,
            sim_cycle_id=self._sim_cycle_id,
            step=step,
            epoch=self._epoch,
            seed=self._seed,
            data=data or {},
        )
        line = json.dumps(asdict(record), default=str)
        for handler in self._handlers:
            print(line, file=handler, flush=True)

    def debug(
        self,
        event_type: EventType | str,
        message: str,
        step: int | None = None,
        data: dict | None = None,
    ) -> None:
        self._emit("DEBUG", event_type, message, step, data)

    def info(
        self,
        event_type: EventType | str,
        message: str,
        step: int | None = None,
        data: dict | None = None,
    ) -> None:
        self._emit("INFO", event_type, message, step, data)

    def warning(
        self,
        event_type: EventType | str,
        message: str,
        step: int | None = None,
        data: dict | None = None,
    ) -> None:
        self._emit("WARNING", event_type, message, step, data)

    def error(
        self,
        event_type: EventType | str,
        message: str,
        step: int | None = None,
        data: dict | None = None,
    ) -> None:
        self._emit("ERROR", event_type, message, step, data)
