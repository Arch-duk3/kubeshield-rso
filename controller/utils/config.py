"""
KRSI Framework — Configuration Loader

Loads and validates YAML experiment configs, providing typed dataclasses
to the rest of the codebase. Eliminates all hardcoded constants.

WHY typed config dataclasses:
  - IDE autocomplete and type checking catches config typos at dev time
  - Validation at load time (not buried in training loops 1000 steps in)
  - Reproducibility: the full config is logged at experiment start
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class SimulatorConfig:
    host: str = "localhost"
    port: int = 8080
    seed: int = 42

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ExperimentConfig:
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    epochs: int = 100
    steps_per_epoch: int = 100
    output_dir: str = "datasets"
    log_level: str = "INFO"


@dataclass
class EnvironmentConfig:
    max_nodes: int = 10
    max_replicas: int = 20
    initial_nodes: int = 3
    initial_replicas: int = 3
    sla_latency_ms: float = 50.0
    max_energy_per_node: float = 200.0
    r_min: float = 0.6
    r_critical: float = 0.3
    heuristic_cooldown: int = 2


@dataclass
class AgentConfig:
    type: str = "ddqn"
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.1
    batch_size: int = 32
    replay_buffer_size: int = 10000
    target_update_freq: int = 5
    hidden_layers: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    grad_clip_norm: float = 1.0


@dataclass
class RewardConfig:
    lambda_penalty: float = 2.0
    gamma_action: float = 0.1
    eta_sla: float = 1.5
    tanh_scale: float = 5.0
    stability_bonus: float = 0.1


@dataclass
class KRSIConfig:
    window_size: int = 100
    softmin_k: float = 5.0
    sustainability_weights: str = "adaptive"
    resilience_weights: str = "adaptive"


@dataclass
class LoggingConfig:
    format: str = "json"
    level: str = "INFO"
    output: str = "stdout"
    log_dir: str = "logs"
    structured: bool = True


@dataclass
class KRSIFrameworkConfig:
    """Root configuration object for the entire KRSI framework."""

    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    krsi: KRSIConfig = field(default_factory=KRSIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict:
        """Serialise config for experiment logging."""
        return asdict(self)


def load_config(path: str | Path = "configs/default.yaml") -> KRSIFrameworkConfig:
    """Load and validate a YAML config file, returning a typed config object.

    Falls back to defaults for any missing keys, so partial configs work.

    Args:
        path: Path to the YAML config file.

    Returns:
        KRSIFrameworkConfig populated from the file.

    Raises:
        FileNotFoundError: if the config path does not exist.
        yaml.YAMLError: if the YAML is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    def _extract(cls, key: str) -> object:
        section = raw.get(key, {})
        # Only pass keys that the dataclass actually accepts
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in section.items() if k in valid_fields}
        return cls(**filtered)

    return KRSIFrameworkConfig(
        simulator=_extract(SimulatorConfig, "simulator"),
        experiment=_extract(ExperimentConfig, "experiment"),
        environment=_extract(EnvironmentConfig, "environment"),
        agent=_extract(AgentConfig, "agent"),
        reward=_extract(RewardConfig, "reward"),
        krsi=_extract(KRSIConfig, "krsi"),
        logging=_extract(LoggingConfig, "logging"),
    )


def load_config_from_env() -> KRSIFrameworkConfig:
    """Load config path from KRSI_CONFIG env var, defaulting to configs/default.yaml."""
    config_path = os.environ.get("KRSI_CONFIG", "configs/default.yaml")
    return load_config(config_path)
