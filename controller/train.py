"""
KRSI Framework — Training Entrypoint

Orchestrates the full experiment lifecycle:
  1. Load config from YAML (fully reproducible)
  2. Set random seeds (deterministic)
  3. Connect to simulator (with retry)
  4. Run training loop per seed
  5. Export results to CSV + JSON

Usage:
  python -m controller.train --config configs/default.yaml --seed 42 --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controller.agents.ddqn_agent import DDQNAgent
from controller.env.k8s_env import K8sEnv
from controller.env.sim_client import SimulatorClient
from controller.utils.config import KRSIFrameworkConfig, load_config
from controller.utils.logger import EventType, KRSILogger


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(seed: int, cfg: KRSIFrameworkConfig, log: KRSILogger) -> dict[str, Any]:
    """Run a single training experiment for the given seed.

    Returns a summary dict with final metrics for statistical analysis.
    """
    set_seeds(seed)
    log.set_context(seed=seed)
    log.info(EventType.SEED_SET, f"Seed set: {seed}", data={"seed": seed})

    # Connect to simulator with retry
    client = SimulatorClient(cfg.simulator.url)
    log.info(
        EventType.SIM_CONNECT_RETRY, "Waiting for simulator...", data={"url": cfg.simulator.url}
    )
    if not client.wait_for_simulator(max_retries=30, delay=1.0):
        log.error(
            EventType.SIM_STEP_ERROR,
            "Simulator unreachable after 30 retries",
            data={"url": cfg.simulator.url},
        )
        sys.exit(1)
    log.info(EventType.SIM_CONNECTED, "Connected to simulator", data={"url": cfg.simulator.url})

    # Build environment and agent
    env = K8sEnv(cfg=cfg, log=log, client=client)
    agent = DDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space,
        cfg=cfg.agent,
        log=log,
    )

    # Prepare output paths
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"experiment_seed_{seed}.csv"
    model_path = output_dir / f"model_seed_{seed}.pt"

    results = []
    all_krsi = []
    all_rewards = []
    global_step = 0

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "global_step",
                "epoch",
                "step",
                "seed",
                "S",
                "R",
                "KRSI",
                "action",
                "action_name",
                "n_rep",
                "n_nodes",
                "latency_ms",
                "energy_w",
                "reward",
                "epsilon",
            ]
        )

        for epoch in range(cfg.experiment.epochs):
            log.set_context(epoch=epoch)
            log.info(EventType.EPOCH_START, f"Epoch {epoch} started", data={"epoch": epoch})

            state = env.reset()
            epoch_reward = 0.0
            epoch_krsi = 0.0
            losses = []

            for step in range(cfg.experiment.steps_per_epoch):
                r_val = float(state[4])  # R is index 4 in state vector

                action, reason = agent.select_action(
                    state=state,
                    r_critical=env.R_critical,
                    R=r_val,
                    heuristic_fn=env.heuristic_action,
                )

                next_state, reward, done, info = env.step(action)
                agent.store(state, action, reward, next_state, done)

                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                epoch_reward += reward
                epoch_krsi += info["KRSI"]
                global_step += 1
                all_krsi.append(info["KRSI"])
                all_rewards.append(reward)

                writer.writerow(
                    [
                        global_step,
                        epoch,
                        step,
                        seed,
                        round(info["S"], 6),
                        round(info["R"], 6),
                        round(info["KRSI"], 6),
                        info["action"],
                        info["action_name"],
                        info["n_rep"],
                        info["n_nodes"],
                        round(info["latency"], 3),
                        round(info["energy"], 3),
                        round(reward, 6),
                        round(agent.epsilon, 4),
                    ]
                )

                if done:
                    break

            agent.decay_epsilon()
            avg_krsi = epoch_krsi / cfg.experiment.steps_per_epoch
            avg_loss = float(np.mean(losses)) if losses else 0.0

            epoch_summary = {
                "seed": seed,
                "epoch": epoch,
                "avg_krsi": round(avg_krsi, 6),
                "epoch_reward": round(epoch_reward, 4),
                "avg_loss": round(avg_loss, 6),
                "epsilon": round(agent.epsilon, 4),
            }
            results.append(epoch_summary)

            if epoch % cfg.agent.target_update_freq == 0:
                agent.sync_target_network()

            log.info(EventType.EPOCH_END, f"Epoch {epoch} complete", data=epoch_summary)

    # Save model checkpoint
    agent.save(model_path)

    # Export epoch-level JSON results
    json_path = output_dir / f"results_seed_{seed}.json"
    with open(json_path, "w") as jf:
        json.dump({"config": cfg.to_dict(), "results": results}, jf, indent=2)

    summary = {
        "seed": seed,
        "mean_krsi": float(np.mean(all_krsi)),
        "std_krsi": float(np.std(all_krsi)),
        "mean_reward": float(np.mean(all_rewards)),
        "final_epsilon": agent.epsilon,
        "total_steps": global_step,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "model_path": str(model_path),
    }
    log.info(EventType.TRAIN_END, "Experiment complete", data=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="KRSI RL Controller Training")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (runs single seed)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.experiment.seeds = [args.seed]
    if args.epochs is not None:
        cfg.experiment.epochs = args.epochs

    log = KRSILogger(
        component="controller.train",
        level=cfg.experiment.log_level,
        output=cfg.logging.output,
        log_dir=cfg.logging.log_dir,
    )

    log.info(EventType.EXPERIMENT_CONFIG, "Experiment config loaded", data=cfg.to_dict())

    all_summaries = []
    for seed in cfg.experiment.seeds:
        summary = run_experiment(seed=seed, cfg=cfg, log=log)
        all_summaries.append(summary)

    # Print final multi-seed summary
    mean_krsis = [s["mean_krsi"] for s in all_summaries]
    log.info(
        EventType.TRAIN_END,
        "All seeds complete",
        data={
            "seeds": cfg.experiment.seeds,
            "mean_krsi_across_seeds": round(float(np.mean(mean_krsis)), 6),
            "std_krsi_across_seeds": round(float(np.std(mean_krsis)), 6),
        },
    )


if __name__ == "__main__":
    main()
