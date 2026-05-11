"""
KRSI Framework — Automated Experiment Runner

Orchestrates a full suite of experiments across multiple configurations
and seeds, aggregating results for research papers and benchmarks.

Features:
  - Parallel/Sequential execution of config variants
  - Automatic result aggregation into summary tables
  - Generation of statistical comparison plots (placeholder)
  - Config-coupled artifact preservation
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def run_command(cmd: List[str]) -> bool:
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return False


def aggregate_results(output_dir: Path) -> pd.DataFrame:
    """Scan the output directory for JSON results and aggregate them."""
    all_data = []
    for json_file in output_dir.glob("results_seed_*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            # Extract final metrics from the last epoch
            last_epoch = data["results"][-1]
            all_data.append({
                "seed": last_epoch["seed"],
                "avg_krsi": last_epoch["avg_krsi"],
                "total_reward": last_epoch["epoch_reward"],
                "config_type": data["config"].get("agent", {}).get("type", "unknown")
            })
    
    return pd.DataFrame(all_data)


def main():
    parser = argparse.ArgumentParser(description="KRSI Multi-Experiment Runner")
    parser.add_argument("--configs", nargs="+", default=["configs/default.yaml"], 
                        help="List of YAML configs to run")
    parser.add_argument("--output-dir", default="experiments/results", 
                        help="Root directory for all experiment results")
    args = parser.parse_args()

    root_output = Path(args.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for config_path in args.configs:
        config_name = Path(config_path).stem
        current_output = root_output / config_name
        current_output.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Running Experiment Suite: {config_name}")
        
        # Invoke the training script
        # Note: In a real research environment, we might use a task queue or parallelise this
        cmd = [
            sys.executable, "-m", "controller.train",
            "--config", config_path
        ]
        
        # Override output dir in the env or via config logic if needed
        # For now, we assume the config points to the right place or we move files after
        if run_command(cmd):
            df = aggregate_results(Path("datasets")) # default output
            
            if not df.empty:
                stats = {
                    "config": config_name,
                    "mean_krsi": df["avg_krsi"].mean(),
                    "std_krsi": df["avg_krsi"].std(),
                    "mean_reward": df["total_reward"].mean(),
                }
                summary_records.append(stats)
                
                # Save aggregated CSV for this config
                df.to_csv(current_output / "aggregated_results.csv", index=False)

    # Final summary table
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        print("\n" + "="*50)
        print("EXPERIMENT SUITE SUMMARY")
        print("="*50)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(root_output / "suite_summary.csv", index=False)


if __name__ == "__main__":
    main()
