#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "generalization" / "toy_example.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the appendix-only toy-example figure.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the toy-example YAML config.")
    return parser.parse_args()


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    input_csv = _resolve_path(config_path, str(cfg["outputs"]["csv"]))
    output_fig = _resolve_path(config_path, str(cfg["outputs"]["figure"]))

    df = pd.read_csv(input_csv).sort_values("friction")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    ax.axhline(0.0, color="#9ca3af", linewidth=1.0, linestyle="--")
    ax.plot(
        df["friction"],
        df["delta_target_tempered_minus_responsive"],
        marker="o",
        color="#1d4ed8",
        linewidth=2.0,
        label="Target-based gap",
    )
    ax.plot(
        df["friction"],
        df["delta_executed_tempered_minus_responsive"],
        marker="o",
        color="#d97706",
        linewidth=2.0,
        label="Executed-based gap",
    )
    ax.set_xlabel("Friction level")
    ax.set_ylabel("Tempered minus responsive score")
    ax.set_title("Evaluation Object Changes the Reading")
    ax.legend(frameon=False, loc="lower right")

    ax2 = axes[1]
    ax2.plot(
        df["friction"],
        df["mean_target_exec_gap_responsive"],
        marker="o",
        color="#b91c1c",
        linewidth=2.0,
        label="Responsive proposal-execution gap",
    )
    ax2.plot(
        df["friction"],
        df["mean_target_exec_gap_tempered"],
        marker="o",
        color="#047857",
        linewidth=2.0,
        label="Tempered proposal-execution gap",
    )
    ax2.set_xlabel("Friction level")
    ax2.set_ylabel("Mean |target - executed|")
    ax2.set_title("Friction Widens the Realized Gap")
    ax2.legend(frameon=False, loc="upper left")

    fig.suptitle("Appendix-Only Toy Example", y=1.02, fontsize=10)
    fig.tight_layout()

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[toy-example] wrote figure to {output_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
