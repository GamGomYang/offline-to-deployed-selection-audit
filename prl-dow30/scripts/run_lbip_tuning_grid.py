#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small LBIP hyperparameter grid on the validation split.")
    parser.add_argument("--base-config", required=True, help="Base LBIP validation YAML.")
    parser.add_argument("--out-root", required=True, help="Directory to store grid runs and summary.")
    parser.add_argument("--ridge-grid", default="3,10,30", help="Comma-separated ridge_alpha grid.")
    parser.add_argument("--training-eta-grid", default="0.05,0.082,0.1", help="Comma-separated training_eta grid.")
    parser.add_argument("--kappas", default="0,0.0005,0.001", help="Comma-separated validation kappas.")
    parser.add_argument("--etas", default="1,0.5,0.2,0.1,0.082,0.05,0.02", help="Comma-separated eta sweep.")
    parser.add_argument("--target-mode", default="mean_variance", help="Target mode for LBIP family.")
    parser.add_argument("--anchor-grid", default="0", help="Comma-separated anchor_strength grid.")
    parser.add_argument("--shrink-grid", default="0", help="Comma-separated equal_weight_shrink grid.")
    parser.add_argument("--offline", action="store_true", help="Use cache only / offline mode.")
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _fmt(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config).resolve()
    base_cfg = yaml.safe_load(base_config_path.read_text())

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    ridge_grid = _parse_float_list(args.ridge_grid)
    training_eta_grid = _parse_float_list(args.training_eta_grid)
    kappas = _parse_float_list(args.kappas)
    etas = _parse_float_list(args.etas)
    anchor_grid = _parse_float_list(args.anchor_grid)
    shrink_grid = _parse_float_list(args.shrink_grid)

    summary_rows: list[dict] = []

    for ridge_alpha in ridge_grid:
        for training_eta in training_eta_grid:
            for anchor_strength in anchor_grid:
                for equal_weight_shrink in shrink_grid:
                    cfg = json.loads(json.dumps(base_cfg))
                    lbip = cfg.setdefault("lbip", {})
                    lbip["ridge_alpha"] = float(ridge_alpha)
                    lbip["training_eta"] = float(training_eta)
                    lbip["target_mode"] = str(args.target_mode)
                    lbip["anchor_strength"] = float(anchor_strength)
                    lbip["equal_weight_shrink"] = float(equal_weight_shrink)
                    tag = f"ridge_{_fmt(ridge_alpha)}__traineta_{_fmt(training_eta)}__anchor_{_fmt(anchor_strength)}__shrink_{_fmt(equal_weight_shrink)}"
                    cfg["output"] = dict(cfg.get("output", {}) or {})
                    cfg["output"]["experiment_name"] = f"{cfg['output'].get('experiment_name', base_config_path.stem)}_{tag}"
                    run_out = out_root / tag
                    cfg["output"]["root"] = str(run_out)
                    cfg_path = cfg_dir / f"{tag}.yaml"
                    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

                    cmd = [
                        str(ROOT / ".venv" / "bin" / "python"),
                        str(ROOT / "scripts" / "run_information_parity_baselines.py"),
                        "--config",
                        str(cfg_path),
                        "--out",
                        str(run_out),
                        "--kappas",
                        *[str(v) for v in kappas],
                        "--etas",
                        *[str(v) for v in etas],
                    ]
                    if args.offline:
                        cmd.append("--offline")
                    subprocess.run(cmd, check=True)

                    selection_path = run_out / "selection" / "validation_eta_selection.json"
                    payload = json.loads(selection_path.read_text())
                    rows = pd.DataFrame(payload["rows"])
                    selected_eta = float(payload["selected_eta"])
                    selected_row = rows.loc[rows["selected"] == True].iloc[0]
                    best_row = rows.loc[rows["score_mean_median_sharpe_pos_kappa"].idxmax()]
                    summary_rows.append(
                        {
                            "target_mode": str(args.target_mode),
                            "anchor_strength": float(anchor_strength),
                            "equal_weight_shrink": float(equal_weight_shrink),
                            "ridge_alpha": float(ridge_alpha),
                            "training_eta": float(training_eta),
                            "selected_eta": selected_eta,
                            "selected_score": float(selected_row["score_mean_median_sharpe_pos_kappa"]),
                            "selected_delta_score_vs_eta1": float(selected_row["score_mean_median_delta_sharpe_vs_eta1_pos_kappa"]),
                            "selected_turnover": float(selected_row["median_turnover_exec_pos_kappa_mean"]),
                            "best_raw_eta": float(best_row["eta"]),
                            "best_raw_score": float(best_row["score_mean_median_sharpe_pos_kappa"]),
                            "run_root": str(run_out),
                        }
                    )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["selected_score", "selected_eta", "selected_delta_score_vs_eta1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary.to_csv(out_root / "grid_summary.csv", index=False)

    best = summary.iloc[0].to_dict()
    (out_root / "best_config.json").write_text(json.dumps(best, indent=2))
    print(f"BEST_RIDGE_ALPHA={best['ridge_alpha']}")
    print(f"BEST_TRAINING_ETA={best['training_eta']}")
    print(f"BEST_SELECTED_ETA={best['selected_eta']}")
    print(f"BEST_RUN_ROOT={best['run_root']}")


if __name__ == "__main__":
    main()
