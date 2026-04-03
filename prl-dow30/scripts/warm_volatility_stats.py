#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.train import prepare_market_and_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm the split-specific volatility stats file before parallel training.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["config_path"] = args.config
    data_cfg = cfg.get("data", {}) or {}
    env_cfg = cfg.get("env", {}) or {}
    paper_mode = bool(data_cfg.get("paper_mode", False))
    require_cache = bool(data_cfg.get("require_cache", False) or paper_mode)
    offline = bool(data_cfg.get("offline", False) or paper_mode)
    cache_only = require_cache or offline

    _, features = prepare_market_and_features(
        config=cfg,
        lv=int(env_cfg["Lv"]),
        force_refresh=bool(data_cfg.get("force_refresh", True)),
        offline=offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        session_opts=data_cfg.get("session_opts"),
        cache_only=cache_only,
    )
    print(features.stats_path)


if __name__ == "__main__":
    main()
