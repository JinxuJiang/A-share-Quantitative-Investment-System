"""Versioned storage helpers for market-timing backtest runs."""

from __future__ import annotations

import json
import math
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_run_id(value: str) -> str:
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "run-id may contain only ASCII letters, digits, dots, underscores and hyphens: "
            f"{value}"
        )
    return value


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_merged_config(base_path: Path, strategy_path: Path | None = None) -> dict:
    with Path(base_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if strategy_path is not None:
        with Path(strategy_path).open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
        config = deep_merge(config, override)
    backtest = config["backtest"]
    bear = float(backtest["bear_exposure"])
    neutral = float(backtest["neutral_exposure"])
    bull = float(backtest["target_exposure"])
    if not (0.0 <= bear <= neutral <= bull <= 1.0):
        raise ValueError("exposures must satisfy 0 <= bear <= neutral <= bull <= 1")
    return config


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def git_metadata(project_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, check=True,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None
