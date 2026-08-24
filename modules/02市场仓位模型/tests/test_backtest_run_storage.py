from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYER4_DIR = next(PROJECT_ROOT.glob("04*"))
sys.path.insert(0, str(LAYER4_DIR))

from run_storage import load_merged_config, validate_run_id, write_json  # noqa: E402


def test_strategy_override_keeps_base_settings() -> None:
    config = load_merged_config(
        LAYER4_DIR / "config.yaml",
        LAYER4_DIR / "strategy_configs" / "bull90_neutral45_bear0.yaml",
    )
    assert config["backtest"]["target_exposure"] == pytest.approx(0.90)
    assert config["backtest"]["neutral_exposure"] == pytest.approx(0.45)
    assert config["backtest"]["bear_exposure"] == pytest.approx(0.0)
    assert config["backtest"]["commission"] == pytest.approx(0.002)
    assert config["data"]["predictions"]["ridge"]


@pytest.mark.parametrize("value", ["../escape", "bad id", "中文", "-leading"])
def test_run_id_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_run_id(value)


def test_performance_json_uses_standard_null(tmp_path: Path) -> None:
    path = tmp_path / "performance.json"
    write_json(path, {"win_rate": float("nan"), "sharpe": 1.2})
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw) == {"win_rate": None, "sharpe": 1.2}
