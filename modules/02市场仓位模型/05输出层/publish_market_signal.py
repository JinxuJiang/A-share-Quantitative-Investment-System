# -*- coding: utf-8 -*-
"""将已验收的市场模型发布为稳定的周频仓位信号。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAYER3_DIR = PROJECT_ROOT / "03模型训练层"
LAYER4_DIR = PROJECT_ROOT / "04经济意义与回测层"
EXPORT_ROOT = Path(__file__).resolve().parent / "exports"
TRAINING_CONFIG_PATH = LAYER3_DIR / "config.yaml"
SIGNAL_CONFIG_PATH = LAYER4_DIR / "config.yaml"
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SUPPORTED_MODELS = ("ridge", "cnn_gru")

sys.path.insert(0, str(LAYER4_DIR))
from backtest import (  # noqa: E402
    build_decision_dates,
    build_rebalance_plan,
    common_prediction_dates,
    load_predictions,
    load_price,
    load_yaml,
    resolve_layer_path,
    slice_backtest_period,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发布正式市场仓位信号")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--release-id", required=True, help="不可覆盖的发布ID")
    parser.add_argument("--run-id", required=True, help="已经回测并验收通过的策略run-id")
    parser.add_argument(
        "--set-current",
        action="store_true",
        help="发布成功后将current.json明确指向本release",
    )
    return parser.parse_args()


def validate_id(value: str, field_name: str) -> str:
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name}只能包含英文字母、数字、点、下划线和连字符: {value}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def build_market_signal(model_name: str, signal_config: dict) -> tuple[pd.DataFrame, Path]:
    prediction_paths = {
        name: resolve_layer_path(path)
        for name, path in signal_config["data"]["predictions"].items()
    }
    if model_name not in prediction_paths:
        raise ValueError(f"信号配置未声明模型: {model_name}")

    prediction = load_predictions(prediction_paths[model_name], model_name)
    predictions = {model_name: prediction}
    common_dates = common_prediction_dates(predictions)
    price = load_price(resolve_layer_path(signal_config["data"]["target_price"]))
    price = slice_backtest_period(
        price,
        signal_config["backtest"],
        default_end_date=common_dates.max(),
    )
    decision_dates = build_decision_dates(
        common_dates,
        price.index,
        str(signal_config["backtest"]["rebalance_frequency"]),
    )
    plan = build_rebalance_plan(
        model_name,
        prediction,
        decision_dates,
        price.index,
        signal_config["backtest"],
    )

    training_config = load_yaml(TRAINING_CONFIG_PATH)
    entry_offset = int(training_config["label"]["entry_offset"])
    exit_offset = int(training_config["label"]["exit_offset"])
    horizon_days = exit_offset - entry_offset
    if horizon_days <= 0 or horizon_days > np.iinfo(np.int16).max:
        raise ValueError(f"预测周期无效: {horizon_days}")

    signal = plan.rename(
        columns={
            "planned_execution_date": "execution_date",
            "ts_code": "market_code",
            "prediction_smoothed": "forecast_return",
            "prediction_zscore": "signal_zscore",
            "desired_state": "market_state",
            "target_exposure": "target_equity_exposure",
        }
    )[
        [
            "signal_date",
            "execution_date",
            "market_code",
            "forecast_return",
            "signal_zscore",
            "market_state",
            "target_equity_exposure",
        ]
    ].copy()
    signal["signal_date"] = pd.to_datetime(signal["signal_date"], errors="raise")
    signal["execution_date"] = pd.to_datetime(signal["execution_date"], errors="coerce")
    signal["market_code"] = signal["market_code"].astype("string")
    signal["market_state"] = signal["market_state"].astype("string")
    signal["horizon_days"] = np.int16(horizon_days)

    required = [
        "signal_date",
        "market_code",
        "forecast_return",
        "signal_zscore",
        "market_state",
        "target_equity_exposure",
        "horizon_days",
    ]
    if signal[required].isna().any().any():
        raise ValueError("正式市场信号的必要字段存在缺失")
    numeric = signal[["forecast_return", "signal_zscore", "target_equity_exposure"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("正式市场信号包含NaN或无穷值")
    if signal["signal_date"].duplicated().any():
        raise ValueError("正式市场信号存在重复signal_date")
    if not signal["signal_date"].is_monotonic_increasing:
        raise ValueError("正式市场信号未按signal_date递增")
    if not signal["market_code"].eq("000852.SH").all():
        raise ValueError("正式市场信号包含非中证1000标的")
    if not signal["market_state"].isin(["bear", "neutral", "bull"]).all():
        raise ValueError("正式市场信号包含未知市场状态")
    if not signal["target_equity_exposure"].between(0.0, 1.0).all():
        raise ValueError("正式市场信号目标仓位超出[0,1]")
    known_execution = signal["execution_date"].notna()
    if not (
        signal.loc[known_execution, "execution_date"]
        > signal.loc[known_execution, "signal_date"]
    ).all():
        raise ValueError("已知execution_date必须晚于signal_date")
    missing_execution = signal["execution_date"].isna()
    if missing_execution.sum() > 1 or (
        missing_execution.any() and not bool(missing_execution.iloc[-1])
    ):
        raise ValueError("只有最新信号允许缺少execution_date")

    return signal, prediction_paths[model_name]


def publish(model_name: str, release_id: str, run_id: str, set_current: bool = False) -> Path:
    model_name = validate_id(model_name, "model")
    release_id = validate_id(release_id, "release-id")
    run_id = validate_id(run_id, "run-id")
    base_signal_config = load_yaml(SIGNAL_CONFIG_PATH)
    report_root = resolve_layer_path(base_signal_config["output"]["reports_dir"])
    data_root = resolve_layer_path(base_signal_config["output"]["data_dir"])
    run_report = report_root / run_id
    run_config_path = run_report / "config_snapshot.yaml"
    run_manifest_path = run_report / "run_manifest.json"
    validation_path = data_root / "processed" / run_id / "logs" / "backtest_validation_report.json"
    if not (run_config_path.exists() and run_manifest_path.exists() and validation_path.exists()):
        raise FileNotFoundError(f"run-id is not a complete validated backtest: {run_id}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if model_name not in run_manifest["models"]:
        raise ValueError(f"run-id {run_id} does not contain model {model_name}")
    if int(validation["summary"]["fail"]) != 0:
        raise ValueError(f"run-id {run_id} did not pass backtest validation")
    signal_config = load_yaml(run_config_path)
    training_config = load_yaml(TRAINING_CONFIG_PATH)

    releases_dir = EXPORT_ROOT / "releases"
    release_dir = releases_dir / release_id
    if release_dir.exists():
        raise FileExistsError(f"release已存在，禁止覆盖: {release_dir}")
    releases_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=releases_dir))
    published_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")

    try:
        signal, source_path = build_market_signal(model_name, signal_config)
        output_path = temp_dir / "market_signal.parquet"
        signal.to_parquet(output_path, index=False, compression="zstd")
        git_commit, git_dirty = git_metadata()
        backtest_config = signal_config["backtest"]
        smoothing_config = training_config["prediction_smoothing"]
        horizon_days = int(signal["horizon_days"].iloc[0])
        model_config_path = LAYER3_DIR / "models" / model_name / "config.yaml"
        manifest = {
            "schema_version": "market_signal_v1",
            "release_id": release_id,
            "source_backtest_run_id": run_id,
            "source_model": model_name,
            "source_file": str(source_path.relative_to(PROJECT_ROOT)),
            "source_file_sha256": sha256_file(source_path),
            "source_training_config_sha256": sha256_file(TRAINING_CONFIG_PATH),
            "source_model_config_sha256": sha256_file(model_config_path),
            "source_signal_config_sha256": sha256_file(run_config_path),
            "source_backtest_manifest_sha256": sha256_file(run_manifest_path),
            "source_backtest_validation_sha256": sha256_file(validation_path),
            "source_git_commit": git_commit,
            "source_git_dirty": git_dirty,
            "target_market": "000852.SH",
            "label_name": training_config["label"]["name"],
            "label_formula": "open[T+21] / open[T+1] - 1",
            "forecast_horizon_days": horizon_days,
            "forecast_column": "smoothed_prediction",
            "smoothing": {
                "method": smoothing_config["method"],
                "halflife_days": int(smoothing_config["halflife_days"]),
                "adjust": bool(smoothing_config["adjust"]),
                "min_periods": int(smoothing_config["min_periods"]),
            },
            "signal_policy": {
                "rebalance_frequency": backtest_config["rebalance_frequency"],
                "standardization_window": int(backtest_config["standardization_window"]),
                "standardization_uses_current_observation": False,
                "forecast_threshold": float(backtest_config["signal_threshold"]),
                "state_z_threshold": float(backtest_config["state_z_threshold"]),
                "exposure_mapping": {
                    "bear": float(backtest_config["bear_exposure"]),
                    "neutral": float(backtest_config["neutral_exposure"]),
                    "bull": float(backtest_config["target_exposure"]),
                },
            },
            "signal_available_after": "market_close",
            "execution_rule": "NEXT_TRADING_DAY_OPEN",
            "execution_lag_trading_days": 1,
            "nullable_execution_date": "仅最新信号在下一交易日行情尚未到达时允许为空",
            "data_start": signal["signal_date"].min().date().isoformat(),
            "data_end": signal["signal_date"].max().date().isoformat(),
            "published_at": published_at,
            "row_count": len(signal),
            "columns": {
                "signal_date": "datetime64[ns]",
                "execution_date": "datetime64[ns]; 最新信号允许为空",
                "market_code": "string",
                "forecast_return": str(signal["forecast_return"].dtype),
                "signal_zscore": str(signal["signal_zscore"].dtype),
                "market_state": "string; bear/neutral/bull",
                "target_equity_exposure": str(signal["target_equity_exposure"].dtype),
                "horizon_days": "int16",
            },
            "output_sha256": sha256_file(output_path),
        }
        write_json_atomic(temp_dir / "manifest.json", manifest)
        temp_dir.replace(release_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    if set_current:
        current = {
            "schema_version": "market_signal_current_v1",
            "release_id": release_id,
            "manifest": f"releases/{release_id}/manifest.json",
            "updated_at": published_at,
        }
        write_json_atomic(EXPORT_ROOT / "current.json", current)
    return release_dir


def main() -> None:
    args = parse_args()
    release_dir = publish(args.model, args.release_id, args.run_id, args.set_current)
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "release_id": manifest["release_id"],
        "source_model": manifest["source_model"],
        "source_backtest_run_id": manifest["source_backtest_run_id"],
        "date_start": manifest["data_start"],
        "date_end": manifest["data_end"],
        "rows": manifest["row_count"],
        "release": str(release_dir),
        "set_current": bool(args.set_current),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
