from __future__ import annotations

"""构建全部周度日期的组合决策输入历史表。"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

LAYER_ROOT = Path(__file__).resolve().parents[1]
if str(LAYER_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER_ROOT))

from adapters.alpha_reader import AlphaReleaseReader
from adapters.common import read_json, read_yaml, resolve_from, sha256_file
from adapters.market_data_reader import LocalMarketDataReader
from adapters.market_signal_reader import MarketSignalReleaseReader
from estimation.alpha.alpha_transformer import AlphaTransformer
from estimation.risk.covariance_estimator import LedoitWolfCovarianceEstimator
from estimation.risk.return_builder import ReturnMatrixBuilder
from universe.universe_builder import UniverseBuilder
from validation.snapshot_validator import SnapshotValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建全部周度日期的决策输入历史表")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-date", help="可选起始signal_date，默认使用全部共同周度日期")
    parser.add_argument("--end-date", help="可选结束signal_date，默认使用全部共同周度日期")
    parser.add_argument("--release-id")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def official_week_ends(schedule: pd.DataFrame) -> pd.DatetimeIndex:
    required = {"cal_date", "is_open"}
    if not required.issubset(schedule.columns):
        raise ValueError(f"交易安排缺少字段: {sorted(required - set(schedule.columns))}")
    dates = pd.to_datetime(schedule["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
    open_days = dates.loc[pd.to_numeric(schedule["is_open"], errors="coerce").eq(1)].dropna()
    return pd.DatetimeIndex(open_days.groupby(open_days.dt.to_period("W-FRI")).max()).sort_values()


def choose_dates(
    alpha_dates: pd.DatetimeIndex,
    market_signals: pd.DataFrame,
    week_ends: pd.DatetimeIndex,
    start_date: str | None,
    end_date: str | None,
) -> list[pd.Timestamp]:
    if market_signals["signal_date"].duplicated().any():
        raise ValueError("市场信号包含重复signal_date")
    market_dates = pd.DatetimeIndex(market_signals["signal_date"].unique()).normalize().sort_values()
    outside_calendar = market_dates.difference(week_ends)
    if len(outside_calendar):
        raise ValueError(f"市场信号包含非周末交易日: {outside_calendar.strftime('%Y-%m-%d').tolist()}")
    common = market_dates.intersection(alpha_dates.normalize()).sort_values()
    if start_date:
        common = common[common >= pd.Timestamp(start_date).normalize()]
    if end_date:
        common = common[common <= pd.Timestamp(end_date).normalize()]
    if len(common) == 0:
        raise ValueError("指定范围内没有Alpha与市场信号的共同周度日期")
    return [pd.Timestamp(date).normalize() for date in common]


def exact_row(frame: pd.DataFrame, date: pd.Timestamp, name: str) -> pd.Series:
    if date not in frame.index:
        raise ValueError(f"{name}缺少{date.date()}")
    row = frame.loc[date]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"{name}在{date.date()}存在重复日期")
    return row


def existing_release_is_current(
    release_dir: Path,
    output_root: Path,
    release_id: str,
    config: dict,
    source_releases: dict,
) -> bool:
    if not release_dir.exists():
        return False
    manifest_path = release_dir / "manifest.json"
    current_path = output_root / "current.json"
    if not manifest_path.exists() or not current_path.exists():
        raise FileExistsError(f"历史发布目录不完整，不允许覆盖: {release_dir}")
    manifest = read_json(manifest_path)
    current = read_json(current_path)
    same = (
        manifest.get("schema_version") == "decision_history_v1"
        and manifest.get("release_id") == release_id
        and manifest.get("source_releases") == source_releases
        and manifest.get("config") == config
        and current.get("release_id") == release_id
    )
    if not same:
        raise FileExistsError(f"同名历史发布的输入或配置不同: {release_dir}")
    for name, expected_hash in manifest.get("files", {}).items():
        path = release_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"历史发布文件缺失或哈希不一致: {path}")
    print(json.dumps({"status": "already_current", "release_id": release_id}, ensure_ascii=False))
    return True


def main() -> None:
    args = parse_args()
    config = read_yaml(resolve_from(LAYER_ROOT, args.config))
    if config.get("schema_version") != "decision_input_config_v1":
        raise ValueError("配置schema_version不是decision_input_config_v1")

    data_root = resolve_from(LAYER_ROOT, config["paths"]["local_data_root"])
    output_root = resolve_from(LAYER_ROOT, config["paths"]["output_root"])
    alpha_reader = AlphaReleaseReader(data_root / "source_alpha")
    market_reader = MarketSignalReleaseReader(data_root / "source_market_signal")
    data_reader = LocalMarketDataReader(data_root)
    alpha_manifest = alpha_reader.validate()
    market_manifest = market_reader.validate()
    source_releases = {
        "alpha": alpha_manifest.get("release_id"),
        "market_signal": market_manifest.get("release_id"),
    }

    market_signals = market_reader.read_all()
    dates = choose_dates(
        alpha_reader.available_dates(),
        market_signals,
        official_week_ends(data_reader.read_trade_schedule()),
        args.start_date,
        args.end_date,
    )
    first_date, last_date = dates[0], dates[-1]
    release_id = args.release_id or f"decision_history_{first_date:%Y%m%d}_{last_date:%Y%m%d}_v1"
    releases_root = output_root / "releases"
    release_dir = releases_root / release_id
    if existing_release_is_current(release_dir, output_root, release_id, config, source_releases):
        return

    horizon_days = int(config["decision"]["horizon_days"])
    top_n = int(config["decision"]["top_n"])
    selected_market = market_signals.loc[market_signals["signal_date"].isin(dates)].copy()
    if not selected_market["horizon_days"].eq(horizon_days).all():
        raise ValueError("市场信号预测周期与决策层配置不一致")
    market_by_date = selected_market.set_index(
        "signal_date", drop=False, verify_integrity=True
    )

    print("加载全量历史行情和状态...", flush=True)
    close = data_reader.read_market_field("close")
    st_status = data_reader.read_status("st_status")
    suspend_status = data_reader.read_status("suspend_status")
    stock_info = data_reader.read_stock_info()
    observation_counts = close.notna().cumsum()

    universe_builder = UniverseBuilder(
        board_scope=config["decision"]["board_scope"],
        min_price_observations=config["universe"]["min_price_observations"],
        require_non_st=config["universe"]["require_non_st"],
        require_not_suspended=config["universe"]["require_not_suspended"],
    )
    alpha_transformer = AlphaTransformer(config["alpha"]["rank_method"])
    return_builder = ReturnMatrixBuilder(config["risk"]["lookback_returns"])
    covariance_estimator = LedoitWolfCovarianceEstimator(config["risk"]["covariance_horizon_days"])
    validator = SnapshotValidator(top_n, horizon_days)

    decision_frames: list[pd.DataFrame] = []
    covariance_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for number, signal_date in enumerate(dates, 1):
        print(f"[{number}/{len(dates)}] 构建 {signal_date:%Y-%m-%d}", flush=True)
        alpha = alpha_reader.read_date(signal_date)
        if not alpha["horizon_days"].eq(horizon_days).all():
            raise ValueError(f"{signal_date.date()} Alpha预测周期不一致")
        codes = alpha["stock_code"].astype(str).tolist()
        missing_close = sorted(set(codes) - set(close.columns))
        if missing_close:
            raise ValueError(f"close缺少Alpha股票，示例: {missing_close[:5]}")
        close_asof = close.loc[close.index <= signal_date, codes]
        if close_asof.empty:
            raise ValueError(f"{signal_date.date()}以前没有行情")
        count_row = observation_counts.loc[observation_counts.index <= signal_date].iloc[-1].reindex(codes)
        universe = universe_builder.build(
            alpha=alpha,
            stock_info=stock_info,
            st_status=exact_row(st_status, signal_date, "st_status"),
            suspend_status=exact_row(suspend_status, signal_date, "suspend_status"),
            price_observations=count_row,
            signal_date=signal_date,
        )
        universe, selected = alpha_transformer.transform(universe, top_n)
        selected["horizon_days"] = horizon_days
        selected_codes = selected["stock_code"].astype(str).tolist()
        _, returns = return_builder.build(close_asof, selected_codes, signal_date)
        covariance, asset_risk, diagnostics = covariance_estimator.fit(returns, signal_date)
        market_signal = market_by_date.loc[signal_date]
        validation = validator.validate(
            universe, selected, covariance, asset_risk, market_signal, signal_date
        )

        decision = selected[
            [
                "signal_date", "stock_code", "industry", "alpha_score", "source_alpha_rank",
                "eligible_alpha_rank", "alpha_zscore", "selection_rank", "alpha_method",
                "horizon_days",
            ]
        ].copy()
        decision = decision.merge(
            asset_risk[["stock_code", "volatility_daily", "volatility_20d"]],
            on="stock_code",
            how="left",
            validate="one_to_one",
        )
        for column in MarketSignalReleaseReader.DECISION_COLUMNS:
            if column != "signal_date" and column != "horizon_days":
                decision[column] = market_signal[column]
        decision_frames.append(decision)
        covariance_frames.append(covariance)
        summary_rows.append(
            {
                "signal_date": signal_date,
                "eligible_stock_count": int(universe["is_eligible"].sum()),
                "selected_stock_count": int(len(selected)),
                "target_equity_exposure": float(market_signal["target_equity_exposure"]),
                "market_state": str(market_signal["market_state"]),
                "shrinkage": float(diagnostics["shrinkage"]),
                "min_eigenvalue_20d": float(validation["min_eigenvalue_20d"]),
                "validation_status": str(validation["status"]),
            }
        )

    decision_inputs = pd.concat(decision_frames, ignore_index=True)
    covariance_history = pd.concat(covariance_frames, ignore_index=True)
    period_summary = pd.DataFrame(summary_rows)
    if decision_inputs.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("历史决策输入存在重复的日期+股票")
    if covariance_history.duplicated(["signal_date", "asset_i", "asset_j"]).any():
        raise ValueError("历史协方差存在重复的日期+股票对")
    if not decision_inputs.groupby("signal_date").size().eq(top_n).all():
        raise ValueError("并非每一期都严格包含Top N股票")

    releases_root.mkdir(parents=True, exist_ok=True)
    temp_dir = releases_root / f".{release_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    decision_inputs.to_parquet(temp_dir / "decision_inputs.parquet", index=False)
    covariance_history.to_parquet(temp_dir / "covariance.parquet", index=False)
    period_summary.to_parquet(temp_dir / "period_summary.parquet", index=False)
    (temp_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    file_names = ["decision_inputs.parquet", "covariance.parquet", "period_summary.parquet", "config.yaml"]
    manifest = {
        "schema_version": "decision_history_v1",
        "release_id": release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "horizon_days": horizon_days,
        "top_n": top_n,
        "source_releases": source_releases,
        "config": config,
        "row_counts": {
            "decision_inputs": int(len(decision_inputs)),
            "covariance": int(len(covariance_history)),
            "period_summary": int(len(period_summary)),
        },
        "validation": {"status": "passed", "period_count": int(len(dates))},
        "files": {name: sha256_file(temp_dir / name) for name in file_names},
    }
    write_json(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, release_dir)

    current = {
        "schema_version": "decision_history_current_v1",
        "release_id": release_id,
        "manifest": f"releases/{release_id}/manifest.json",
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    current_temp = output_root / ".current.json.tmp"
    write_json(current_temp, current)
    os.replace(current_temp, output_root / "current.json")
    print(json.dumps(current, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
