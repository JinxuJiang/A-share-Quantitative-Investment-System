"""分别优化周度与月度组合权重历史。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LAYER_ROOT = Path(__file__).resolve().parent
if str(LAYER_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER_ROOT))

from optimization.alpha_risk_optimizer import AlphaRiskOptimizer
try:
    from validation.optimization_validator import OptimizationValidator
except ModuleNotFoundError:
    from optimization_validator import OptimizationValidator

FREQUENCIES = ("weekly", "monthly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="优化周度与月度组合历史")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--decision-release")
    parser.add_argument("--release-id")
    parser.add_argument(
        "--frequency", choices=(*FREQUENCIES, "both"), default="both"
    )
    return parser.parse_args()


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_decision_history(
    output_root: Path, release_id: str | None
) -> tuple[Path, dict]:
    if release_id is None:
        current = read_json(output_root / "current.json")
        if current.get("schema_version") != "decision_history_current_v2":
            raise ValueError("01层 current 不是日度决策历史，请先重建01层")
        release_id = str(current["release_id"])
    release_dir = output_root / "releases" / release_id
    manifest = read_json(release_dir / "manifest.json")
    if manifest.get("schema_version") != "decision_history_v2":
        raise ValueError("01层 manifest 不是 decision_history_v2")
    if manifest.get("decision_frequency") != "daily":
        raise ValueError("01层决策历史必须覆盖每日交易日")
    for name, expected_hash in manifest["files"].items():
        path = release_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"01层文件缺失或哈希不一致: {path}")
    return release_dir, manifest


def official_sessions(schedule: pd.DataFrame) -> pd.DatetimeIndex:
    required = {"cal_date", "is_open"}
    if not required.issubset(schedule.columns):
        raise ValueError(f"交易日历缺少字段: {sorted(required - set(schedule.columns))}")
    dates = pd.to_datetime(
        schedule["cal_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    open_days = dates.loc[
        pd.to_numeric(schedule["is_open"], errors="coerce").eq(1)
    ].dropna()
    return pd.DatetimeIndex(open_days).normalize().drop_duplicates().sort_values()


def select_rebalance_dates(
    decision_dates: pd.DatetimeIndex,
    schedule: pd.DataFrame,
    frequency: str,
) -> list[pd.Timestamp]:
    if frequency not in FREQUENCIES:
        raise ValueError(f"不支持的调仓频率: {frequency}")
    sessions = official_sessions(schedule)
    available = pd.DatetimeIndex(decision_dates).normalize().intersection(sessions)
    period_alias = "W-FRI" if frequency == "weekly" else "M"
    official = pd.DataFrame({"date": sessions})
    official["period"] = official["date"].dt.to_period(period_alias)
    official_last = official.groupby("period", sort=True)["date"].max()
    candidates = pd.DataFrame({"date": available})
    candidates["period"] = candidates["date"].dt.to_period(period_alias)
    available_last = candidates.groupby("period", sort=True)["date"].max()
    completed = available_last[
        available_last.eq(official_last.reindex(available_last.index))
    ]
    if completed.empty:
        raise ValueError(f"没有完整结束的 {frequency} 决策周期")
    return [pd.Timestamp(value).normalize() for value in completed]


def existing_release_is_current(
    release_dir: Path,
    output_root: Path,
    release_id: str,
    source_release: str,
    frequency: str,
    config: dict,
) -> bool:
    if not release_dir.exists():
        return False
    manifest = read_json(release_dir / "manifest.json")
    current = read_json(output_root / "current.json")
    same = (
        manifest.get("schema_version") == "portfolio_history_v2"
        and manifest.get("source_decision_release") == source_release
        and manifest.get("rebalance_frequency") == frequency
        and manifest.get("config") == config
        and current.get("release_id") == release_id
    )
    if not same:
        raise FileExistsError(f"同名优化版本的输入或配置不同: {release_dir}")
    for name, expected_hash in manifest["files"].items():
        path = release_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"优化文件缺失或哈希不一致: {path}")
    print(json.dumps({"status": "already_current", "release_id": release_id}))
    return True


def optimize_track(
    *,
    frequency: str,
    dates: list[pd.Timestamp],
    decisions: pd.DataFrame,
    covariance_history: pd.DataFrame,
    source_dir: Path,
    source_manifest: dict,
    config: dict,
    output_root: Path,
    requested_release_id: str | None,
) -> None:
    source_release = str(source_manifest["release_id"])
    first_date, last_date = dates[0], dates[-1]
    release_id = requested_release_id or (
        f"portfolio_{frequency}_history_{first_date:%Y%m%d}_{last_date:%Y%m%d}_v1"
    )
    releases_root = output_root / "releases"
    release_dir = releases_root / release_id
    if existing_release_is_current(
        release_dir, output_root, release_id, source_release, frequency, config
    ):
        return

    objective = config["objective"]
    constraints = config["constraints"]
    solver = config["solver"]
    validation_config = config["validation"]
    active_threshold = float(validation_config["active_weight_threshold"])
    optimizer = AlphaRiskOptimizer(
        risk_weight=objective["risk_weight"],
        alpha_weight=objective["alpha_weight"],
        min_stock_weight=constraints["min_stock_weight"],
        max_stock_weight=constraints["max_stock_weight"],
        max_industry_weight=constraints["max_industry_weight"],
        turnover_penalty=objective.get("turnover_penalty", 0.0),
        turnover_smoothing=objective.get("turnover_smoothing", 1.0e-6),
        max_iterations=solver["max_iterations"],
        tolerance=solver["tolerance"],
    )
    validator = OptimizationValidator(
        constraints["min_stock_weight"],
        constraints["max_stock_weight"],
        constraints["max_industry_weight"],
        validation_config["weight_tolerance"],
    )

    decision_dates = pd.to_datetime(decisions["signal_date"]).dt.normalize()
    covariance_dates = pd.to_datetime(
        covariance_history["signal_date"]
    ).dt.normalize()
    weight_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    previous_target_weights: dict[str, float] = {}
    for number, signal_date in enumerate(dates, start=1):
        print(
            f"[{frequency} {number}/{len(dates)}] 优化 {signal_date:%Y-%m-%d}",
            flush=True,
        )
        alpha = decisions.loc[decision_dates.eq(signal_date)].copy()
        alpha = alpha.sort_values("selection_rank")
        if alpha["target_equity_exposure"].nunique() != 1:
            raise ValueError(f"{signal_date.date()} 市场仓位不唯一")
        market_exposure = float(alpha["target_equity_exposure"].iloc[0])
        codes = alpha["stock_code"].astype(str).tolist()
        current_codes = set(codes)
        previous_aligned = np.array(
            [previous_target_weights.get(code, 0.0) for code in codes],
            dtype="float64",
        )
        previous_outside_weight = float(
            sum(
                weight
                for code, weight in previous_target_weights.items()
                if code not in current_codes
            )
        )
        covariance_long = covariance_history.loc[covariance_dates.eq(signal_date)]
        covariance = covariance_long.pivot(
            index="asset_i", columns="asset_j", values="covariance_20d"
        ).reindex(index=codes, columns=codes)
        if covariance.isna().any().any():
            raise ValueError(f"{signal_date.date()} 协方差与候选股票不一致")

        result = optimizer.solve(
            alpha["alpha_zscore"].to_numpy(dtype="float64"),
            covariance.to_numpy(dtype="float64"),
            alpha["industry"].astype(str).to_numpy(),
            target_exposure=market_exposure,
            previous_account_weights=previous_aligned,
            previous_outside_weight=previous_outside_weight,
        )
        deployed_exposure = float(result.diagnostics["deployed_stock_exposure"])
        weights = alpha[
            ["signal_date", "stock_code", "industry", "selection_rank", "alpha_zscore"]
        ].copy()
        weights["rebalance_frequency"] = frequency
        weights["base_weight"] = result.weights
        weights["target_weight"] = result.weights * deployed_exposure
        weights["base_weight_pct"] = (weights["base_weight"] * 100).round(2)
        weights["target_weight_pct"] = (weights["target_weight"] * 100).round(2)
        weights["optimized_rank"] = weights["base_weight"].rank(
            method="first", ascending=False
        ).astype("int32")
        weights["is_active"] = weights["target_weight"].gt(active_threshold)
        weights["market_target_exposure"] = market_exposure
        weights["stock_exposure"] = deployed_exposure
        weights["exposure_shortfall"] = market_exposure - deployed_exposure
        weights["cash_weight"] = 1.0 - deployed_exposure
        validation = validator.validate(weights, market_exposure, deployed_exposure)
        previous_target_weights = dict(
            zip(
                weights["stock_code"].astype(str),
                weights["target_weight"].astype(float),
            )
        )
        weights = weights.sort_values("optimized_rank").reset_index(drop=True)
        weight_frames.append(weights)

        diagnostics = result.diagnostics
        summary_rows.append(
            {
                "signal_date": signal_date,
                "rebalance_frequency": frequency,
                "market_target_exposure": market_exposure,
                "maximum_feasible_exposure": float(
                    diagnostics["maximum_feasible_exposure"]
                ),
                "stock_exposure": deployed_exposure,
                "exposure_shortfall": float(diagnostics["exposure_shortfall"]),
                "cash_weight": float(1.0 - deployed_exposure),
                "candidate_stock_count": int(len(alpha)),
                "active_stock_count": int(weights["is_active"].sum()),
                "predicted_volatility_20d_internal": float(
                    diagnostics["optimized_volatility_20d"]
                ),
                "predicted_volatility_20d_account": float(
                    diagnostics["optimized_volatility_20d"] * deployed_exposure
                ),
                "weighted_alpha_zscore": float(
                    np.dot(weights["alpha_zscore"], weights["base_weight"])
                ),
                "objective_value": float(diagnostics["objective_value"]),
                "estimated_trade_weight": float(
                    diagnostics["estimated_trade_weight"]
                ),
                "turnover_penalty_value": float(
                    diagnostics["turnover_penalty_value"]
                ),
                "solver_success": bool(diagnostics["success"]),
                "solver_iterations": int(diagnostics["iterations"]),
                "max_stock_weight": float(validation["max_stock_weight"]),
                "max_industry_weight": float(validation["max_industry_weight"]),
                "validation_status": str(validation["status"]),
            }
        )

    all_weights = pd.concat(weight_frames, ignore_index=True)
    portfolio_summary = pd.DataFrame(summary_rows)
    if all_weights.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("权重历史存在重复日期+股票")
    if not all_weights.groupby("signal_date").size().eq(
        int(source_manifest["top_n"])
    ).all():
        raise ValueError("并非每一期权重都包含 top_n 股票")
    if not portfolio_summary["solver_success"].all():
        raise ValueError("至少一期历史优化未收敛")

    releases_root.mkdir(parents=True, exist_ok=True)
    temporary = releases_root / f".{release_id}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    all_weights.to_parquet(temporary / "weights.parquet", index=False)
    portfolio_summary.to_parquet(
        temporary / "portfolio_summary.parquet", index=False
    )
    (temporary / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    file_names = ["weights.parquet", "portfolio_summary.parquet", "config.yaml"]
    manifest = {
        "schema_version": "portfolio_history_v2",
        "release_id": release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "rebalance_frequency": frequency,
        "horizon_days": int(source_manifest["horizon_days"]),
        "source_decision_release": source_release,
        "source_decision_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "config": config,
        "row_counts": {
            "weights": int(len(all_weights)),
            "portfolio_summary": int(len(portfolio_summary)),
        },
        "validation": {"status": "passed", "period_count": int(len(dates))},
        "files": {name: sha256_file(temporary / name) for name in file_names},
    }
    write_json(temporary / "manifest.json", manifest)
    os.replace(temporary, release_dir)

    current = {
        "schema_version": "portfolio_history_current_v2",
        "release_id": release_id,
        "manifest": f"releases/{release_id}/manifest.json",
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "rebalance_frequency": frequency,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_current = output_root / ".current.json.tmp"
    write_json(temporary_current, current)
    os.replace(temporary_current, output_root / "current.json")
    print(json.dumps(current, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(
        resolve_from(LAYER_ROOT, args.config).read_text(encoding="utf-8")
    )
    if config.get("schema_version") != "portfolio_optimization_config_v3":
        raise ValueError("配置 schema_version 不是 portfolio_optimization_config_v3")
    if config["constraints"].get("weight_basis") != "account":
        raise ValueError("正式组合约束必须使用账户权重")
    if config["constraints"].get("market_exposure_mode") != "target_with_cash_fallback":
        raise ValueError("正式市场仓位模式必须允许不足部分保留现金")

    decision_output = resolve_from(LAYER_ROOT, config["paths"]["decision_output_root"])
    base_output_root = resolve_from(LAYER_ROOT, config["paths"]["output_root"])
    schedule_path = resolve_from(LAYER_ROOT, config["paths"]["trade_schedule"])
    source_dir, source_manifest = load_decision_history(
        decision_output, args.decision_release
    )
    decisions = pd.read_parquet(source_dir / "decision_inputs.parquet")
    covariance_history = pd.read_parquet(source_dir / "covariance.parquet")
    daily_dates = pd.DatetimeIndex(
        pd.to_datetime(decisions["signal_date"]).dt.normalize().unique()
    ).sort_values()
    if len(daily_dates) != int(source_manifest["period_count"]):
        raise ValueError("决策日期数量与01层 manifest 不一致")
    schedule = pd.read_parquet(schedule_path)
    frequencies = FREQUENCIES if args.frequency == "both" else (args.frequency,)
    if args.release_id and len(frequencies) != 1:
        raise ValueError("--release-id 必须与单一 --frequency 同时使用")
    for frequency in frequencies:
        dates = select_rebalance_dates(daily_dates, schedule, frequency)
        optimize_track(
            frequency=frequency,
            dates=dates,
            decisions=decisions,
            covariance_history=covariance_history,
            source_dir=source_dir,
            source_manifest=source_manifest,
            config=config,
            output_root=base_output_root / frequency,
            requested_release_id=args.release_id,
        )


if __name__ == "__main__":
    main()
