"""分别计算周度与月度组合的参数法 VaR、ES 和可选仓位缩放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

LAYER_ROOT = Path(__file__).resolve().parent
if str(LAYER_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER_ROOT))

from risk.parametric_var_es import ParametricNormalVarEs, calculate_var_scale
from validation.risk_validator import RiskValidator

FREQUENCIES = ("weekly", "monthly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算周度与月度组合风险历史")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--portfolio-release")
    parser.add_argument("--release-id")
    parser.add_argument(
        "--frequency", choices=(*FREQUENCIES, "both"), default="both"
    )
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (LAYER_ROOT / path).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(release_dir: Path, manifest: dict) -> None:
    for name, expected_hash in manifest["files"].items():
        path = release_dir / name
        if not path.exists() or sha256(path) != expected_hash:
            raise ValueError(f"上游文件缺失或哈希不一致: {path}")


def load_portfolio_history(
    output_root: Path, release_id: str | None, frequency: str
) -> tuple[Path, dict]:
    if release_id is None:
        current = read_json(output_root / "current.json")
        if current.get("schema_version") != "portfolio_history_current_v2":
            raise ValueError("02层 current 不是分频组合历史")
        release_id = str(current["release_id"])
    release_dir = output_root / "releases" / release_id
    manifest = read_json(release_dir / "manifest.json")
    if manifest.get("schema_version") != "portfolio_history_v2":
        raise ValueError("02层 manifest 不是 portfolio_history_v2")
    if manifest.get("rebalance_frequency") != frequency:
        raise ValueError("02层版本频率与请求频率不一致")
    verify_files(release_dir, manifest)
    return release_dir, manifest


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
        manifest.get("schema_version") == "risk_history_v2"
        and manifest.get("source_portfolio_release") == source_release
        and manifest.get("rebalance_frequency") == frequency
        and manifest.get("config") == config
        and current.get("release_id") == release_id
    )
    if not same:
        raise FileExistsError(f"同名风险版本的输入或配置不同: {release_dir}")
    verify_files(release_dir, manifest)
    print(json.dumps({"status": "already_current", "release_id": release_id}))
    return True


def assess_track(
    *,
    frequency: str,
    horizon_days: int,
    portfolio_base: Path,
    decision_output: Path,
    output_root: Path,
    portfolio_release_arg: str | None,
    release_id_arg: str | None,
    config: dict,
) -> None:
    portfolio_dir, portfolio_manifest = load_portfolio_history(
        portfolio_base / frequency, portfolio_release_arg, frequency
    )
    portfolio_release = str(portfolio_manifest["release_id"])
    first_date = pd.Timestamp(portfolio_manifest["first_signal_date"])
    last_date = pd.Timestamp(portfolio_manifest["last_signal_date"])
    release_id = release_id_arg or (
        f"risk_{frequency}_{horizon_days}d_{first_date:%Y%m%d}_{last_date:%Y%m%d}_v1"
    )
    track_output = output_root / frequency
    releases_root = track_output / "releases"
    release_dir = releases_root / release_id
    if existing_release_is_current(
        release_dir,
        track_output,
        release_id,
        portfolio_release,
        frequency,
        config,
    ):
        return

    decision_release = str(portfolio_manifest["source_decision_release"])
    decision_dir = decision_output / "releases" / decision_release
    decision_manifest_path = decision_dir / "manifest.json"
    decision_manifest = read_json(decision_manifest_path)
    if decision_manifest.get("schema_version") != "decision_history_v2":
        raise ValueError("01层 manifest 不是 decision_history_v2")
    if sha256(decision_manifest_path) != portfolio_manifest[
        "source_decision_manifest_sha256"
    ]:
        raise ValueError("01层 manifest 哈希与02层引用不一致")
    verify_files(decision_dir, decision_manifest)

    weights_history = pd.read_parquet(portfolio_dir / "weights.parquet")
    portfolio_summary = pd.read_parquet(portfolio_dir / "portfolio_summary.parquet")
    covariance_history = pd.read_parquet(decision_dir / "covariance.parquet")
    dates = sorted(
        pd.to_datetime(portfolio_summary["signal_date"]).dt.normalize().unique()
    )
    if len(dates) != int(portfolio_manifest["period_count"]):
        raise ValueError("组合周期数量与 manifest 不一致")
    source_horizon = int(decision_manifest["horizon_days"])
    source_column = f"covariance_{source_horizon}d"
    covariance_column = f"covariance_{horizon_days}d"
    if source_column not in covariance_history.columns:
        raise ValueError(f"01层协方差缺少字段: {source_column}")

    tolerance = float(config["validation"]["weight_tolerance"])
    validator = RiskValidator(
        weight_tolerance=tolerance,
        psd_tolerance=float(config["validation"]["psd_tolerance"]),
    )
    risk_config = config["risk"]
    calculator = ParametricNormalVarEs(
        confidence_levels=risk_config["confidence_levels"],
        expected_return=float(risk_config["expected_return"]),
    )
    scaling_config = risk_config["scaling"]
    scaling_enabled = bool(scaling_config["enabled"])
    scaling_confidence = float(scaling_config["confidence_level"])
    scaling_label = f"{scaling_confidence:.0%}"
    if scaling_confidence not in calculator.confidence_levels:
        raise ValueError("风险缩放置信度必须包含在 confidence_levels 中")
    var_budget = float(scaling_config["var_budgets"][frequency])
    weights_dates = pd.to_datetime(weights_history["signal_date"]).dt.normalize()
    summary_dates = pd.to_datetime(portfolio_summary["signal_date"]).dt.normalize()
    covariance_dates = pd.to_datetime(covariance_history["signal_date"]).dt.normalize()
    risk_rows: list[dict] = []
    for number, date_value in enumerate(dates, 1):
        signal_date = pd.Timestamp(date_value).normalize()
        print(
            f"[{frequency} {number}/{len(dates)}] 计算 {horizon_days}日风险 "
            f"{signal_date:%Y-%m-%d}",
            flush=True,
        )
        weights = weights_history.loc[weights_dates.eq(signal_date)].copy()
        summary = portfolio_summary.loc[summary_dates.eq(signal_date)]
        if len(summary) != 1:
            raise ValueError(f"{signal_date.date()} 必须只有一行组合汇总")
        market_exposure = float(summary.iloc[0]["stock_exposure"])
        cash_weight = float(summary.iloc[0]["cash_weight"])
        if abs(market_exposure + cash_weight - 1.0) > tolerance:
            raise ValueError(f"{signal_date.date()} 股票与现金权重和不为1")
        covariance = covariance_history.loc[
            covariance_dates.eq(signal_date),
            ["asset_i", "asset_j", source_column],
        ].copy()
        covariance[covariance_column] = (
            covariance[source_column] * horizon_days / source_horizon
        )
        covariance = covariance[["asset_i", "asset_j", covariance_column]]
        weight_vector, matrix, input_validation = validator.validate_inputs(
            weights=weights,
            covariance=covariance,
            signal_date=signal_date,
            market_exposure=market_exposure,
            covariance_column=covariance_column,
            horizon_days=horizon_days,
        )
        measurement = calculator.calculate(weight_vector, matrix)
        result_validation = validator.validate_results(measurement)
        metrics_95 = measurement["metrics"]["95%"]
        metrics_99 = measurement["metrics"]["99%"]
        scaling_var = float(
            measurement["metrics"][scaling_label]["value_at_risk"]
        )
        risk_scale = (
            calculate_var_scale(scaling_var, var_budget)
            if scaling_enabled
            else 1.0
        )
        scaled_stock_exposure = market_exposure * risk_scale
        risk_rows.append(
            {
                "signal_date": signal_date,
                "rebalance_frequency": frequency,
                "horizon_days": horizon_days,
                "stock_exposure": market_exposure,
                "cash_weight": cash_weight,
                "risk_scaling_enabled": scaling_enabled,
                "risk_scaling_confidence": scaling_confidence,
                "var_budget": var_budget,
                "risk_scale": risk_scale,
                "scaled_stock_exposure": scaled_stock_exposure,
                "scaled_cash_weight": 1.0 - scaled_stock_exposure,
                f"scaled_var_{scaling_label.replace('%', '')}_{horizon_days}d": (
                    scaling_var * risk_scale
                ),
                "active_stock_count": int(
                    weights["target_weight"].gt(tolerance).sum()
                ),
                f"portfolio_variance_{horizon_days}d": float(
                    measurement["portfolio_variance"]
                ),
                f"portfolio_volatility_{horizon_days}d": float(
                    measurement["portfolio_volatility"]
                ),
                f"portfolio_volatility_{horizon_days}d_pct": float(
                    measurement["portfolio_volatility_pct"]
                ),
                f"var_95_{horizon_days}d": float(metrics_95["value_at_risk"]),
                f"var_95_{horizon_days}d_pct": float(metrics_95["value_at_risk_pct"]),
                f"es_95_{horizon_days}d": float(metrics_95["expected_shortfall"]),
                f"es_95_{horizon_days}d_pct": float(metrics_95["expected_shortfall_pct"]),
                f"var_99_{horizon_days}d": float(metrics_99["value_at_risk"]),
                f"var_99_{horizon_days}d_pct": float(metrics_99["value_at_risk_pct"]),
                f"es_99_{horizon_days}d": float(metrics_99["expected_shortfall"]),
                f"es_99_{horizon_days}d_pct": float(metrics_99["expected_shortfall_pct"]),
                f"min_eigenvalue_{horizon_days}d": float(
                    input_validation[f"min_eigenvalue_{horizon_days}d"]
                ),
                "validation_status": str(result_validation["status"]),
            }
        )

    risk_history = pd.DataFrame(risk_rows)
    if risk_history["signal_date"].duplicated().any():
        raise ValueError("风险历史存在重复 signal_date")
    if not risk_history["horizon_days"].eq(horizon_days).all():
        raise ValueError("风险历史包含错误期限")
    if not risk_history["validation_status"].eq("passed").all():
        raise ValueError("风险历史存在未通过校验的周期")
    if not risk_history["risk_scale"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("风险缩放系数超出[0,1]")
    if (
        risk_history["scaled_stock_exposure"]
        > risk_history["stock_exposure"] + tolerance
    ).any():
        raise ValueError("风险缩放增加了股票仓位")
    scaled_var_column = (
        f"scaled_var_{scaling_label.replace('%', '')}_{horizon_days}d"
    )
    if scaling_enabled and (
        risk_history[scaled_var_column] > var_budget + tolerance
    ).any():
        raise ValueError("缩放后 VaR 仍超过预算")

    releases_root.mkdir(parents=True, exist_ok=True)
    temp_dir = releases_root / f".{release_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    risk_history.to_parquet(temp_dir / "risk.parquet", index=False)
    (temp_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    file_names = ["risk.parquet", "config.yaml"]
    manifest = {
        "schema_version": "risk_history_v2",
        "release_id": release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "rebalance_frequency": frequency,
        "horizon_days": horizon_days,
        "source_portfolio_release": portfolio_release,
        "source_portfolio_manifest_sha256": sha256(portfolio_dir / "manifest.json"),
        "source_decision_release": decision_release,
        "config": config,
        "row_counts": {"risk": int(len(risk_history))},
        "validation": {"status": "passed", "period_count": int(len(dates))},
        "files": {name: sha256(temp_dir / name) for name in file_names},
    }
    write_json(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, release_dir)

    current = {
        "schema_version": "risk_history_current_v2",
        "release_id": release_id,
        "manifest": f"releases/{release_id}/manifest.json",
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "rebalance_frequency": frequency,
        "horizon_days": horizon_days,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    track_output.mkdir(parents=True, exist_ok=True)
    current_temp = track_output / ".current.json.tmp"
    write_json(current_temp, current)
    os.replace(current_temp, track_output / "current.json")
    print(json.dumps(current, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    if config.get("schema_version") != "portfolio_risk_config_v2":
        raise ValueError("配置 schema_version 不是 portfolio_risk_config_v2")
    risk_config = config["risk"]
    if risk_config["method"] != "parametric_normal":
        raise ValueError("仅支持 parametric_normal")
    scaling = risk_config.get("scaling")
    if not isinstance(scaling, dict):
        raise ValueError("风险配置缺少 scaling")
    if set(scaling.get("var_budgets", {})) != set(FREQUENCIES):
        raise ValueError("VaR 预算必须同时定义 weekly 和 monthly")
    frequencies = FREQUENCIES if args.frequency == "both" else (args.frequency,)
    if (args.portfolio_release or args.release_id) and len(frequencies) != 1:
        raise ValueError("显式版本参数必须与单一 --frequency 同时使用")
    horizons = {name: int(value) for name, value in risk_config["horizons"].items()}
    if horizons != {"weekly": 5, "monthly": 20}:
        raise ValueError("风险期限必须是 weekly=5、monthly=20")
    portfolio_base = resolve_path(config["paths"]["portfolio_output_root"])
    decision_output = resolve_path(config["paths"]["decision_output_root"])
    output_root = resolve_path(config["paths"]["output_root"])
    for frequency in frequencies:
        assess_track(
            frequency=frequency,
            horizon_days=horizons[frequency],
            portfolio_base=portfolio_base,
            decision_output=decision_output,
            output_root=output_root,
            portfolio_release_arg=args.portfolio_release,
            release_id_arg=args.release_id,
            config=config,
        )


if __name__ == "__main__":
    main()
