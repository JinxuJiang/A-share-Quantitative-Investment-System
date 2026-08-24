from __future__ import annotations

"""逐周计算全部历史组合的5日正态参数法VaR/ES。"""

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

from risk.parametric_var_es import ParametricNormalVarEs
from validation.risk_validator import RiskValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐周测量全部历史组合的5日VaR/ES")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--portfolio-release", help="默认读取02层current历史版本")
    parser.add_argument("--release-id")
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (LAYER_ROOT / path).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def load_portfolio_history(output_root: Path, release_id: str | None) -> tuple[Path, dict]:
    if release_id is None:
        current = read_json(output_root / "current.json")
        if current.get("schema_version") != "portfolio_history_current_v1":
            raise ValueError("02层current不是完整历史版本，请先重建02层")
        release_id = str(current["release_id"])
    release_dir = output_root / "releases" / release_id
    manifest = read_json(release_dir / "manifest.json")
    if manifest.get("schema_version") != "portfolio_history_v1":
        raise ValueError("02层manifest不是portfolio_history_v1")
    verify_files(release_dir, manifest)
    return release_dir, manifest


def existing_release_is_current(
    release_dir: Path,
    output_root: Path,
    release_id: str,
    source_release: str,
    config: dict,
) -> bool:
    if not release_dir.exists():
        return False
    manifest = read_json(release_dir / "manifest.json")
    current = read_json(output_root / "current.json")
    same = (
        manifest.get("schema_version") == "risk_history_v1"
        and manifest.get("source_portfolio_release") == source_release
        and manifest.get("config") == config
        and current.get("release_id") == release_id
    )
    if not same:
        raise FileExistsError(f"同名历史风控发布的输入或配置不同: {release_dir}")
    verify_files(release_dir, manifest)
    print(json.dumps({"status": "already_current", "release_id": release_id}, ensure_ascii=False))
    return True


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    if config.get("schema_version") != "portfolio_risk_config_v1":
        raise ValueError("配置schema_version不是portfolio_risk_config_v1")
    risk_config = config["risk"]
    if risk_config["method"] != "parametric_normal":
        raise ValueError("第一版只支持parametric_normal")
    horizon_days = int(risk_config["horizon_days"])
    if horizon_days != 5:
        raise ValueError("周度风控正式输出必须是5个交易日")

    portfolio_output = resolve_path(config["paths"]["portfolio_output_root"])
    decision_output = resolve_path(config["paths"]["decision_output_root"])
    output_root = resolve_path(config["paths"]["output_root"])
    portfolio_dir, portfolio_manifest = load_portfolio_history(
        portfolio_output, args.portfolio_release
    )
    portfolio_release = str(portfolio_manifest["release_id"])
    first_date = pd.Timestamp(portfolio_manifest["first_signal_date"])
    last_date = pd.Timestamp(portfolio_manifest["last_signal_date"])
    release_id = args.release_id or f"risk_history_{first_date:%Y%m%d}_{last_date:%Y%m%d}_v1"
    releases_root = output_root / "releases"
    release_dir = releases_root / release_id
    if existing_release_is_current(release_dir, output_root, release_id, portfolio_release, config):
        return

    decision_release = str(portfolio_manifest["source_decision_release"])
    decision_dir = decision_output / "releases" / decision_release
    decision_manifest_path = decision_dir / "manifest.json"
    decision_manifest = read_json(decision_manifest_path)
    if decision_manifest.get("schema_version") != "decision_history_v1":
        raise ValueError("01层manifest不是decision_history_v1")
    if sha256(decision_manifest_path) != portfolio_manifest["source_decision_manifest_sha256"]:
        raise ValueError("02层引用的01层manifest哈希不一致")
    verify_files(decision_dir, decision_manifest)

    weights_history = pd.read_parquet(portfolio_dir / "weights.parquet")
    portfolio_summary = pd.read_parquet(portfolio_dir / "portfolio_summary.parquet")
    covariance_history = pd.read_parquet(decision_dir / "covariance.parquet")
    dates = sorted(pd.to_datetime(portfolio_summary["signal_date"]).dt.normalize().unique())
    if len(dates) != int(portfolio_manifest["period_count"]):
        raise ValueError("02层历史周期数量与manifest不一致")
    source_horizon = int(decision_manifest["horizon_days"])
    source_column = f"covariance_{source_horizon}d"
    covariance_column = "covariance_5d"
    if source_column not in covariance_history.columns:
        raise ValueError(f"01层协方差缺少字段: {source_column}")

    tolerance = float(config["validation"]["weight_tolerance"])
    validator = RiskValidator(
        weight_tolerance=tolerance,
        psd_tolerance=float(config["validation"]["psd_tolerance"]),
    )
    calculator = ParametricNormalVarEs(
        confidence_levels=risk_config["confidence_levels"],
        expected_return=float(risk_config["expected_return"]),
    )
    risk_rows: list[dict] = []
    for number, date_value in enumerate(dates, 1):
        signal_date = pd.Timestamp(date_value).normalize()
        print(f"[{number}/{len(dates)}] 风险测量 {signal_date:%Y-%m-%d}", flush=True)
        weights = weights_history.loc[
            pd.to_datetime(weights_history["signal_date"]).dt.normalize().eq(signal_date)
        ].copy()
        summary = portfolio_summary.loc[
            pd.to_datetime(portfolio_summary["signal_date"]).dt.normalize().eq(signal_date)
        ]
        if len(summary) != 1:
            raise ValueError(f"{signal_date.date()}组合汇总应有且只有一行")
        market_exposure = float(summary.iloc[0]["stock_exposure"])
        cash_weight = float(summary.iloc[0]["cash_weight"])
        if abs(market_exposure + cash_weight - 1.0) > tolerance:
            raise ValueError(f"{signal_date.date()}股票与现金权重和不等于1")
        covariance = covariance_history.loc[
            pd.to_datetime(covariance_history["signal_date"]).dt.normalize().eq(signal_date),
            ["asset_i", "asset_j", source_column],
        ].copy()
        covariance[covariance_column] = covariance[source_column] * horizon_days / source_horizon
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
        risk_rows.append(
            {
                "signal_date": signal_date,
                "horizon_days": horizon_days,
                "stock_exposure": market_exposure,
                "cash_weight": cash_weight,
                "active_stock_count": int(weights["target_weight"].gt(tolerance).sum()),
                "portfolio_variance_5d": float(measurement["portfolio_variance"]),
                "portfolio_volatility_5d": float(measurement["portfolio_volatility"]),
                "portfolio_volatility_5d_pct": float(measurement["portfolio_volatility_pct"]),
                "var_95_5d": float(metrics_95["value_at_risk"]),
                "var_95_5d_pct": float(metrics_95["value_at_risk_pct"]),
                "es_95_5d": float(metrics_95["expected_shortfall"]),
                "es_95_5d_pct": float(metrics_95["expected_shortfall_pct"]),
                "var_99_5d": float(metrics_99["value_at_risk"]),
                "var_99_5d_pct": float(metrics_99["value_at_risk_pct"]),
                "es_99_5d": float(metrics_99["expected_shortfall"]),
                "es_99_5d_pct": float(metrics_99["expected_shortfall_pct"]),
                "min_eigenvalue_5d": float(input_validation["min_eigenvalue_5d"]),
                "validation_status": "passed"
                if result_validation["status"] == "passed"
                else result_validation["status"],
            }
        )

    risk_history = pd.DataFrame(risk_rows)
    if risk_history["signal_date"].duplicated().any():
        raise ValueError("风险历史表存在重复signal_date")
    if not risk_history["horizon_days"].eq(5).all():
        raise ValueError("风险历史表混入了非5日指标")
    if not risk_history["validation_status"].eq("passed").all():
        raise ValueError("风险历史表存在未通过校验的周期")

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
        "schema_version": "risk_history_v1",
        "release_id": release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "horizon_days": 5,
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
        "schema_version": "risk_history_current_v1",
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
