"""逐周优化全部历史组合，并发布完整权重表。"""

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
from validation.optimization_validator import OptimizationValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐周优化全部历史组合")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--decision-release", help="默认读取01层current历史版本")
    parser.add_argument("--release-id")
    return parser.parse_args()


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_decision_history(output_root: Path, release_id: str | None) -> tuple[Path, dict]:
    if release_id is None:
        current = read_json(output_root / "current.json")
        if current.get("schema_version") != "decision_history_current_v1":
            raise ValueError("01层current不是完整历史版本，请先重建01层")
        release_id = str(current["release_id"])
    release_dir = output_root / "releases" / release_id
    manifest = read_json(release_dir / "manifest.json")
    if manifest.get("schema_version") != "decision_history_v1":
        raise ValueError("01层manifest不是decision_history_v1")
    for name, expected_hash in manifest["files"].items():
        path = release_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"01层文件缺失或哈希不一致: {path}")
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
        manifest.get("schema_version") == "portfolio_history_v1"
        and manifest.get("source_decision_release") == source_release
        and manifest.get("config") == config
        and current.get("release_id") == release_id
    )
    if not same:
        raise FileExistsError(f"同名历史优化发布的输入或配置不同: {release_dir}")
    for name, expected_hash in manifest["files"].items():
        path = release_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"历史优化文件缺失或哈希不一致: {path}")
    print(json.dumps({"status": "already_current", "release_id": release_id}, ensure_ascii=False))
    return True


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve_from(LAYER_ROOT, args.config).read_text(encoding="utf-8"))
    if config.get("schema_version") != "portfolio_optimization_config_v2":
        raise ValueError("配置schema_version不是portfolio_optimization_config_v2")
    if config["constraints"].get("weight_basis") != "account":
        raise ValueError("正式组合约束必须基于账户实际权重")
    if config["constraints"].get("market_exposure_mode") != "target_with_cash_fallback":
        raise ValueError("正式市场仓位模式必须允许风控触发额外现金")

    decision_output = resolve_from(LAYER_ROOT, config["paths"]["decision_output_root"])
    output_root = resolve_from(LAYER_ROOT, config["paths"]["output_root"])
    source_dir, source_manifest = load_decision_history(decision_output, args.decision_release)
    source_release = str(source_manifest["release_id"])
    first_date = pd.Timestamp(source_manifest["first_signal_date"])
    last_date = pd.Timestamp(source_manifest["last_signal_date"])
    release_id = args.release_id or f"portfolio_history_{first_date:%Y%m%d}_{last_date:%Y%m%d}_v1"
    releases_root = output_root / "releases"
    release_dir = releases_root / release_id
    if existing_release_is_current(release_dir, output_root, release_id, source_release, config):
        return

    decisions = pd.read_parquet(source_dir / "decision_inputs.parquet")
    covariance_history = pd.read_parquet(source_dir / "covariance.parquet")
    period_summary = pd.read_parquet(source_dir / "period_summary.parquet")
    dates = sorted(pd.to_datetime(decisions["signal_date"]).dt.normalize().unique())
    if len(dates) != int(source_manifest["period_count"]):
        raise ValueError("01层决策日期数量与manifest不一致")

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

    weight_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    previous_target_weights: dict[str, float] = {}
    for number, date_value in enumerate(dates, 1):
        signal_date = pd.Timestamp(date_value).normalize()
        print(f"[{number}/{len(dates)}] 优化 {signal_date:%Y-%m-%d}", flush=True)
        full_alpha = decisions.loc[
            pd.to_datetime(decisions["signal_date"]).dt.normalize().eq(signal_date)
        ].copy()
        full_alpha = full_alpha.sort_values("selection_rank")
        if full_alpha["target_equity_exposure"].nunique() != 1:
            raise ValueError(f"{signal_date.date()}市场仓位不唯一")
        market_exposure = float(full_alpha["target_equity_exposure"].iloc[0])
        alpha = full_alpha.copy()
        codes = alpha["stock_code"].astype(str).tolist()
        current_code_set = set(codes)
        previous_aligned = np.array(
            [previous_target_weights.get(code, 0.0) for code in codes], dtype="float64"
        )
        previous_outside_weight = float(
            sum(
                weight
                for code, weight in previous_target_weights.items()
                if code not in current_code_set
            )
        )
        cov_long = covariance_history.loc[
            pd.to_datetime(covariance_history["signal_date"]).dt.normalize().eq(signal_date)
        ]
        covariance = cov_long.pivot(
            index="asset_i", columns="asset_j", values="covariance_20d"
        ).reindex(index=codes, columns=codes)
        if covariance.isna().any().any():
            raise ValueError(f"{signal_date.date()}协方差与候选股票不一致")
        result = optimizer.solve(
            alpha["alpha_zscore"].to_numpy(dtype="float64"),
            covariance.to_numpy(dtype="float64"),
            alpha["industry"].astype(str).to_numpy(),
            target_exposure=market_exposure,
            previous_account_weights=previous_aligned,
            previous_outside_weight=previous_outside_weight,
        )
        published = result.weights.copy()
        deployed_exposure = float(result.diagnostics["deployed_stock_exposure"])
        candidate_weights = alpha[
            ["signal_date", "stock_code", "industry", "selection_rank", "alpha_zscore"]
        ].copy()
        candidate_weights["base_weight"] = published
        candidate_weights["target_weight"] = published * deployed_exposure
        weights = full_alpha[
            ["signal_date", "stock_code", "industry", "selection_rank", "alpha_zscore"]
        ].copy()
        weights = weights.merge(
            candidate_weights[["stock_code", "base_weight", "target_weight"]],
            on="stock_code",
            how="left",
            validate="one_to_one",
        )
        weights[["base_weight", "target_weight"]] = weights[
            ["base_weight", "target_weight"]
        ].fillna(0.0)
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
            zip(weights["stock_code"].astype(str), weights["target_weight"].astype(float))
        )
        weights = weights.sort_values("optimized_rank").reset_index(drop=True)
        weight_frames.append(weights)
        diagnostics = result.diagnostics
        summary_rows.append(
            {
                "signal_date": signal_date,
                "market_target_exposure": market_exposure,
                "maximum_feasible_exposure": float(diagnostics["maximum_feasible_exposure"]),
                "stock_exposure": deployed_exposure,
                "exposure_shortfall": float(diagnostics["exposure_shortfall"]),
                "cash_weight": float(1.0 - deployed_exposure),
                "candidate_stock_count": int(len(alpha)),
                "active_stock_count": int(weights["is_active"].sum()),
                "predicted_volatility_20d_internal": float(diagnostics["optimized_volatility_20d"]),
                "predicted_volatility_20d_account": float(
                    diagnostics["optimized_volatility_20d"] * deployed_exposure
                ),
                "weighted_alpha_zscore": float(
                    np.dot(weights["alpha_zscore"], weights["base_weight"])
                ),
                "objective_value": float(diagnostics["objective_value"]),
                "estimated_trade_weight": float(diagnostics["estimated_trade_weight"]),
                "turnover_penalty_value": float(diagnostics["turnover_penalty_value"]),
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
        raise ValueError("完整权重表存在重复日期+股票")
    if not all_weights.groupby("signal_date").size().eq(int(source_manifest["top_n"])).all():
        raise ValueError("完整权重表并非每期都包含Top N股票")
    if not portfolio_summary["solver_success"].all():
        raise ValueError("存在未收敛的历史优化期")

    releases_root.mkdir(parents=True, exist_ok=True)
    temp_dir = releases_root / f".{release_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    all_weights.to_parquet(temp_dir / "weights.parquet", index=False)
    portfolio_summary.to_parquet(temp_dir / "portfolio_summary.parquet", index=False)
    (temp_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    file_names = ["weights.parquet", "portfolio_summary.parquet", "config.yaml"]
    manifest = {
        "schema_version": "portfolio_history_v1",
        "release_id": release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "horizon_days": int(source_manifest["horizon_days"]),
        "source_decision_release": source_release,
        "source_decision_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "config": config,
        "row_counts": {
            "weights": int(len(all_weights)),
            "portfolio_summary": int(len(portfolio_summary)),
        },
        "validation": {"status": "passed", "period_count": int(len(dates))},
        "files": {name: sha256_file(temp_dir / name) for name in file_names},
    }
    write_json(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, release_dir)

    current = {
        "schema_version": "portfolio_history_current_v1",
        "release_id": release_id,
        "manifest": f"releases/{release_id}/manifest.json",
        "first_signal_date": first_date.date().isoformat(),
        "last_signal_date": last_date.date().isoformat(),
        "period_count": int(len(dates)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    current_temp = output_root / ".current.json.tmp"
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(current_temp, current)
    os.replace(current_temp, output_root / "current.json")
    print(json.dumps(current, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
