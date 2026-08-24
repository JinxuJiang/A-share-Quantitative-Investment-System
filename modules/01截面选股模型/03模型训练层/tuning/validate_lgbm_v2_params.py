# -*- coding: utf-8 -*-
"""
LightGBM 参数稳健性验证脚本：Quarterly PIT V2
==========================================

用途：
    对 `tune_lgbm_v2.py` 输出的 top 参数做二次验证，防止选到偶然最优参数。

    本脚本会做三件事：
    - 从 `top_trials.csv` 读取前 top_k 组候选参数；
    - 在不与调参季度重合的验证季度上重新评估；
    - 对每组候选参数生成轻微扰动版本，检查附近参数是否也稳定；
    - 加入 baseline 配置同场对比；
    - 输出一个最终推荐配置 `final_recommended_config.yaml`。

默认验证方式：
    - 排除调参季度；
    - 固定 random_seed=42；
    - 默认使用固定非重合验证季度：2021Q1, 2023Q1, 2024Q4, 2026Q2；
    - 默认验证 top 3 组参数；
    - 每组参数生成 8 个邻近扰动版本。

常用命令：
    cd C:\\Users\\蒋大王\\Desktop\\量化\\截面多因子模型\\03模型训练层

    conda run -n qf python tuning\\validate_lgbm_v2_params.py `
      --study-name lgbm20_v2_001 `
      --top-k 3

    # 指定验证季度
    conda run -n qf python tuning\validate_lgbm_v2_params.py `
       --study-name lgbm5_v2_profit20_002 `
       --base-config configs\horizon5_config.yaml `
       --top-k 3 `
       --validate-periods 2021Q1 2023Q1 2024Q4 2026Q2

参数说明：
    --study-name             调参结果目录名，对应 tuning_results/{study_name}
    --base-config            baseline 配置文件，默认 configs/horizon20_config.yaml
    --top-k                  验证 top 几组候选参数，默认 3
    --n-validate-periods     随机抽取几个非重合验证季度，默认 3
    --validate-periods       手动指定验证季度；指定后不再随机抽取
    --tune-periods           调参季度，用于验证时排除
    --random-seed            验证季度随机种子，默认 42
    --keep-cache             保留临时缓存；默认验证完成后删除缓存

输出：
    03模型训练层/tuning/tuning_results/{study_name}/
        validation_results.csv         # baseline、原参数、扰动参数的验证明细
        final_recommended_config.yaml  # 最终推荐配置，可用于完整训练

最终训练示例：
    conda run -n qf python main_train_v2.py `
      --config tuning\\tuning_results\\lgbm20_v2_001\\final_recommended_config.yaml `
      --exp-id exp_20d_optuna_v1 `
      --start-date 2020-01-01 `
      --end-date 2026-06-01 `
      -y

注意：
    1. 本脚本会真实训练模型。
    2. 如果没有候选参数的 robust_score 超过 baseline，会推荐 baseline 配置。
    3. 验证脚本输出的最终配置不会覆盖 configs 目录。
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_constructor_v1 import DataConstructorV1
from dataset.quarterly_splitter_v2 import QuarterlySplitterV2, QuarterlySplit
from models.lightgbm_model import LightGBMModel
from models.lightgbm_rank_model import LightGBMRankModel


DEFAULT_TUNE_PERIODS = ["2020Q2", "2021Q4", "2022Q2", "2022Q4", "2024Q1", "2025Q4"]
DEFAULT_VALIDATE_PERIODS = ["2021Q1", "2023Q1", "2024Q4", "2026Q2"]
PARAM_COLUMNS = [
    "max_depth",
    "num_leaves",
    "min_data_in_leaf",
    "learning_rate",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "lambda_l1",
    "lambda_l2",
    "min_gain_to_split",
]

PARAM_DEFAULTS = {
    "max_depth": -1,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate robust LightGBM V2 tuning candidates")
    parser.add_argument("--study-name", required=True, help="Study directory under tuning_results")
    parser.add_argument("--base-config", default="configs/horizon20_config.yaml", help="Base config path")
    parser.add_argument("--top-k", type=int, default=3, help="Top candidates to validate")
    parser.add_argument("--n-validate-periods", type=int, default=3, help="Random non-overlap validation quarters when --validate-periods is empty")
    parser.add_argument("--validate-periods", nargs="+", default=DEFAULT_VALIDATE_PERIODS, help="Explicit validation quarters")
    parser.add_argument("--tune-periods", nargs="+", default=DEFAULT_TUNE_PERIODS, help="Tune quarters to exclude")
    parser.add_argument("--random-seed", type=int, default=42, help="Validation quarter random seed")
    parser.add_argument("--keep-cache", action="store_true", help="Keep validation cache after completion")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: Path) -> Dict:
    if not config_path.is_absolute():
        config_path = TRAINING_DIR / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return apply_defaults(config)


def apply_defaults(config: Dict) -> Dict:
    config.setdefault("data", {})
    config["data"].setdefault("open_column", "open")
    config["data"].setdefault("st_status_path", "01数据/data/tushare_data/st_status.parquet")
    config.setdefault("training", {})
    config["training"].setdefault("save_models", False)
    config["training"].setdefault("save_feature_importance", False)
    config.setdefault("quarterly_v2", {})
    config["quarterly_v2"].setdefault("train_window", config.get("walk_forward", {}).get("train_window", "3Y"))
    config["quarterly_v2"].setdefault("valid_window", "2M")
    horizon = config["data"].get("label", {}).get("horizon", 20)
    config["quarterly_v2"].setdefault("gap", horizon + 1)
    config["quarterly_v2"].setdefault("smooth_halflife", 10)
    resolve_project_paths(config)
    return config


def resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def resolve_project_paths(config: Dict):
    data = config.get("data", {})
    if "factor_paths" in data:
        for key, value in list(data["factor_paths"].items()):
            data["factor_paths"][key] = resolve_path(value)
    for key in ["market_data_path", "st_status_path", "net_profit_path"]:
        if key in data and data[key]:
            data[key] = resolve_path(data[key])
    output = config.get("output", {})
    if output.get("experiments_dir"):
        output["experiments_dir"] = resolve_path(output["experiments_dir"])


def make_splitter(config: Dict) -> QuarterlySplitterV2:
    dc = DataConstructorV1(config)
    close_df = dc._load_close_data()
    wf_config = config.get("walk_forward", {})
    qv2 = config.get("quarterly_v2", {})
    horizon = config["data"]["label"].get("horizon", 20)
    return QuarterlySplitterV2(
        dates=close_df.index.tolist(),
        train_window=qv2.get("train_window", wf_config.get("train_window", "3Y")),
        valid_window=qv2.get("valid_window", "2M"),
        label_horizon=horizon,
        gap=qv2.get("gap", horizon + 1),
        start_date=wf_config.get("start_date"),
        end_date=wf_config.get("end_date"),
    )


def is_evaluable(splitter: QuarterlySplitterV2, split: QuarterlySplit) -> bool:
    horizon = splitter.label_horizon
    all_dates = splitter.dates
    try:
        idx = all_dates.get_loc(split.period_end)
    except KeyError:
        return False
    return idx + horizon + 1 < len(all_dates)


def choose_validation_splits(splitter: QuarterlySplitterV2, args) -> List[QuarterlySplit]:
    all_splits = list(splitter.get_splits())
    if args.validate_periods:
        requested = set(args.validate_periods)
        splits = [s for s in all_splits if s.model_period in requested]
        missing = sorted(requested - {s.model_period for s in splits})
        if missing:
            raise ValueError(f"Requested validation periods not available: {missing}")
        return sorted(splits, key=lambda s: s.period_start)

    tune_periods = set(args.tune_periods)
    min_tune_year = min(int(p[:4]) for p in tune_periods)
    candidates = [
        s
        for s in all_splits
        if s.model_period not in tune_periods
        and s.period_start.year >= min_tune_year
        and is_evaluable(splitter, s)
    ]
    if len(candidates) < args.n_validate_periods:
        raise ValueError(f"Only {len(candidates)} validation candidates available")

    rng = random.Random(args.random_seed)
    by_year: Dict[int, List[QuarterlySplit]] = {}
    for split in candidates:
        by_year.setdefault(split.period_start.year, []).append(split)

    picked: List[QuarterlySplit] = []
    years = sorted(by_year)
    rng.shuffle(years)
    for year in years:
        group = by_year[year]
        picked.append(rng.choice(group))
        if len(picked) >= args.n_validate_periods:
            break
    return sorted(picked, key=lambda s: s.period_start)


def ensure_period_cache(config: Dict, split: QuarterlySplit, cache_root: Path) -> Path:
    period_dir = cache_root / split.model_period
    done_path = period_dir / "_DONE"
    if done_path.exists():
        return period_dir

    logging.info("Building validation cache for %s", split.model_period)
    period_dir.mkdir(parents=True, exist_ok=True)
    dc = DataConstructorV1(config)
    X_train, y_train = dc.build(split.train_dates, apply_profit_filter=True)
    X_valid, y_valid = dc.build(split.valid_dates, apply_profit_filter=False)
    X_pred = dc.build_for_prediction(split.pred_dates)
    labels = dc._compute_labels(split.pred_dates)
    actual = labels.stack(dropna=False).rename("actual_return")
    actual.index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), s) for d, s in actual.index],
        names=["date", "stock_code"],
    )
    actual = actual.reindex(X_pred.index)

    X_train.to_parquet(period_dir / "X_train.parquet")
    y_train.to_frame("target").to_parquet(period_dir / "y_train.parquet")
    X_valid.to_parquet(period_dir / "X_valid.parquet")
    y_valid.to_frame("target").to_parquet(period_dir / "y_valid.parquet")
    X_pred.to_parquet(period_dir / "X_pred.parquet")
    actual.to_frame("actual_return").to_parquet(period_dir / "actual_return.parquet")
    done_path.write_text("ok\n", encoding="utf-8")
    return period_dir


def load_period_cache(period_dir: Path):
    X_train = pd.read_parquet(period_dir / "X_train.parquet")
    y_train = pd.read_parquet(period_dir / "y_train.parquet")["target"]
    X_valid = pd.read_parquet(period_dir / "X_valid.parquet")
    y_valid = pd.read_parquet(period_dir / "y_valid.parquet")["target"]
    X_pred = pd.read_parquet(period_dir / "X_pred.parquet")
    actual = pd.read_parquet(period_dir / "actual_return.parquet")["actual_return"]
    return X_train, y_train, X_valid, y_valid, X_pred, actual


def create_model(config: Dict):
    model_name = config["model"].get("name", "lightgbm")
    if model_name == "lightgbm_rank":
        return LightGBMRankModel(config)
    return LightGBMModel(config)


def calc_prediction_metrics(pred_score: np.ndarray, X_pred: pd.DataFrame, actual: pd.Series) -> Dict:
    df = pd.DataFrame(
        {
            "date": X_pred.index.get_level_values(0),
            "pred_score": pred_score,
            "actual_return": actual.reindex(X_pred.index).values,
        }
    )
    eval_df = df[df["actual_return"].notna()].copy()
    daily_ics = []
    for _, day_df in eval_df.groupby("date"):
        if len(day_df) >= 10:
            corr = day_df["pred_score"].corr(day_df["actual_return"], method="spearman")
            if not np.isnan(corr):
                daily_ics.append(corr)
    return {
        "n_eval_samples": int(len(eval_df)),
        "rank_ic": float(eval_df["pred_score"].corr(eval_df["actual_return"], method="spearman"))
        if len(eval_df) >= 10
        else np.nan,
        "daily_rank_ic_mean": float(np.mean(daily_ics)) if daily_ics else np.nan,
        "daily_rank_ic_std": float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else np.nan,
    }


def score_from_period_metrics(period_metrics: List[Dict], params: Dict) -> Dict:
    period_ics = np.array(
        [m["daily_rank_ic_mean"] for m in period_metrics if not pd.isna(m.get("daily_rank_ic_mean"))],
        dtype=float,
    )
    if len(period_ics) == 0:
        return {
            "score": -999.0,
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "negative_quarter_ratio": np.nan,
            "worst_quarter_ic": np.nan,
            "complexity_penalty": np.nan,
        }
    mean_ic = float(np.mean(period_ics))
    std_ic = float(np.std(period_ics, ddof=1)) if len(period_ics) > 1 else 0.0
    negative_ratio = float(np.mean(period_ics < 0))
    worst_ic = float(np.min(period_ics))
    complexity_penalty = (
        0.00002 * float(params.get("num_leaves", 31))
        + 0.00001 * max(0.0, 200.0 - float(params.get("min_data_in_leaf", 200)))
    )
    score = mean_ic - 0.5 * std_ic - 0.03 * negative_ratio + 0.2 * min(worst_ic, 0.0) - complexity_penalty
    return {
        "score": float(score),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "negative_quarter_ratio": negative_ratio,
        "worst_quarter_ic": worst_ic,
        "complexity_penalty": float(complexity_penalty),
    }


def config_with_params(base_config: Dict, params: Dict) -> Dict:
    config = copy.deepcopy(base_config)
    config["model"]["params"].update(params)
    config.setdefault("training", {})
    config["training"]["save_models"] = False
    config["training"]["save_feature_importance"] = False
    return config


def run_candidate(candidate_id: str, params: Dict, base_config: Dict, period_dirs: List[Path]) -> Dict:
    config = config_with_params(base_config, params)
    period_metrics = []
    start = time.time()
    for period_dir in period_dirs:
        X_train, y_train, X_valid, y_valid, X_pred, actual = load_period_cache(period_dir)
        model = create_model(config)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred_score = model.predict(X_pred)
        metrics = calc_prediction_metrics(pred_score, X_pred, actual)
        metrics["model_period"] = period_dir.name
        period_metrics.append(metrics)
    score_metrics = score_from_period_metrics(period_metrics, params)
    out = {"candidate_id": candidate_id, "elapsed_sec": time.time() - start}
    out.update(score_metrics)
    for metrics in period_metrics:
        period = metrics["model_period"]
        out[f"{period}_daily_rank_ic_mean"] = metrics["daily_rank_ic_mean"]
        out[f"{period}_rank_ic"] = metrics["rank_ic"]
    return out


def clamp(value, low, high):
    return max(low, min(high, value))


def valid_num_leaves(value: int, max_depth: int) -> int:
    value = int(round(value))
    value = max(7, value)
    if max_depth > 0:
        value = min(value, 2**max_depth)
    return int(value)


def generate_neighbors(params: Dict) -> List[Dict]:
    neighbors = []
    specs = [
        ("min_data_in_leaf_down", {"min_data_in_leaf": max(50, int(params["min_data_in_leaf"] * 0.8))}),
        ("min_data_in_leaf_up", {"min_data_in_leaf": int(params["min_data_in_leaf"] * 1.2)}),
        ("lambda_l2_down", {"lambda_l2": max(0.0, float(params["lambda_l2"]) * 0.7)}),
        ("lambda_l2_up", {"lambda_l2": float(params["lambda_l2"]) * 1.3}),
        ("feature_fraction_down", {"feature_fraction": clamp(float(params["feature_fraction"]) - 0.1, 0.3, 1.0)}),
        ("feature_fraction_up", {"feature_fraction": clamp(float(params["feature_fraction"]) + 0.1, 0.3, 1.0)}),
        ("learning_rate_down", {"learning_rate": max(0.001, float(params["learning_rate"]) * 0.8)}),
        ("learning_rate_up", {"learning_rate": min(0.1, float(params["learning_rate"]) * 1.2)}),
    ]
    for name, patch in specs:
        neighbor = copy.deepcopy(params)
        neighbor.update(patch)
        neighbor["num_leaves"] = valid_num_leaves(int(neighbor["num_leaves"]), int(neighbor["max_depth"]))
        neighbors.append({"variant": name, "params": neighbor})
    return neighbors


def extract_params(row: pd.Series) -> Dict:
    params = {}
    for col in PARAM_COLUMNS:
        value = row[col]
        if col in {"max_depth", "num_leaves", "min_data_in_leaf", "bagging_freq"}:
            params[col] = int(round(float(value)))
        else:
            params[col] = float(value)
    return params


def to_builtin(value):
    """Convert numpy/pandas scalar containers to plain YAML-safe Python types."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if pd.isna(value) if isinstance(value, float) else False:
        return None
    return value


def aggregate_robustness(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_id, group in results[results["source_id"] != "baseline"].groupby("source_id"):
        original = group[group["variant"] == "original"]
        neighbors = group[group["variant"] != "original"]
        if original.empty:
            continue
        original_score = float(original.iloc[0]["score"])
        neighbor_scores = neighbors["score"].dropna()
        neighbor_mean = float(neighbor_scores.mean()) if len(neighbor_scores) else np.nan
        neighbor_std = float(neighbor_scores.std(ddof=1)) if len(neighbor_scores) > 1 else 0.0
        neighbor_min = float(neighbor_scores.min()) if len(neighbor_scores) else np.nan
        drop = original_score - neighbor_mean if not pd.isna(neighbor_mean) else np.nan
        robust_score = 0.5 * original_score + 0.5 * neighbor_mean - neighbor_std - 0.25 * max(0.0, drop)
        row = {
            "source_id": source_id,
            "original_score": original_score,
            "neighbor_mean_score": neighbor_mean,
            "neighbor_std_score": neighbor_std,
            "neighbor_min_score": neighbor_min,
            "score_drop_from_original": drop,
            "robust_score": robust_score,
        }
        for col in PARAM_COLUMNS:
            row[col] = original.iloc[0][col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("robust_score", ascending=False, na_position="last")


def main():
    args = parse_args()
    setup_logging()
    base_config = load_config(Path(args.base_config))
    study_dir = SCRIPT_DIR / "tuning_results" / args.study_name
    top_path = study_dir / "top_trials.csv"
    if not top_path.exists():
        raise FileNotFoundError(f"top_trials.csv not found: {top_path}")

    cache_root = SCRIPT_DIR / ".cache" / f"{args.study_name}_validate"
    cache_root.mkdir(parents=True, exist_ok=True)

    splitter = make_splitter(base_config)
    validation_splits = choose_validation_splits(splitter, args)
    validation_periods = [s.model_period for s in validation_splits]
    logging.info("Validation periods: %s", validation_periods)
    period_dirs = [ensure_period_cache(base_config, split, cache_root) for split in validation_splits]

    top_trials = pd.read_csv(top_path).head(args.top_k)
    rows = []

    baseline_params = {col: base_config["model"]["params"].get(col, PARAM_DEFAULTS[col]) for col in PARAM_COLUMNS}
    baseline_metrics = run_candidate("baseline", baseline_params, base_config, period_dirs)
    baseline_row = {
        "source_id": "baseline",
        "variant": "baseline",
        "candidate_id": "baseline",
        **baseline_metrics,
        **baseline_params,
    }
    rows.append(baseline_row)

    for _, trial_row in top_trials.iterrows():
        source_id = f"trial_{int(trial_row['trial_id']):03d}"
        params = extract_params(trial_row)
        original_metrics = run_candidate(f"{source_id}_original", params, base_config, period_dirs)
        rows.append(
            {
                "source_id": source_id,
                "variant": "original",
                "candidate_id": f"{source_id}_original",
                "tune_score": trial_row.get("score", np.nan),
                **original_metrics,
                **params,
            }
        )
        for neighbor in generate_neighbors(params):
            metrics = run_candidate(f"{source_id}_{neighbor['variant']}", neighbor["params"], base_config, period_dirs)
            rows.append(
                {
                    "source_id": source_id,
                    "variant": neighbor["variant"],
                    "candidate_id": f"{source_id}_{neighbor['variant']}",
                    "tune_score": trial_row.get("score", np.nan),
                    **metrics,
                    **neighbor["params"],
                }
            )

    results = pd.DataFrame(rows)
    robustness = aggregate_robustness(results)
    baseline_score = float(results.loc[results["source_id"] == "baseline", "score"].iloc[0])
    robustness["baseline_score"] = baseline_score
    robustness["beats_baseline"] = robustness["robust_score"] > baseline_score

    results = results.merge(
        robustness[["source_id", "robust_score", "neighbor_mean_score", "neighbor_std_score", "neighbor_min_score"]],
        on="source_id",
        how="left",
    )
    results["validation_periods"] = ",".join(validation_periods)
    results.to_csv(study_dir / "validation_results.csv", index=False, encoding="utf-8-sig")

    if len(robustness) == 0:
        raise RuntimeError("No robust candidates found")
    recommended = robustness.iloc[0]
    recommended_source = recommended["source_id"]
    if not bool(recommended["beats_baseline"]):
        logging.warning("No candidate robust_score beat baseline; recommending baseline config")
        final_config = copy.deepcopy(base_config)
        recommended_source = "baseline"
    else:
        final_config = copy.deepcopy(base_config)
        final_params = {col: recommended[col] for col in PARAM_COLUMNS if col in recommended and not pd.isna(recommended[col])}
        for col in {"max_depth", "num_leaves", "min_data_in_leaf", "bagging_freq"}:
            if col in final_params:
                final_params[col] = int(round(float(final_params[col])))
        final_config["model"]["params"].update(final_params)

    final_config.setdefault("tuning_metadata", {})
    final_config["tuning_metadata"].update(
        {
            "study_name": args.study_name,
            "recommended_source": recommended_source,
            "validation_periods": validation_periods,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    with open(study_dir / "final_recommended_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(final_config), f, allow_unicode=True, sort_keys=False)

    logging.info("Validation outputs written to %s", study_dir)
    logging.info("Recommended source: %s", recommended_source)

    if not args.keep_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
        logging.info("Removed cache: %s", cache_root)


if __name__ == "__main__":
    main()
