# -*- coding: utf-8 -*-
"""
LightGBM 参数调优脚本：Quarterly PIT V2
====================================

用途：
    在 V2 固定季度训练口径下，为 LightGBM 选择更稳健的超参数。

    本脚本不是完整训练入口，而是调参入口：
    - 固定选取少量代表季度；
    - 每组参数都在这些季度上训练/预测；
    - 用稳定性目标函数打分，而不是只看单季度收益或单个 IC；
    - 缓存选定季度的 X/y 数据，避免每个 trial 反复构造因子数据；
    - 输出 top 参数，供 `validate_lgbm_v2_params.py` 做非重合季度验证。

默认代表季度：
    2020Q2, 2021Q4, 2022Q2, 2022Q4, 2024Q1, 2025Q4

常用命令：
    cd C:\\Users\\蒋大王\\Desktop\\量化\\截面多因子模型\\03模型训练层

    conda run -n qf python tuning\\tune_lgbm_v2.py `
      --base-config configs\\horizon20_config.yaml `
      --study-name lgbm20_v2_001 `
      --n-trials 20

    # 试运行时可减少 trial 数，并降低树数量
    conda run -n qf python tuning\\tune_lgbm_v2.py `
      --base-config configs\\horizon20_config.yaml `
      --study-name lgbm20_v2_smoke `
      --n-trials 2 `
      --trial-n-estimators 200 `
      --trial-early-stopping 20

参数说明：
    --base-config            基准配置文件，默认 configs/horizon20_config.yaml
    --study-name             本次调参名称；输出目录使用该名称
    --n-trials               Optuna 搜索次数，第一版建议 20
    --tune-periods           调参代表季度；默认使用上方 6 个季度
    --top-k                  输出到 top_trials.csv 的候选数量，默认 10
    --trial-n-estimators     可选：调参阶段覆盖 n_estimators
    --trial-early-stopping   可选：调参阶段覆盖 early_stopping_rounds
    --keep-cache             保留临时缓存；默认调参完成后删除缓存
    --random-seed            随机种子，默认 42

输出：
    03模型训练层/tuning/tuning_results/{study_name}/
        study_results.csv     # 所有 trial 结果
        top_trials.csv        # top 候选参数，验证脚本读取它
        tuning_summary.md     # 人看的调参摘要

缓存：
    03模型训练层/tuning/.cache/{study_name}/

注意：
    1. 本脚本需要在 qf 环境中运行。
    2. 本脚本会真实训练模型；不要把 --n-trials 设置得过大后直接运行。
    3. 本脚本不会生成最终配置，最终配置由验证脚本输出。
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def parse_args():
    parser = argparse.ArgumentParser(description="Tune LightGBM params for Quarterly PIT V2")
    parser.add_argument("--base-config", default="configs/horizon20_config.yaml", help="Base config path")
    parser.add_argument("--study-name", default=None, help="Study/output name")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trial count")
    parser.add_argument("--tune-periods", nargs="+", default=DEFAULT_TUNE_PERIODS, help="Representative quarters")
    parser.add_argument("--top-k", type=int, default=10, help="Rows to write to top_trials.csv")
    parser.add_argument("--trial-n-estimators", type=int, default=None, help="Optional trial n_estimators override")
    parser.add_argument("--trial-early-stopping", type=int, default=None, help="Optional early_stopping_rounds override")
    parser.add_argument("--keep-cache", action="store_true", help="Keep tuning cache after completion")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
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
    config.setdefault("output", {})
    config["output"].setdefault("experiments_dir", "03模型训练层/experiments")
    config["output"].setdefault("predictions_filename", "predictions.parquet")
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
    data_constructor = DataConstructorV1(config)
    close_df = data_constructor._load_close_data()
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


def select_splits(splitter: QuarterlySplitterV2, periods: Iterable[str]) -> List[QuarterlySplit]:
    period_set = set(periods)
    splits = [split for split in splitter.get_splits() if split.model_period in period_set]
    found = {split.model_period for split in splits}
    missing = sorted(period_set - found)
    if missing:
        raise ValueError(f"Requested tune periods not available: {missing}")
    return sorted(splits, key=lambda s: s.period_start)


def ensure_period_cache(config: Dict, split: QuarterlySplit, cache_root: Path) -> Path:
    period_dir = cache_root / split.model_period
    done_path = period_dir / "_DONE"
    if done_path.exists():
        return period_dir

    logging.info("Building cache for %s", split.model_period)
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

    if len(X_train) == 0 or len(X_valid) == 0 or len(X_pred) == 0:
        raise ValueError(f"Empty cached dataset for {split.model_period}")

    X_train.to_parquet(period_dir / "X_train.parquet")
    y_train.to_frame("target").to_parquet(period_dir / "y_train.parquet")
    X_valid.to_parquet(period_dir / "X_valid.parquet")
    y_valid.to_frame("target").to_parquet(period_dir / "y_valid.parquet")
    X_pred.to_parquet(period_dir / "X_pred.parquet")
    actual.to_frame("actual_return").to_parquet(period_dir / "actual_return.parquet")

    meta = {
        "model_period": split.model_period,
        "period_start": str(split.period_start.date()),
        "period_end": str(split.period_end.date()),
        "train_start": str(split.train_dates[0].date()),
        "train_end": str(split.train_dates[-1].date()),
        "valid_start": str(split.valid_dates[0].date()),
        "valid_end": str(split.valid_dates[-1].date()),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_pred": int(len(X_pred)),
    }
    with open(period_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.dump(meta, f, allow_unicode=True, sort_keys=False)
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


def suggest_params(trial, base_params: Dict) -> Dict:
    params = copy.deepcopy(base_params)
    max_depth = trial.suggest_int("max_depth", 3, 8)
    raw_num_leaves = trial.suggest_categorical("num_leaves", [7, 15, 31, 63, 95, 127])
    params.update(
        {
            "max_depth": max_depth,
            "num_leaves": min(raw_num_leaves, 2**max_depth),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9, step=0.05),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0, step=0.05),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 20.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        }
    )
    return params


def config_with_params(base_config: Dict, params: Dict, args) -> Dict:
    config = copy.deepcopy(base_config)
    config["model"]["params"].update(params)
    if args.trial_n_estimators is not None:
        config["model"]["params"]["n_estimators"] = args.trial_n_estimators
    if args.trial_early_stopping is not None:
        config["model"]["params"]["early_stopping_rounds"] = args.trial_early_stopping
    config.setdefault("training", {})
    config["training"]["save_models"] = False
    config["training"]["save_feature_importance"] = False
    return config


def run_one_period(config: Dict, period_dir: Path) -> Dict:
    X_train, y_train, X_valid, y_valid, X_pred, actual = load_period_cache(period_dir)
    model = create_model(config)
    model.fit(X_train, y_train, X_valid, y_valid)
    pred_score = model.predict(X_pred)
    metrics = calc_prediction_metrics(pred_score, X_pred, actual)
    metrics["model_period"] = period_dir.name
    return metrics


def write_results(study, out_dir: Path, args, base_config: Dict):
    rows = []
    for t in study.trials:
        row = {
            "trial_id": t.number,
            "state": str(t.state),
            "score": t.value if t.value is not None else np.nan,
        }
        row.update(t.user_attrs)
        for name in PARAM_COLUMNS:
            row[name] = t.user_attrs.get(f"param_{name}", t.params.get(name, np.nan))
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last")
    results.to_csv(out_dir / "study_results.csv", index=False, encoding="utf-8-sig")

    complete = results[results["state"].str.contains("COMPLETE", na=False)].copy()
    top = complete.head(args.top_k)
    top.to_csv(out_dir / "top_trials.csv", index=False, encoding="utf-8-sig")

    best = top.iloc[0].to_dict() if len(top) else {}

    lines = [
        "# LightGBM V2 Tuning Summary",
        "",
        f"- study_name: `{args.study_name}`",
        f"- n_trials: `{args.n_trials}`",
        f"- tune_periods: `{', '.join(args.tune_periods)}`",
        f"- completed_trials: `{len(complete)}`",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Best Trial",
                f"- trial_id: `{int(best['trial_id'])}`",
                f"- score: `{best['score']:.8f}`",
                f"- mean_ic: `{best.get('mean_ic', np.nan):.8f}`",
                f"- std_ic: `{best.get('std_ic', np.nan):.8f}`",
                f"- negative_quarter_ratio: `{best.get('negative_quarter_ratio', np.nan):.4f}`",
                f"- worst_quarter_ic: `{best.get('worst_quarter_ic', np.nan):.8f}`",
                "",
                "## Params",
            ]
        )
        for p in PARAM_COLUMNS:
            lines.append(f"- {p}: `{best.get(p)}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "- `study_results.csv`",
            "- `top_trials.csv`",
            "- `tuning_summary.md`",
        ]
    )
    (out_dir / "tuning_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    setup_logging()
    if args.study_name is None:
        args.study_name = f"lgbm20_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required. Install it in the qf env: pip install optuna") from exc

    np.random.seed(args.random_seed)
    base_config = load_config(Path(args.base_config))
    out_dir = SCRIPT_DIR / "tuning_results" / args.study_name
    cache_root = SCRIPT_DIR / ".cache" / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    splitter = make_splitter(base_config)
    splits = select_splits(splitter, args.tune_periods)
    for split in splits:
        ensure_period_cache(base_config, split, cache_root)

    base_params = copy.deepcopy(base_config["model"]["params"])
    trial_records: List[Tuple[int, Dict]] = []

    def objective(trial):
        params = suggest_params(trial, base_params)
        for name in PARAM_COLUMNS:
            trial.set_user_attr(f"param_{name}", params.get(name))
        config = config_with_params(base_config, params, args)
        period_metrics = []
        start = time.time()
        for idx, split in enumerate(splits):
            metrics = run_one_period(config, cache_root / split.model_period)
            period_metrics.append(metrics)
            interim = score_from_period_metrics(period_metrics, params)["score"]
            trial.report(interim, step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score_metrics = score_from_period_metrics(period_metrics, params)
        for key, value in score_metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("elapsed_sec", time.time() - start)
        for metrics in period_metrics:
            period = metrics["model_period"]
            trial.set_user_attr(f"{period}_daily_rank_ic_mean", metrics["daily_rank_ic_mean"])
            trial.set_user_attr(f"{period}_rank_ic", metrics["rank_ic"])
            trial.set_user_attr(f"{period}_n_eval_samples", metrics["n_eval_samples"])
        trial_records.append((trial.number, score_metrics))
        return score_metrics["score"]

    sampler = optuna.samplers.TPESampler(seed=args.random_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, math.ceil(args.n_trials * 0.2)))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=args.study_name)
    study.optimize(objective, n_trials=args.n_trials)

    write_results(study, out_dir, args, base_config)
    logging.info("Tuning outputs written to %s", out_dir)

    if not args.keep_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
        logging.info("Removed cache: %s", cache_root)


if __name__ == "__main__":
    main()
