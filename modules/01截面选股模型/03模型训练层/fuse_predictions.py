#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型预测融合脚本 V2：Quarterly PIT 口径
========================================

用途：
    融合多个 V2 单模型输出的 `smoothed_predictions.parquet`。

    V2 不再区分 test/live，也不再使用 split_date：
    - 每个输入模型都是一条 PIT predictions 序列；
    - 每天取所有模型共同股票集合；
    - 每个模型每天做截面 rank 标准化；
    - 用滞后 IC 均值生成动态权重；
    - 输出一条融合后的 `predictions.parquet` 和 `smoothed_predictions.parquet`。

常用命令：
    python fuse_predictions.py --exps lgbm5_tushare_profit20_v2 lgbm20_tushare_profit20_v2 lgbm60_tushare_profit20_v2 --base-idx 1 --output-exp ensemble_5d_20d_60d_profit20_v2

参数说明：
    --exps          输入模型实验ID列表，按顺序排列，索引从0开始
    --base-idx      基准模型索引；决定统一 IC 收益口径和权重滞后天数
                    lag = 基准模型 horizon + 1，用于避免使用尚未兑现的收益计算权重
    --output-exp    融合后实验ID

输入：
    03模型训练层/experiments/{exp_id}/smoothed_predictions.parquet

输出：
    03模型训练层/experiments/{output_exp}/
        predictions.parquet
        smoothed_predictions.parquet
        config.yaml
        fusion_config.yaml

说明：
    输出的 `pred_score` 和 `pred_score_smooth` 都是融合后的 rank 分数。
    `config.yaml` 是回测层和输出层读取的标准实验配置，其中 horizon 取基准模型 horizon。
    `fusion_config.yaml` 保存完整的来源模型、滞后参数和最终权重。
    为了兼容现有回测，推荐继续使用：
        python backtrader.eval.py --exp-id {output_exp} --use-smooth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="多模型预测融合 V2 (Quarterly PIT)")
    parser.add_argument(
        "--exps",
        nargs="+",
        required=True,
        help="多个模型实验ID，如 qv2_5d_full_v2 qv2_20d_full_v2 qv2_60d_full_v2",
    )
    parser.add_argument(
        "--base-idx",
        type=int,
        default=0,
        help=(
            "基准模型索引，决定统一IC收益口径和权重滞后天数 "
            "lag=base_model_horizon+1；推荐20d模型索引"
        ),
    )
    parser.add_argument("--output-exp", required=True, help="输出实验ID，如 qv2_ensemble_5d_20d_60d")
    return parser.parse_args()


def load_model_config(exp_dir: Path) -> Dict:
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    label_config = config.get("data", {}).get("label", {})
    horizon = label_config.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError(f"{config_path} 缺少有效的 data.label.horizon")
    return {
        "horizon": horizon,
        "label_formula": label_config.get("formula", "forward_return"),
        "use_open_price": bool(label_config.get("use_open_price", False)),
    }


def validate_label_timing(models_data: List[Dict]) -> None:
    """融合模型允许 horizon 不同，但标签公式和交易时点必须一致。"""
    signatures = {
        (model["label_formula"], model["use_open_price"])
        for model in models_data
    }
    if len(signatures) > 1:
        details = ", ".join(
            f"{model['exp_id']}="
            f"{model['label_formula']}/use_open_price={model['use_open_price']}"
            for model in models_data
        )
        raise ValueError(f"输入实验的标签口径不一致，禁止融合: {details}")


def load_model_data(exp_dir: Path) -> pd.DataFrame:
    pred_file = exp_dir / "smoothed_predictions.parquet"
    if not pred_file.exists():
        raise FileNotFoundError(f"平滑预测不存在: {pred_file}")

    df = pd.read_parquet(pred_file)
    df["date"] = pd.to_datetime(df["date"])
    if "pred_score_smooth" not in df.columns:
        raise ValueError(f"{pred_file} 缺少 pred_score_smooth 列")
    if "actual_return" not in df.columns:
        df["actual_return"] = np.nan
    if "is_evaluable" not in df.columns:
        df["is_evaluable"] = df["actual_return"].notna()
    return df


def merge_with_intersection(
    dfs: List[pd.DataFrame],
    value_col: str = "pred_score_smooth",
    base_idx: int = 0,
) -> pd.DataFrame:
    """按 date + stock_code 取交集，并使用基准模型的 actual_return 作为统一 IC 口径。"""
    if not dfs:
        raise ValueError("至少需要一个模型进行融合")
    if base_idx < 0 or base_idx >= len(dfs):
        raise ValueError(f"base_idx 超出范围: {base_idx}, 模型数量={len(dfs)}")

    prepared = []
    for i, df in enumerate(dfs):
        keep_cols = ["date", "stock_code", value_col]
        extra_cols = []
        if i == base_idx:
            for col in ["actual_return", "is_evaluable", "model_period", "fold_id"]:
                if col in df.columns:
                    extra_cols.append(col)

        part = df[keep_cols + extra_cols].copy()
        part = part.rename(columns={value_col: f"pred_{i}"})
        prepared.append(part)

    merged = prepared[0]
    for i in range(1, len(prepared)):
        merged = merged.merge(
            prepared[i],
            on=["date", "stock_code"],
            how="inner",
        )
    return merged


def rank_standardize(df: pd.DataFrame, n_models: int) -> pd.DataFrame:
    df = df.copy()
    for i in range(n_models):
        df[f"rank_{i}"] = df.groupby("date")[f"pred_{i}"].rank(pct=True)
    return df


def calc_daily_ic(df: pd.DataFrame, n_models: int) -> pd.DataFrame:
    records = []
    for date, day_df in df.groupby("date", sort=True):
        eval_df = day_df[day_df["actual_return"].notna()].copy()
        record = {"date": date}
        if len(eval_df) < 10:
            for i in range(n_models):
                record[f"ic_{i}"] = np.nan
            records.append(record)
            continue

        actual_rank = eval_df["actual_return"].rank()
        for i in range(n_models):
            pred_rank = eval_df[f"pred_{i}"].rank()
            record[f"ic_{i}"] = pred_rank.corr(actual_rank)
        records.append(record)
    return pd.DataFrame(records)


def calc_lagged_weights(daily_ic: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    第 t 天权重只使用 t-lag 及以前的 IC。
    对 open[T+horizon+1] / open[T+1] 标签，lag 应为 horizon + 1。
    若历史正IC不足，则使用等权。
    """
    n_models = len([c for c in daily_ic.columns if c.startswith("ic_")])
    records = []
    ic_cols = [f"ic_{i}" for i in range(n_models)]

    for t, row in daily_ic.reset_index(drop=True).iterrows():
        if t < lag:
            weights = np.repeat(1.0 / n_models, n_models)
        else:
            hist_ic = daily_ic.iloc[: t - lag + 1][ic_cols].mean(skipna=True).clip(lower=0)
            if hist_ic.notna().any() and hist_ic.sum() > 0:
                weights = (hist_ic / hist_ic.sum()).fillna(0).values
            else:
                weights = np.repeat(1.0 / n_models, n_models)

        rec = {"date": row["date"]}
        for i in range(n_models):
            rec[f"weight_{i}"] = float(weights[i])
        records.append(rec)
    return pd.DataFrame(records)


def fuse_with_weights(df: pd.DataFrame, weights_df: pd.DataFrame, n_models: int) -> pd.DataFrame:
    df = df.merge(weights_df, on="date", how="left").copy()
    df["rank_fused"] = 0.0
    for i in range(n_models):
        df["rank_fused"] += df[f"weight_{i}"] * df[f"rank_{i}"]

    output_cols = ["date", "stock_code", "rank_fused", "actual_return"]
    for col in ["is_evaluable", "model_period", "fold_id"]:
        if col in df.columns:
            output_cols.append(col)

    out = df[output_cols].copy()
    out = out.rename(columns={"rank_fused": "pred_score_smooth"})
    out["pred_score"] = out["pred_score_smooth"]
    if "is_evaluable" not in out.columns:
        out["is_evaluable"] = out["actual_return"].notna()
    if "fold_id" not in out.columns:
        out["fold_id"] = 0

    ordered_cols = ["date", "stock_code", "pred_score", "pred_score_smooth", "actual_return", "is_evaluable", "fold_id"]
    if "model_period" in out.columns:
        ordered_cols.append("model_period")
    return out[ordered_cols].sort_values(["date", "stock_code"]).reset_index(drop=True)


def save_fusion_configs(
    output_dir: Path,
    models_data: List[Dict],
    lag: int,
    final_weights: pd.Series,
    base_idx: int,
):
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    fusion_config = {
        "fusion_info": {
            "mode": "quarterly_pit_v2",
            "n_models": len(models_data),
            "base_model": models_data[base_idx]["exp_id"],
            "base_model_index": base_idx,
            "ic_lag": lag,
            "timestamp": timestamp,
        },
        "models": [],
        "final_weights": {},
    }

    for i, model in enumerate(models_data):
        fusion_config["models"].append(
            {
                "index": i,
                "exp_id": model["exp_id"],
                "horizon": model["horizon"],
                "date_start": str(model["df"]["date"].min().date()),
                "date_end": str(model["df"]["date"].max().date()),
                "final_weight": float(final_weights.iloc[i]),
            }
        )
        fusion_config["final_weights"][f"model_{i}"] = float(final_weights.iloc[i])

    fusion_config_path = output_dir / "fusion_config.yaml"
    with open(fusion_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(fusion_config, f, sort_keys=False, allow_unicode=True)

    base_model = models_data[base_idx]
    standard_config = {
        "data": {
            "label": {
                "formula": base_model["label_formula"],
                "horizon": base_model["horizon"],
                "use_open_price": base_model["use_open_price"],
            }
        },
        "model": {
            "name": "weighted_rank_fusion",
            "params": {
                "score_type": "daily_cross_section_rank",
                "weight_method": "lagged_positive_mean_ic",
            },
        },
        "output": {
            "predictions_filename": "predictions.parquet",
        },
        "fusion_metadata": {
            "generated_at": timestamp,
            "config_file": "fusion_config.yaml",
            "base_model": base_model["exp_id"],
            "base_model_index": base_idx,
            "source_experiments": [model["exp_id"] for model in models_data],
            "ic_lag": lag,
        },
    }
    standard_config_path = output_dir / "config.yaml"
    with open(standard_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(standard_config, f, sort_keys=False, allow_unicode=True)

    print(f"\n[标准配置已保存] {standard_config_path}")
    print(f"[融合明细已保存] {fusion_config_path}")


def main():
    args = parse_args()
    if args.base_idx < 0 or args.base_idx >= len(args.exps):
        raise ValueError(f"--base-idx 超出范围: {args.base_idx}, exps数量={len(args.exps)}")

    print("=" * 70)
    print("多模型预测融合 V2 (Quarterly PIT)")
    print("=" * 70)
    print(f"输入模型: {args.exps}")
    print(f"基准模型索引: {args.base_idx}")
    print(f"输出实验ID: {args.output_exp}")

    base_dir = Path(__file__).parent / "experiments"
    output_dir = base_dir / args.output_exp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] 加载模型数据...")
    models_data = []
    for exp_id in args.exps:
        exp_path = base_dir / exp_id
        model_config = load_model_config(exp_path)
        df = load_model_data(exp_path)
        print(
            f"  {exp_id}: horizon={model_config['horizon']}, "
            f"{df['date'].min().date()} ~ {df['date'].max().date()}, rows={len(df)}"
        )
        models_data.append({"exp_id": exp_id, **model_config, "df": df})

    n_models = len(models_data)
    validate_label_timing(models_data)

    print("\n[2/6] 合并数据（date+stock交集）...")
    merged = merge_with_intersection(
        [m["df"] for m in models_data],
        base_idx=args.base_idx,
    )
    print(f"  交集后: rows={len(merged)}, dates={merged['date'].nunique()}")
    print(f"  日期范围: {merged['date'].min().date()} ~ {merged['date'].max().date()}")
    print(f"  IC收益口径: {models_data[args.base_idx]['exp_id']} 的 actual_return")

    print("\n[3/6] 每日截面rank标准化...")
    merged = rank_standardize(merged, n_models)

    print("\n[4/6] 计算每日IC和滞后权重...")
    daily_ic = calc_daily_ic(merged, n_models)
    lag = models_data[args.base_idx]["horizon"] + 1
    weights = calc_lagged_weights(daily_ic, lag=lag)
    for i, model in enumerate(models_data):
        print(f"  Model {i} ({model['exp_id']}): mean IC={daily_ic[f'ic_{i}'].mean():.4f}")
    print(f"  lag={lag} (base model: {models_data[args.base_idx]['exp_id']})")

    print("\n[5/6] 融合预测...")
    fused = fuse_with_weights(merged, weights, n_models)

    pred_file = output_dir / "predictions.parquet"
    smooth_file = output_dir / "smoothed_predictions.parquet"
    fused.to_parquet(pred_file, index=False)
    fused.to_parquet(smooth_file, index=False)
    print(f"  已保存: {pred_file} ({len(fused)}行)")
    print(f"  已保存: {smooth_file} ({len(fused)}行)")

    print("\n[6/6] 保存融合配置...")
    last_weights = weights[[f"weight_{i}" for i in range(n_models)]].iloc[-1]
    save_fusion_configs(output_dir, models_data, lag, last_weights, args.base_idx)

    print("\n" + "=" * 70)
    print("融合完成")
    print(f"输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
