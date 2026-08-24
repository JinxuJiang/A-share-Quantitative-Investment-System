# -*- coding: utf-8 -*-
"""
主训练脚本 V2：Quarterly PIT 固定季度模型
========================================

用途：
    V2 保留 V1 的数据构造、标签计算、模型配置和 LightGBM 参数，
    只替换训练/预测切分方式：

    - 每个自然季度训练一个固定模型；
    - 该季度所有交易日都用同一个季度模型预测；
    - 不再生成 live_predictions；
    - predictions.parquet 即 V2 的 PIT 预测主输出；
    - smoothed_predictions.parquet 使用固定 half-life 平滑，默认 10 天。

常用命令：
    # 首次全量训练并建立冻结状态
    python main_train_v2.py --config horizon5_profit20_tuned_config.yaml --exp-id lgbm5_tushare_profit20 --start-date 2020-01-01 --end-date 2026-07-28 --freeze -y

    # 下月继续同一实验：复用冻结模型，只追加缺失预测
    python main_train_v2.py --config horizon5_profit20_tuned_config.yaml --exp-id lgbm5_tushare_profit20 --start-date 2020-01-01 --end-date 2026-08-31 --freeze -y

参数说明：
    --exp-id             实验ID；若不以 _v2 结尾，会自动追加 _v2
    --config             配置文件路径，沿用 V1 的 horizon5/20/60 配置即可
    --start-date         预测开始日期；不会截断训练历史
    --end-date           数据截止日期 / as-of 日期
    --train-window       V2训练窗口，默认沿用配置里的 walk_forward.train_window，通常 3Y
    --valid-window       V2验证窗口，默认 2M
    --gap                训练/验证/预测之间的交易日隔离，默认 horizon + 1
    --smooth-halflife    固定平滑半衰期，默认 10
    --horizon            覆盖配置里的 data.label.horizon
    --freeze             启用冻结/增量模式：复用已有季度模型，只追加缺失预测
    --reset-freeze       重置同一实验的冻结状态，从本次范围重新生成
    -y, --yes            跳过确认提示

输出：
    03模型训练层/experiments/{exp_id}_v2/
        config.yaml
        quarterly_splits.parquet
        predictions.parquet
        smoothed_predictions.parquet
        summary.parquet
        models/
        feature_importance/
        feature_importance_v2.png
        state/freeze_manifest.json
        state/models/model_YYYYQn.pkl

注意：
    V2 当前先支持单模型训练；多周期 ensemble 需要后续使用 V2 融合脚本。
    月度增量更新通常只需要在同一 exp-id 上加 --freeze；
    如果换模型、换参数或重新训练新实验，建议换新的 exp-id，不需要 --reset-freeze。
    --reset-freeze 只用于明确要重建同一 exp-id 的冻结状态。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.quarterly_trainer_v2 import QuarterlyTrainerV2


def parse_args():
    parser = argparse.ArgumentParser(description="截面多因子模型 - Quarterly PIT训练 V2")
    parser.add_argument("--exp-id", "-e", type=str, default=None, help="实验ID；未指定则自动生成")
    parser.add_argument("--config", "-c", type=str, default="configs/horizon20_config.yaml", help="配置文件路径")
    parser.add_argument("--start-date", type=str, default=None, help="训练/预测开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="训练/预测结束日期 YYYY-MM-DD")
    parser.add_argument("--train-window", type=str, default=None, help="V2训练窗口，如 3Y")
    parser.add_argument("--valid-window", type=str, default=None, help="V2验证窗口，如 2M")
    parser.add_argument("--gap", type=int, default=None, help="V2 gap交易日数，默认 horizon+1")
    parser.add_argument("--smooth-halflife", type=float, default=None, help="固定平滑半衰期，默认10")
    parser.add_argument("--horizon", type=int, default=None, help="覆盖label horizon")
    parser.add_argument("--freeze", action="store_true", help="启用冻结/增量模式：复用已有季度模型，只追加缺失预测")
    parser.add_argument("--reset-freeze", action="store_true", help="重置冻结状态，从本次范围重新生成")
    parser.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    production_path = Path(__file__).parent / "configs" / "production" / config_path.name
    if production_path.exists():
        with open(production_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    fallback_path = Path(__file__).parent / "configs" / config_path.name
    if fallback_path.exists():
        with open(fallback_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"找不到配置文件: {config_path}")


def apply_defaults(config: dict) -> dict:
    config.setdefault("data", {})
    config["data"].setdefault("open_column", "open")
    config["data"].setdefault("st_status_path", "01数据/data/tushare_data/st_status.parquet")
    config.setdefault("training", {})
    config["training"].setdefault("save_models", True)
    config["training"].setdefault("save_feature_importance", True)
    config.setdefault("output", {})
    config["output"].setdefault("experiments_dir", "03模型训练层/experiments")
    config["output"].setdefault("predictions_filename", "predictions.parquet")
    config.setdefault("quarterly_v2", {})
    config["quarterly_v2"].setdefault("train_window", config.get("walk_forward", {}).get("train_window", "3Y"))
    config["quarterly_v2"].setdefault("valid_window", "2M")
    config["quarterly_v2"].setdefault("smooth_halflife", 10)
    horizon = config["data"].get("label", {}).get("horizon", 20)
    config["quarterly_v2"].setdefault("gap", horizon + 1)
    return config


def update_config_from_args(config: dict, args) -> dict:
    if args.start_date:
        config.setdefault("walk_forward", {})["start_date"] = args.start_date
    if args.end_date:
        config.setdefault("walk_forward", {})["end_date"] = args.end_date
    if args.horizon is not None:
        config["data"]["label"]["horizon"] = args.horizon
        config["quarterly_v2"]["gap"] = args.horizon + 1
    if args.train_window:
        config["quarterly_v2"]["train_window"] = args.train_window
    if args.valid_window:
        config["quarterly_v2"]["valid_window"] = args.valid_window
    if args.gap is not None:
        config["quarterly_v2"]["gap"] = args.gap
    if args.smooth_halflife is not None:
        config["quarterly_v2"]["smooth_halflife"] = args.smooth_halflife
    if args.freeze:
        config["quarterly_v2"]["freeze"] = True
    if args.reset_freeze:
        config["quarterly_v2"]["freeze"] = True
        config["quarterly_v2"]["reset_freeze"] = True
    return config


def print_summary(config: dict, exp_id: str):
    qv2 = config["quarterly_v2"]
    label = config["data"]["label"]
    print("\n" + "=" * 70)
    print("Quarterly PIT V2 配置摘要")
    print("=" * 70)
    print(f"实验ID: {exp_id}")
    print(f"horizon: {label.get('horizon')} | open标签: {label.get('use_open_price', True)}")
    print(f"train_window: {qv2['train_window']}")
    print(f"valid_window: {qv2['valid_window']}")
    print(f"gap: {qv2['gap']} 个交易日")
    print(f"smooth_halflife: {qv2['smooth_halflife']}")
    print(f"freeze: {qv2.get('freeze', False)}")
    print(f"reset_freeze: {qv2.get('reset_freeze', False)}")
    print(f"start_date: {config.get('walk_forward', {}).get('start_date')}")
    print(f"end_date: {config.get('walk_forward', {}).get('end_date')}")
    print("=" * 70)


def main():
    args = parse_args()
    setup_logging()

    script_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    config = apply_defaults(load_config(config_path))
    config = update_config_from_args(config, args)

    exp_id = args.exp_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_v2"
    if not exp_id.endswith("_v2"):
        exp_id = f"{exp_id}_v2"

    print_summary(config, exp_id)
    if not args.yes:
        response = input("确认开始V2训练？(y/n): ").strip().lower()
        if response != "y":
            print("训练已取消")
            return

    trainer = QuarterlyTrainerV2(config, exp_id=exp_id)
    trainer.run()

    print("\n[OK] V2训练完成")
    print(f"实验目录: {trainer.exp_dir}")
    print(f"预测结果: {trainer.exp_dir / 'predictions.parquet'}")
    print(f"平滑预测: {trainer.exp_dir / 'smoothed_predictions.parquet'}")


if __name__ == "__main__":
    main()
