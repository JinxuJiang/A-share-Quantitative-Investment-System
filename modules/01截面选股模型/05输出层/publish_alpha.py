# -*- coding: utf-8 -*-
"""
正式 Alpha 发布脚本
===================

用途：
    将已完成训练和回测验收的 smoothed_predictions.parquet，
    转换为供后续组合仓库读取的标准截面 Alpha。

常用命令：
    python publish_alpha.py --exp-id ensemble_5d_20d_60d_profit20_v2 --release-id alpha_5d_20d_60d_tushare_profit20_v2

输出：
    exports/releases/{release_id}/
        stock_alpha.parquet
        manifest.json
    exports/current.json

注意：
    每次发布都会自动更新 current.json；
    已存在的 release-id 不允许覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "03模型训练层" / "experiments"
EXPORT_ROOT = Path(__file__).resolve().parent / "exports"
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发布正式截面 alpha")
    parser.add_argument("--exp-id", required=True, help="已验收实验 ID，例如 lgbm20_tushare_profit20_v2")
    parser.add_argument("--release-id", required=True, help="不可变发布 ID，例如 alpha_20d_tushare_profit20_20260728_v1")
    return parser.parse_args()


def validate_id(value: str, field_name: str) -> str:
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} 只能包含英文字母、数字、点、下划线和连字符: {value}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件内容无效: {path}")
    return data


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


def build_stock_alpha(source_path: Path, horizon_days: int) -> pd.DataFrame:
    required_columns = ["date", "stock_code", "pred_score_smooth"]
    source = pd.read_parquet(source_path, columns=required_columns)

    missing = [column for column in required_columns if column not in source.columns]
    if missing:
        raise ValueError(f"预测文件缺少字段: {missing}")

    alpha = source.rename(
        columns={
            "date": "signal_date",
            "pred_score_smooth": "alpha_score",
        }
    )
    alpha["signal_date"] = pd.to_datetime(alpha["signal_date"], errors="raise")
    alpha["stock_code"] = alpha["stock_code"].astype("string")
    alpha["alpha_score"] = pd.to_numeric(alpha["alpha_score"], errors="raise")

    if alpha[["signal_date", "stock_code", "alpha_score"]].isna().any().any():
        raise ValueError("正式 alpha 中不允许 signal_date、stock_code 或 alpha_score 为空")
    if not np.isfinite(alpha["alpha_score"].to_numpy()).all():
        raise ValueError("正式 alpha 中存在无穷值")
    if alpha.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("正式 alpha 中存在重复的 signal_date + stock_code")

    alpha["alpha_rank"] = (
        alpha.groupby("signal_date", sort=False)["alpha_score"]
        .rank(method="average", pct=True)
        .astype("float32")
    )
    alpha["horizon_days"] = np.int16(horizon_days)

    return alpha[
        [
            "signal_date",
            "stock_code",
            "alpha_score",
            "alpha_rank",
            "horizon_days",
        ]
    ]


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
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def publish(exp_id: str, release_id: str) -> Path:
    exp_id = validate_id(exp_id, "exp-id")
    release_id = validate_id(release_id, "release-id")

    experiment_dir = EXPERIMENTS_DIR / exp_id
    config_path = experiment_dir / "config.yaml"
    source_path = experiment_dir / "smoothed_predictions.parquet"
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"找不到实验目录: {experiment_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到实验配置: {config_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到平滑预测: {source_path}")

    config = load_yaml(config_path)
    try:
        horizon_days = int(config["data"]["label"]["horizon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("实验配置缺少有效的 data.label.horizon") from exc
    if horizon_days <= 0 or horizon_days > np.iinfo(np.int16).max:
        raise ValueError(f"horizon_days 超出允许范围: {horizon_days}")

    releases_dir = EXPORT_ROOT / "releases"
    release_dir = releases_dir / release_id
    if release_dir.exists():
        raise FileExistsError(f"release 已存在，禁止覆盖: {release_dir}")

    releases_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=releases_dir))
    published_at = datetime.now().astimezone().isoformat(timespec="seconds")

    try:
        alpha = build_stock_alpha(source_path, horizon_days)
        output_path = temp_dir / "stock_alpha.parquet"
        alpha.to_parquet(output_path, index=False, compression="zstd")

        git_commit, git_dirty = git_metadata()
        manifest = {
            "schema_version": "stock_alpha_v1",
            "release_id": release_id,
            "source_exp_id": exp_id,
            "source_file": str(source_path.relative_to(PROJECT_ROOT)),
            "source_file_sha256": sha256_file(source_path),
            "source_config_sha256": sha256_file(config_path),
            "source_git_commit": git_commit,
            "source_git_dirty": git_dirty,
            "source_model": config.get("model", {}).get("name"),
            "source_tuning_study": config.get("tuning_metadata", {}).get("study_name"),
            "source_tuning_candidate": config.get("tuning_metadata", {}).get("recommended_source"),
            "horizon_days": horizon_days,
            "signal_available_after": "market_close",
            "execution_lag_trading_days": 1,
            "data_start": alpha["signal_date"].min().date().isoformat(),
            "data_end": alpha["signal_date"].max().date().isoformat(),
            "published_at": published_at,
            "row_count": len(alpha),
            "date_count": int(alpha["signal_date"].nunique()),
            "stock_count": int(alpha["stock_code"].nunique()),
            "columns": {
                "signal_date": "datetime64[ns]",
                "stock_code": "string",
                "alpha_score": str(alpha["alpha_score"].dtype),
                "alpha_rank": "float32; 当日截面百分位，数值越大排名越高",
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

    current = {
        "schema_version": "stock_alpha_current_v1",
        "release_id": release_id,
        "manifest": f"releases/{release_id}/manifest.json",
        "updated_at": published_at,
    }
    write_json_atomic(EXPORT_ROOT / "current.json", current)
    return release_dir


def main() -> None:
    args = parse_args()
    release_dir = publish(args.exp_id, args.release_id)
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))

    print("=" * 70)
    print("正式 alpha 发布完成")
    print("=" * 70)
    print(f"release_id : {manifest['release_id']}")
    print(f"source_exp : {manifest['source_exp_id']}")
    print(f"date_range : {manifest['data_start']} -> {manifest['data_end']}")
    print(f"rows       : {manifest['row_count']:,}")
    print(f"release    : {release_dir}")
    print(f"current    : {EXPORT_ROOT / 'current.json'}")


if __name__ == "__main__":
    main()
