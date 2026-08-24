# -*- coding: utf-8 -*-
"""
Quarterly PIT trainer V2.

V2 keeps V1 intact and changes the signal-generation contract:
- one fixed model per natural quarter;
- the whole quarter is the prediction/test period for that model;
- no live/test splice is produced;
- predictions.parquet is the PIT prediction chain for backtests;
- smoothing uses a fixed halflife to avoid historical smooth revisions.
"""

from __future__ import annotations

import logging
import json
import hashlib
import copy
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from dataset.data_constructor_v1 import DataConstructorV1
from dataset.quarterly_splitter_v2 import QuarterlySplitterV2, QuarterlySplit
from models.lightgbm_model import LightGBMModel
from models.lightgbm_rank_model import LightGBMRankModel


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class QuarterlyTrainerV2:
    """Train one model per natural quarter and emit PIT predictions."""

    def __init__(self, config: Dict, exp_id: Optional[str] = None):
        self.config = config
        if exp_id is None:
            exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_v2"
        self.exp_id = exp_id

        self.exp_dir = Path(config["output"]["experiments_dir"]) / exp_id
        self.models_dir = self.exp_dir / "models"
        self.importance_dir = self.exp_dir / "feature_importance"
        self.logs_dir = self.exp_dir / "logs"
        self.state_dir = self.exp_dir / "state"
        self.state_models_dir = self.state_dir / "models"
        self.manifest_path = self.state_dir / "freeze_manifest.json"
        self.freeze_enabled = bool(config.get("quarterly_v2", {}).get("freeze", False))
        self.reset_freeze = bool(config.get("quarterly_v2", {}).get("reset_freeze", False))
        self._create_directories()

        self.data_constructor = DataConstructorV1(config)
        self.splitter: Optional[QuarterlySplitterV2] = None
        self.all_predictions: List[pd.DataFrame] = []
        self.summary_rows: List[Dict] = []
        self.manifest: Dict = {}
        self.existing_predictions: Optional[pd.DataFrame] = None
        self.existing_smoothed_predictions: Optional[pd.DataFrame] = None
        self.existing_summary: Optional[pd.DataFrame] = None

        if self.freeze_enabled:
            self._initialize_freeze_state()

    def _create_directories(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.importance_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.freeze_enabled:
            self.state_models_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_config_path(value):
        """Normalize project paths so absolute and project-relative forms compare equal."""
        if not isinstance(value, str) or not value.strip():
            return value
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        normalized = str(path.resolve(strict=False))
        return os.path.normcase(normalized)

    @classmethod
    def _normalized_data_config(cls, data_config: Dict) -> Dict:
        data = copy.deepcopy(data_config)
        factor_paths = data.get("factor_paths")
        if isinstance(factor_paths, dict):
            data["factor_paths"] = {
                key: cls._normalize_config_path(value)
                for key, value in factor_paths.items()
            }
        for key, value in list(data.items()):
            if key.endswith("_path") and key != "factor_paths":
                data[key] = cls._normalize_config_path(value)
        return data

    def _config_signature(self, config: Optional[Dict] = None, normalize_paths: bool = False) -> Dict:
        config = config or self.config
        qv2 = config.get("quarterly_v2", {})
        wf = config.get("walk_forward", {})
        data = config.get("data", {})
        if normalize_paths:
            data = self._normalized_data_config(data)
        signature = {
            "data": data,
            "model": config.get("model", {}),
            "quarterly_v2": {
                "train_window": qv2.get("train_window"),
                "valid_window": qv2.get("valid_window"),
                "gap": qv2.get("gap"),
                "smooth_halflife": qv2.get("smooth_halflife"),
                "smooth_window_multiplier": 2,
            },
            "walk_forward": {
                "train_window": wf.get("train_window"),
            },
        }
        return signature

    def _config_hash(self, config: Optional[Dict] = None, normalize_paths: bool = False) -> str:
        signature = self._config_signature(config=config, normalize_paths=normalize_paths)
        dumped = yaml.dump(signature, sort_keys=True, allow_unicode=True)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]

    def _new_manifest(self) -> Dict:
        qv2 = self.config.get("quarterly_v2", {})
        return {
            "schema_version": "qv2_freeze_1",
            "exp_id": self.exp_id,
            "config_hash": self._config_hash(),
            "normalized_config_hash": self._config_hash(normalize_paths=True),
            "smooth_halflife": float(qv2.get("smooth_halflife", 10)),
            "smooth_window_multiplier": 2,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "periods": {},
        }

    def _initialize_freeze_state(self):
        if self.reset_freeze and self.state_dir.exists():
            shutil.rmtree(self.state_dir)
            self.state_models_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.exists() and not self.reset_freeze:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            current_hash = self._config_hash()
            current_normalized_hash = self._config_hash(normalize_paths=True)
            if self.manifest.get("config_hash") != current_hash:
                compatible = self.manifest.get("normalized_config_hash") == current_normalized_hash

                # Backward compatibility for manifests created before paths were normalized.
                # The experiment config is the snapshot that originally produced the manifest.
                if not compatible:
                    saved_config_path = self.exp_dir / "config.yaml"
                    if saved_config_path.exists():
                        with open(saved_config_path, "r", encoding="utf-8") as f:
                            saved_config = yaml.safe_load(f)
                        saved_hash_matches_manifest = (
                            self._config_hash(config=saved_config)
                            == self.manifest.get("config_hash")
                        )
                        compatible = (
                            saved_hash_matches_manifest
                            and self._config_hash(config=saved_config, normalize_paths=True)
                            == current_normalized_hash
                        )

                if not compatible:
                    raise ValueError(
                        "freeze manifest 的关键配置与当前配置不一致；请换 exp_id 或使用 --reset-freeze"
                    )

                logger.warning(
                    "freeze: 配置仅存在绝对路径/项目相对路径差异，允许复用现有冻结状态"
                )
                self.manifest["config_hash"] = current_hash
            self.manifest["normalized_config_hash"] = current_normalized_hash
        else:
            self.manifest = self._new_manifest()

        if not self.reset_freeze:
            pred_path = self.exp_dir / self.config["output"].get("predictions_filename", "predictions.parquet")
            smooth_path = self.exp_dir / "smoothed_predictions.parquet"
            summary_path = self.exp_dir / "summary.parquet"
            if pred_path.exists():
                self.existing_predictions = pd.read_parquet(pred_path)
                self.existing_predictions["date"] = pd.to_datetime(self.existing_predictions["date"])
                logger.info(f"freeze: 已加载历史raw预测 {len(self.existing_predictions)} 行")
            if smooth_path.exists():
                self.existing_smoothed_predictions = pd.read_parquet(smooth_path)
                self.existing_smoothed_predictions["date"] = pd.to_datetime(self.existing_smoothed_predictions["date"])
                logger.info(f"freeze: 已加载历史smooth预测 {len(self.existing_smoothed_predictions)} 行")
            if summary_path.exists():
                self.existing_summary = pd.read_parquet(summary_path)

    def _save_manifest(self):
        if not self.freeze_enabled:
            return
        self.manifest["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def _existing_dates_for_period(self, model_period: str) -> set:
        if self.existing_predictions is None or len(self.existing_predictions) == 0:
            return set()
        df = self.existing_predictions
        if "model_period" in df.columns:
            df = df[df["model_period"] == model_period]
        return set(pd.to_datetime(df["date"]).dt.normalize().unique())

    def _state_model_path(self, split: QuarterlySplit) -> Path:
        return self.state_models_dir / f"model_{split.model_period}.pkl"

    def _find_legacy_model_path(self, split: QuarterlySplit) -> Optional[Path]:
        matches = sorted(self.models_dir.glob(f"model_{split.model_period}_fold_*.pkl"))
        return matches[-1] if matches else None

    def _load_model(self, path: Path):
        model_name = self.config["model"].get("name", "lightgbm")
        if model_name == "lightgbm_rank":
            return LightGBMRankModel.load(path)
        return LightGBMModel.load(path)

    def _load_or_prepare_frozen_model(self, split: QuarterlySplit):
        state_path = self._state_model_path(split)
        if state_path.exists():
            logger.info(f"freeze: 复用季度模型 {state_path}")
            return self._load_model(state_path)

        legacy_path = self._find_legacy_model_path(split)
        if legacy_path is not None and legacy_path.exists():
            shutil.copy2(legacy_path, state_path)
            logger.info(f"freeze: 从已有模型建立稳定缓存 {legacy_path} -> {state_path}")
            return self._load_model(state_path)

        return None

    def _update_manifest_period(self, split: QuarterlySplit, pred_df: Optional[pd.DataFrame] = None):
        if not self.freeze_enabled:
            return
        periods = self.manifest.setdefault("periods", {})
        existing = periods.get(split.model_period, {})
        state_path = self._state_model_path(split)

        pred_end = existing.get("pred_end")
        if self.existing_predictions is not None:
            period_df = self.existing_predictions
            if "model_period" in period_df.columns:
                period_df = period_df[period_df["model_period"] == split.model_period]
            if len(period_df) > 0:
                pred_end = str(pd.to_datetime(period_df["date"]).max().date())
        if pred_df is not None and len(pred_df) > 0:
            pred_end = str(pd.to_datetime(pred_df["date"]).max().date())

        periods[split.model_period] = {
            "model_path": str(state_path.relative_to(self.exp_dir)),
            "train_start": str(split.train_dates[0].date()),
            "train_end": str(split.train_dates[-1].date()),
            "valid_start": str(split.valid_dates[0].date()),
            "valid_end": str(split.valid_dates[-1].date()),
            "pred_end": pred_end,
        }

    def _save_config(self):
        config_path = self.exp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置已保存: {config_path}")

    def _initialize_splitter(self) -> QuarterlySplitterV2:
        close_df = self.data_constructor._load_close_data()
        dates = close_df.index.tolist()
        wf_config = self.config.get("walk_forward", {})
        v2_config = self.config.get("quarterly_v2", {})
        label_horizon = self.config["data"]["label"].get("horizon", 20)
        gap = v2_config.get("gap", wf_config.get("gap_valid_test", label_horizon + 1))

        splitter = QuarterlySplitterV2(
            dates=dates,
            train_window=v2_config.get("train_window", wf_config.get("train_window", "3Y")),
            valid_window=v2_config.get("valid_window", "2M"),
            label_horizon=label_horizon,
            gap=gap,
            start_date=wf_config.get("start_date"),
            end_date=wf_config.get("end_date"),
        )

        splits_path = self.exp_dir / "quarterly_splits.parquet"
        splitter.to_frame().to_parquet(splits_path, index=False)
        logger.info(f"季度切分已保存: {splits_path}")
        logger.info(f"季度fold数量: {splitter.get_n_splits()}")
        logger.info(f"V2 gap: {gap} 个交易日")
        return splitter

    def _create_model(self):
        model_name = self.config["model"].get("name", "lightgbm")
        if model_name == "lightgbm_rank":
            logger.info("使用 LambdaRank 排序模型")
            return LightGBMRankModel(self.config)
        logger.info("使用 LightGBM 回归模型")
        return LightGBMModel(self.config)

    def _attach_actual_return(self, pred_df: pd.DataFrame, pred_dates: List[pd.Timestamp]) -> pd.DataFrame:
        try:
            labels = self.data_constructor._compute_labels(pred_dates)
            labels_long = labels.stack(dropna=False).rename("actual_return").reset_index()
            labels_long.columns = ["date", "stock_code", "actual_return"]
            labels_long["date"] = pd.to_datetime(labels_long["date"])
            pred_df = pred_df.merge(labels_long, on=["date", "stock_code"], how="left")
        except Exception as exc:
            logger.warning(f"回填actual_return失败，保留NaN: {exc}")
            pred_df["actual_return"] = np.nan
        pred_df["is_evaluable"] = pred_df["actual_return"].notna()
        return pred_df

    def _predict_period(self, model, split: QuarterlySplit) -> pd.DataFrame:
        X_pred = self.data_constructor.build_for_prediction(split.pred_dates)
        if len(X_pred) == 0:
            return pd.DataFrame()

        pred_score = model.predict(X_pred)
        pred_df = pd.DataFrame(
            {
                "date": X_pred.index.get_level_values(0),
                "stock_code": X_pred.index.get_level_values(1),
                "pred_score": pred_score,
                "fold_id": split.fold_id,
                "model_period": split.model_period,
                "period_start": split.period_start,
                "period_end": split.period_end,
                "train_start": split.train_dates[0],
                "train_end": split.train_dates[-1],
                "valid_start": split.valid_dates[0],
                "valid_end": split.valid_dates[-1],
            }
        )
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df = self._attach_actual_return(pred_df, split.pred_dates)
        return pred_df

    def _save_summary(self):
        if not self.summary_rows and self.existing_summary is None:
            return

        pieces = []
        if self.freeze_enabled and self.existing_summary is not None:
            pieces.append(self.existing_summary)
        if self.summary_rows:
            pieces.append(pd.DataFrame(self.summary_rows))
        summary_df = pd.concat(pieces, axis=0, ignore_index=True)
        if "model_period" in summary_df.columns:
            summary_df = summary_df.drop_duplicates(["model_period"], keep="last")
        if "period_start" in summary_df.columns:
            summary_df = summary_df.sort_values("period_start")

        summary_path = self.exp_dir / "summary.parquet"
        summary_df.to_parquet(summary_path, index=False)
        logger.info(f"quarterly summary saved: {summary_path}")

        if HAS_MATPLOTLIB and len(summary_df) > 0:
            try:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(summary_df["period_start"], summary_df["daily_rank_ic_mean"], marker="o")
                ax.axhline(0, color="black", linestyle="--", alpha=0.3)
                ax.set_title("Quarterly V2 Daily Rank IC")
                ax.set_ylabel("Daily Rank IC Mean")
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                fig.tight_layout()
                plot_path = self.exp_dir / "quarterly_rank_ic_v2.png"
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"rank IC plot saved: {plot_path}")
            except Exception as exc:
                logger.warning(f"failed to generate rank IC plot: {exc}")
            self._plot_feature_importance_v2()

    def _plot_feature_importance_v2(self):
        if not HAS_MATPLOTLIB:
            return

        importance_rows = []
        for path in sorted(self.importance_dir.glob("importance_*_fold_*.csv")):
            try:
                imp = pd.read_csv(path, index_col=0)
                if "importance" not in imp.columns:
                    continue
                model_period = path.stem.replace("importance_", "").split("_fold_")[0]
                for feature, row in imp.iterrows():
                    importance_rows.append(
                        {
                            "model_period": model_period,
                            "feature": feature,
                            "importance": row.get("importance", np.nan),
                        }
                    )
            except Exception as exc:
                logger.warning(f"failed to read feature importance {path}: {exc}")

        if not importance_rows:
            logger.info("no feature importance csv files found, skip feature_importance_v2.png")
            return

        imp_df = pd.DataFrame(importance_rows)
        pivot = imp_df.pivot_table(index="model_period", columns="feature", values="importance", aggfunc="mean")
        mean_importance = pivot.mean(axis=0).sort_values(ascending=False)
        top_features = mean_importance.head(15)
        top10_features = mean_importance.head(10).index

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax1 = axes[0]
        bars = ax1.barh(range(len(top_features)), top_features.values)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features.index)
        ax1.set_xlabel("Average Importance (Gain)")
        ax1.set_title("Top 15 Feature Importance (V2)")
        ax1.invert_yaxis()
        max_val = max(top_features.values) if len(top_features) else 0
        for i, (bar, val) in enumerate(zip(bars, top_features.values)):
            ax1.text(val + max_val * 0.01, i, f"{val:.0f}", va="center", fontsize=9)

        ax2 = axes[1]
        heat = pivot[top10_features].T
        im = ax2.imshow(heat.values, aspect="auto", cmap="YlOrRd")
        ax2.set_yticks(range(len(top10_features)))
        ax2.set_yticklabels(top10_features)
        ax2.set_xticks(range(len(heat.columns)))
        ax2.set_xticklabels(heat.columns, rotation=45, ha="right")
        ax2.set_title("Feature Importance Heatmap (V2)")
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Importance", rotation=270, labelpad=15)

        fig.tight_layout()
        fi_plot_path = self.exp_dir / "feature_importance_v2.png"
        fig.savefig(fi_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"feature importance plot saved: {fi_plot_path}")

    @staticmethod
    def _calc_metrics(df: pd.DataFrame) -> Dict:
        eval_df = df[df["is_evaluable"]].copy()
        if len(eval_df) < 10:
            return {
                "n_eval_samples": len(eval_df),
                "ic": np.nan,
                "rank_ic": np.nan,
                "daily_rank_ic_mean": np.nan,
                "daily_rank_ic_std": np.nan,
            }

        daily_rank_ics = []
        for _, day_df in eval_df.groupby("date"):
            if len(day_df) >= 10:
                corr = day_df["pred_score"].corr(day_df["actual_return"], method="spearman")
                if not np.isnan(corr):
                    daily_rank_ics.append(corr)

        return {
            "n_eval_samples": len(eval_df),
            "ic": eval_df["pred_score"].corr(eval_df["actual_return"]),
            "rank_ic": eval_df["pred_score"].corr(eval_df["actual_return"], method="spearman"),
            "daily_rank_ic_mean": float(np.mean(daily_rank_ics)) if daily_rank_ics else np.nan,
            "daily_rank_ic_std": float(np.std(daily_rank_ics, ddof=1)) if len(daily_rank_ics) > 1 else np.nan,
        }

    def _train_quarter_legacy(self, split: QuarterlySplit) -> pd.DataFrame:
        logger.info("\n" + "=" * 70)
        logger.info(f"训练 {split.model_period} | {split.period_start.date()} ~ {split.period_end.date()}")
        logger.info("=" * 70)
        logger.info(f"Train: {split.train_dates[0].date()} ~ {split.train_dates[-1].date()} ({len(split.train_dates)}天)")
        logger.info(f"Valid: {split.valid_dates[0].date()} ~ {split.valid_dates[-1].date()} ({len(split.valid_dates)}天)")
        logger.info(f"Pred : {split.pred_dates[0].date()} ~ {split.pred_dates[-1].date()} ({len(split.pred_dates)}天)")

        start_time = time.time()
        X_train, y_train = self.data_constructor.build(split.train_dates, apply_profit_filter=True)
        X_valid, y_valid = self.data_constructor.build(split.valid_dates, apply_profit_filter=False)
        if len(X_train) == 0 or len(X_valid) == 0:
            logger.warning(f"{split.model_period}: 训练或验证数据为空，跳过")
            return pd.DataFrame()

        model = self._create_model()
        model.fit(X_train, y_train, X_valid, y_valid)
        pred_df = self._predict_period(model, split)
        if len(pred_df) == 0:
            logger.warning(f"{split.model_period}: 预测数据为空，跳过")
            return pd.DataFrame()

        if self.config["training"].get("save_models", True):
            model_path = self.models_dir / f"model_{split.model_period}_fold_{split.fold_id:03d}.pkl"
            model.save(model_path)
            logger.info(f"模型已保存: {model_path}")

        if self.config["training"].get("save_feature_importance", True):
            try:
                importance = model.get_feature_importance()
                importance_path = self.importance_dir / f"importance_{split.model_period}_fold_{split.fold_id:03d}.csv"
                importance.to_csv(importance_path)
                logger.info(f"特征重要性已保存: {importance_path}")
            except Exception as exc:
                logger.warning(f"保存特征重要性失败: {exc}")

        metrics = self._calc_metrics(pred_df)
        row = {
            "fold_id": split.fold_id,
            "model_period": split.model_period,
            "period_start": split.period_start,
            "period_end": split.period_end,
            "train_start": split.train_dates[0],
            "train_end": split.train_dates[-1],
            "valid_start": split.valid_dates[0],
            "valid_end": split.valid_dates[-1],
            "n_train_samples": len(X_train),
            "n_valid_samples": len(X_valid),
            "n_pred_samples": len(pred_df),
            "elapsed_sec": time.time() - start_time,
            **metrics,
        }
        self.summary_rows.append(row)
        logger.info(
            f"{split.model_period} 完成: n_pred={len(pred_df)}, "
            f"rank_ic={row['rank_ic']:.4f}, daily_rank_ic_mean={row['daily_rank_ic_mean']:.4f}"
        )
        return pred_df

    def _train_quarter(self, split: QuarterlySplit) -> pd.DataFrame:
        logger.info("\n" + "=" * 70)
        logger.info(f"Train/Predict {split.model_period} | {split.period_start.date()} ~ {split.period_end.date()}")
        logger.info("=" * 70)
        logger.info(f"Train: {split.train_dates[0].date()} ~ {split.train_dates[-1].date()} ({len(split.train_dates)} dates)")
        logger.info(f"Valid: {split.valid_dates[0].date()} ~ {split.valid_dates[-1].date()} ({len(split.valid_dates)} dates)")
        logger.info(f"Pred : {split.pred_dates[0].date()} ~ {split.pred_dates[-1].date()} ({len(split.pred_dates)} dates)")

        start_time = time.time()
        pred_dates = split.pred_dates
        if self.freeze_enabled:
            existing_dates = self._existing_dates_for_period(split.model_period)
            pred_dates = [d for d in split.pred_dates if pd.Timestamp(d).normalize() not in existing_dates]
            if len(pred_dates) == 0:
                logger.info(f"freeze: {split.model_period} all prediction dates already exist, skip")
                self._load_or_prepare_frozen_model(split)
                self._update_manifest_period(split)
                return pd.DataFrame()
            logger.info(f"freeze: {split.model_period} new prediction dates {len(pred_dates)} / {len(split.pred_dates)}")

        model = self._load_or_prepare_frozen_model(split) if self.freeze_enabled else None
        X_train = y_train = X_valid = y_valid = None
        trained_this_run = False

        if model is None:
            X_train, y_train = self.data_constructor.build(split.train_dates, apply_profit_filter=True)
            X_valid, y_valid = self.data_constructor.build(split.valid_dates, apply_profit_filter=False)
            if len(X_train) == 0 or len(X_valid) == 0:
                logger.warning(f"{split.model_period}: empty train/valid data, skip")
                return pd.DataFrame()

            model = self._create_model()
            model.fit(X_train, y_train, X_valid, y_valid)
            trained_this_run = True
        else:
            logger.info(f"freeze: {split.model_period} model reused, no retraining")

        pred_split = QuarterlySplit(
            fold_id=split.fold_id,
            model_period=split.model_period,
            period_start=split.period_start,
            period_end=split.period_end,
            train_dates=split.train_dates,
            valid_dates=split.valid_dates,
            pred_dates=pred_dates,
        )
        pred_df = self._predict_period(model, pred_split)
        if len(pred_df) == 0:
            logger.warning(f"{split.model_period}: empty prediction data, skip")
            return pd.DataFrame()

        if self.config["training"].get("save_models", True) and trained_this_run:
            model_path = self.models_dir / f"model_{split.model_period}_fold_{split.fold_id:03d}.pkl"
            model.save(model_path)
            logger.info(f"model saved: {model_path}")
            if self.freeze_enabled:
                state_model_path = self._state_model_path(split)
                model.save(state_model_path)
                logger.info(f"freeze: model cached at {state_model_path}")

        if self.config["training"].get("save_feature_importance", True) and trained_this_run:
            try:
                importance = model.get_feature_importance()
                importance_path = self.importance_dir / f"importance_{split.model_period}_fold_{split.fold_id:03d}.csv"
                importance.to_csv(importance_path)
                logger.info(f"feature importance saved: {importance_path}")
            except Exception as exc:
                logger.warning(f"failed to save feature importance: {exc}")

        metrics = self._calc_metrics(pred_df)
        row = {
            "fold_id": split.fold_id,
            "model_period": split.model_period,
            "period_start": split.period_start,
            "period_end": split.period_end,
            "train_start": split.train_dates[0],
            "train_end": split.train_dates[-1],
            "valid_start": split.valid_dates[0],
            "valid_end": split.valid_dates[-1],
            "n_train_samples": len(X_train) if X_train is not None else np.nan,
            "n_valid_samples": len(X_valid) if X_valid is not None else np.nan,
            "n_pred_samples": len(pred_df),
            "elapsed_sec": time.time() - start_time,
            **metrics,
        }
        self.summary_rows.append(row)
        self._update_manifest_period(split, pred_df)
        logger.info(
            f"{split.model_period} done: n_pred={len(pred_df)}, "
            f"rank_ic={row['rank_ic']:.4f}, daily_rank_ic_mean={row['daily_rank_ic_mean']:.4f}"
        )
        return pred_df

    def _save_predictions_legacy(self) -> pd.DataFrame:
        if not self.all_predictions:
            logger.warning("没有预测结果可保存")
            return pd.DataFrame()

        pred_df = pd.concat(self.all_predictions, axis=0, ignore_index=True)
        pred_df = pred_df.sort_values(["date", "stock_code", "fold_id"])
        dup_count = pred_df.duplicated(["date", "stock_code"], keep=False).sum()
        if dup_count:
            logger.warning(f"发现 {dup_count} 条重复date+stock预测，保留最后一条")
            pred_df = pred_df.drop_duplicates(["date", "stock_code"], keep="last")

        pred_path = self.exp_dir / self.config["output"].get("predictions_filename", "predictions.parquet")
        pred_df.to_parquet(pred_path, index=False)
        logger.info(f"PIT预测已保存: {pred_path}")
        logger.info(f"日期范围: {pred_df['date'].min()} ~ {pred_df['date'].max()}")
        return pred_df

    def _save_predictions(self) -> pd.DataFrame:
        if not self.all_predictions and self.existing_predictions is None:
            logger.warning("no predictions to save")
            return pd.DataFrame()

        pieces = []
        n_existing = 0
        if self.freeze_enabled and self.existing_predictions is not None:
            existing = self.existing_predictions.copy()
            existing["_is_existing"] = True
            pieces.append(existing)
            n_existing = len(existing)
        for df in self.all_predictions:
            part = df.copy()
            part["_is_existing"] = False
            pieces.append(part)

        pred_df = pd.concat(pieces, axis=0, ignore_index=True)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        if self.freeze_enabled:
            pred_df = pred_df.sort_values(["date", "stock_code", "_is_existing"])
        elif "fold_id" in pred_df.columns:
            pred_df = pred_df.sort_values(["date", "stock_code", "fold_id"])
        else:
            pred_df = pred_df.sort_values(["date", "stock_code"])
        dup_count = pred_df.duplicated(["date", "stock_code"], keep=False).sum()
        if dup_count:
            if self.freeze_enabled:
                logger.warning(f"freeze: found {dup_count} duplicate date+stock rows; keeping existing frozen rows")
                pred_df = pred_df.drop_duplicates(["date", "stock_code"], keep="last")
            else:
                logger.warning(f"found {dup_count} duplicate date+stock rows; keeping latest rows")
                pred_df = pred_df.drop_duplicates(["date", "stock_code"], keep="last")
        pred_df = pred_df.drop(columns=["_is_existing"])
        pred_df = pred_df.sort_values(["date", "stock_code"]).reset_index(drop=True)

        pred_path = self.exp_dir / self.config["output"].get("predictions_filename", "predictions.parquet")
        pred_df.to_parquet(pred_path, index=False)
        self.existing_predictions = pred_df.copy()
        logger.info(f"PIT predictions saved: {pred_path}")
        logger.info(f"prediction rows: {len(pred_df)} (existing loaded: {n_existing}, new chunks: {len(self.all_predictions)})")
        logger.info(f"date range: {pred_df['date'].min()} ~ {pred_df['date'].max()}")
        return pred_df

    def _save_summary_legacy(self):
        if not self.summary_rows:
            return
        summary_df = pd.DataFrame(self.summary_rows)
        summary_path = self.exp_dir / "summary.parquet"
        summary_df.to_parquet(summary_path, index=False)
        logger.info(f"季度汇总已保存: {summary_path}")

        if HAS_MATPLOTLIB and len(summary_df) > 0:
            try:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(summary_df["period_start"], summary_df["daily_rank_ic_mean"], marker="o")
                ax.axhline(0, color="black", linestyle="--", alpha=0.3)
                ax.set_title("Quarterly V2 Daily Rank IC")
                ax.set_ylabel("Daily Rank IC Mean")
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                fig.tight_layout()
                plot_path = self.exp_dir / "quarterly_rank_ic_v2.png"
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Rank IC图已保存: {plot_path}")
            except Exception as exc:
                logger.warning(f"生成Rank IC图失败: {exc}")

    @staticmethod
    def smooth_predictions(df: pd.DataFrame, halflife: float = 10.0) -> pd.DataFrame:
        window = max(1, int(round(halflife * 2)))
        weights = (0.5 ** (np.arange(window) / halflife))[::-1]
        weights = weights / weights.sum()

        def ewma(values):
            arr = np.asarray(values, dtype=float)
            result = np.empty(len(arr), dtype=float)
            for i in range(len(arr)):
                start = max(0, i - window + 1)
                w = weights[-(i - start + 1):]
                w = w / w.sum()
                result[i] = np.average(arr[start : i + 1], weights=w)
            return result

        out = df.sort_values(["stock_code", "date"]).copy()
        out["pred_score_smooth"] = out.groupby("stock_code")["pred_score"].transform(ewma)
        return out.sort_values(["date", "stock_code"]).reset_index(drop=True)

    def _save_smoothed_predictions_legacy(self, pred_df: pd.DataFrame):
        halflife = float(self.config.get("quarterly_v2", {}).get("smooth_halflife", 10))
        smooth_df = self.smooth_predictions(pred_df, halflife=halflife)
        smooth_path = self.exp_dir / "smoothed_predictions.parquet"
        cols = list(smooth_df.columns)
        smooth_df[cols].to_parquet(smooth_path, index=False)
        logger.info(f"固定halflife={halflife:g}的平滑预测已保存: {smooth_path}")

    def _save_smoothed_predictions(self, pred_df: pd.DataFrame):
        halflife = float(self.config.get("quarterly_v2", {}).get("smooth_halflife", 10))
        smooth_df = self.smooth_predictions(pred_df, halflife=halflife)
        if self.freeze_enabled and self.existing_smoothed_predictions is not None:
            existing = self.existing_smoothed_predictions.copy()
            existing["date"] = pd.to_datetime(existing["date"])
            existing_keys = pd.MultiIndex.from_frame(existing[["date", "stock_code"]])
            smooth_keys = pd.MultiIndex.from_frame(smooth_df[["date", "stock_code"]])
            new_smooth = smooth_df[~smooth_keys.isin(existing_keys)].copy()
            smooth_df = pd.concat([existing, new_smooth], axis=0, ignore_index=True)
            smooth_df = smooth_df.drop_duplicates(["date", "stock_code"], keep="first")
            logger.info(f"freeze: smooth existing rows kept={len(existing)}, new rows appended={len(new_smooth)}")
        smooth_df = smooth_df.sort_values(["date", "stock_code"]).reset_index(drop=True)
        smooth_path = self.exp_dir / "smoothed_predictions.parquet"
        smooth_df.to_parquet(smooth_path, index=False)
        self.existing_smoothed_predictions = smooth_df.copy()
        logger.info(f"fixed halflife={halflife:g} smoothed predictions saved: {smooth_path}")

    def run(self):
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("开始 Quarterly PIT Trainer V2")
        logger.info("=" * 80)
        self._save_config()
        self.splitter = self._initialize_splitter()

        for split in self.splitter.get_splits():
            try:
                pred_df = self._train_quarter(split)
                if len(pred_df) > 0:
                    self.all_predictions.append(pred_df)
            except Exception as exc:
                logger.error(f"{split.model_period} 训练失败: {exc}", exc_info=True)
                continue

        pred_df = self._save_predictions()
        if len(pred_df) > 0:
            self._save_smoothed_predictions(pred_df)
        self._save_summary()
        self._save_manifest()

        logger.info("=" * 80)
        logger.info(f"Quarterly PIT Trainer V2 完成，总耗时 {(time.time() - start_time) / 60:.2f} 分钟")
        logger.info(f"实验目录: {self.exp_dir}")
        logger.info("=" * 80)
