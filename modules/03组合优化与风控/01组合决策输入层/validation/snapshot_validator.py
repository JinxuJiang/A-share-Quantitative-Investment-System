from __future__ import annotations

import numpy as np
import pandas as pd


class SnapshotValidator:
    """Fail fast when a decision snapshot is internally inconsistent."""

    def __init__(self, top_n: int, horizon_days: int, psd_tolerance: float = 1e-10):
        self.top_n = int(top_n)
        self.horizon_days = int(horizon_days)
        self.psd_tolerance = float(psd_tolerance)

    def validate(
        self,
        universe: pd.DataFrame,
        alpha_signal: pd.DataFrame,
        covariance: pd.DataFrame,
        asset_risk: pd.DataFrame,
        market_signal: pd.Series,
        signal_date: pd.Timestamp,
    ) -> dict:
        date = pd.Timestamp(signal_date).normalize()
        selected = alpha_signal.copy()
        if len(selected) != self.top_n:
            raise ValueError(f"选股数量应为{self.top_n}，实际为{len(selected)}")
        if selected["stock_code"].duplicated().any():
            raise ValueError("alpha_signal存在重复股票")
        if not selected["is_eligible"].fillna(False).all():
            raise ValueError("alpha_signal包含不可投资股票")
        if not selected["is_selected"].fillna(False).all():
            raise ValueError("alpha_signal包含未入选股票")
        if not np.isfinite(selected[["alpha_score", "alpha_zscore"]].to_numpy(dtype="float64")).all():
            raise ValueError("alpha_signal包含非有限Alpha值")
        expected_ranks = np.arange(1, self.top_n + 1)
        if not np.array_equal(selected["selection_rank"].to_numpy(), expected_ranks):
            raise ValueError("selection_rank必须从1连续排列到top_n")

        selected_codes = selected["stock_code"].astype(str).tolist()
        expected_pairs = self.top_n * self.top_n
        if len(covariance) != expected_pairs:
            raise ValueError(f"协方差长表应有{expected_pairs}行，实际为{len(covariance)}")
        matrix = covariance.pivot(index="asset_i", columns="asset_j", values="covariance_20d")
        matrix = matrix.reindex(index=selected_codes, columns=selected_codes)
        values = matrix.to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("协方差矩阵缺少选中股票或包含非有限值")
        if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
            raise ValueError("协方差矩阵不对称")
        min_eigenvalue = float(np.linalg.eigvalsh(values).min())
        if min_eigenvalue < -self.psd_tolerance:
            raise ValueError(f"协方差矩阵不是半正定矩阵，最小特征值={min_eigenvalue}")

        risk_codes = asset_risk["stock_code"].astype(str).tolist()
        if set(risk_codes) != set(selected_codes) or len(risk_codes) != self.top_n:
            raise ValueError("asset_risk股票集合与alpha_signal不一致")
        if not np.isfinite(asset_risk[["volatility_daily", "volatility_20d"]].to_numpy()).all():
            raise ValueError("asset_risk包含非有限值")

        for name, frame in {
            "alpha_signal": selected,
            "covariance": covariance,
            "asset_risk": asset_risk,
        }.items():
            dates = pd.to_datetime(frame["signal_date"]).dt.normalize().unique()
            if len(dates) != 1 or pd.Timestamp(dates[0]) != date:
                raise ValueError(f"{name}的signal_date不一致")
            if not frame["horizon_days"].eq(self.horizon_days).all():
                raise ValueError(f"{name}的horizon_days不一致")

        if pd.Timestamp(market_signal["signal_date"]).normalize() != date:
            raise ValueError("市场信号与截面Alpha的signal_date不一致")
        if int(market_signal["horizon_days"]) != self.horizon_days:
            raise ValueError("市场信号的horizon_days不一致")
        exposure = float(market_signal["target_equity_exposure"])
        if not 0.0 <= exposure <= 1.0:
            raise ValueError("target_equity_exposure必须位于[0, 1]")

        selected_in_universe = universe.loc[universe["is_selected"].fillna(False), "stock_code"]
        if set(selected_in_universe.astype(str)) != set(selected_codes):
            raise ValueError("universe与alpha_signal的入选股票不一致")
        return {
            "status": "passed",
            "selected_assets": self.top_n,
            "covariance_rows": expected_pairs,
            "min_eigenvalue_20d": min_eigenvalue,
        }
