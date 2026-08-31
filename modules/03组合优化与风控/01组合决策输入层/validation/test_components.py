from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VALIDATION_ROOT = ROOT / "validation"
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from estimation.alpha.alpha_transformer import AlphaTransformer
from estimation.build_decision_snapshot import official_sessions
from estimation.risk.covariance_estimator import LedoitWolfCovarianceEstimator
from estimation.risk.return_builder import ReturnMatrixBuilder
from universe.universe_builder import UniverseBuilder
from snapshot_validator import SnapshotValidator


class ComponentTests(unittest.TestCase):
    def test_official_sessions_preserve_each_open_a_share_date(self) -> None:
        schedule = pd.DataFrame(
            {
                "cal_date": ["20260720", "20260721", "20260722", "20260723"],
                "is_open": [1, 1, 0, 1],
            }
        )
        result = official_sessions(schedule)
        expected = pd.to_datetime(["2026-07-20", "2026-07-21", "2026-07-23"])
        self.assertEqual(result.tolist(), expected.tolist())

    def test_decision_pipeline_components(self) -> None:
        date = pd.Timestamp("2026-07-24")
        codes = [f"{i:06d}.SZ" for i in range(1, 6)] + ["300001.SZ"]
        alpha = pd.DataFrame(
            {
                "signal_date": date,
                "stock_code": codes,
                "alpha_score": np.arange(6, dtype=float),
                "source_alpha_rank": np.linspace(0.1, 0.9, 6),
                "horizon_days": 20,
            }
        )
        stock_info = pd.DataFrame(
            {
                "order_book_id": codes,
                "market": ["主板"] * 5 + ["创业板"],
                "list_date": ["20200101"] * 6,
                "delist_date": [None] * 6,
            }
        )
        status = pd.Series(0, index=codes)
        observations = pd.Series(253, index=codes)
        universe = UniverseBuilder(min_price_observations=253).build(
            alpha, stock_info, status, status, observations, date
        )
        universe, selected = AlphaTransformer().transform(universe, top_n=3)
        self.assertEqual(selected["stock_code"].tolist(), [codes[4], codes[3], codes[2]])
        self.assertEqual(str(selected["selection_rank"].dtype), "int32")
        self.assertTrue(selected["eligible_alpha_rank"].between(0, 1).all())

        rng = np.random.default_rng(7)
        daily_returns = rng.normal(0.0003, 0.015, size=(252, 3))
        price_index = pd.bdate_range(end=date, periods=253)
        prices = pd.DataFrame(
            np.vstack([np.ones(3), np.cumprod(1 + daily_returns, axis=0)]) * 100,
            index=price_index,
            columns=selected["stock_code"],
        )
        _, returns = ReturnMatrixBuilder(252).build(prices, selected["stock_code"].tolist(), date)
        covariance, asset_risk, _ = LedoitWolfCovarianceEstimator(20).fit(returns, date)
        selected["horizon_days"] = 20
        market_signal = pd.Series(
            {
                "signal_date": date,
                "target_equity_exposure": 0.45,
                "horizon_days": 20,
            }
        )
        result = SnapshotValidator(3, 20).validate(
            universe, selected, covariance, asset_risk, market_signal, date
        )
        self.assertEqual(result["status"], "passed")
        self.assertGreaterEqual(result["min_eigenvalue_20d"], -1e-10)


if __name__ == "__main__":
    unittest.main()
