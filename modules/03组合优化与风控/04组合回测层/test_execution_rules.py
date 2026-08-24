from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("backtrader.eval.py")
SPEC = importlib.util.spec_from_file_location("portfolio_backtest", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"无法加载回测模块: {MODULE_PATH}")
BACKTEST_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKTEST_MODULE)
should_skip_minimum_rebalance = BACKTEST_MODULE.should_skip_minimum_rebalance


class MinimumRebalanceTests(unittest.TestCase):
    def test_small_adjustment_is_skipped_when_exposure_is_unchanged(self) -> None:
        self.assertTrue(
            should_skip_minimum_rebalance(
                target_weight=0.050,
                current_weight=0.046,
                current_shares=500,
                desired_target_shares=600,
                threshold=0.005,
                target_exposure=0.45,
                previous_target_exposure=0.45,
            )
        )

    def test_entry_exit_and_exposure_change_are_never_skipped(self) -> None:
        common = dict(
            target_weight=0.050,
            current_weight=0.046,
            threshold=0.005,
            target_exposure=0.45,
            previous_target_exposure=0.45,
        )
        self.assertFalse(
            should_skip_minimum_rebalance(
                **common, current_shares=0, desired_target_shares=500
            )
        )
        self.assertFalse(
            should_skip_minimum_rebalance(
                **common, current_shares=500, desired_target_shares=0
            )
        )
        changed = dict(common)
        changed["target_exposure"] = 0.90
        self.assertFalse(
            should_skip_minimum_rebalance(
                **changed, current_shares=500, desired_target_shares=600
            )
        )


if __name__ == "__main__":
    unittest.main()
