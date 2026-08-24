from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk.parametric_var_es import ParametricNormalVarEs
from validation.risk_validator import RiskValidator


class VarEsTests(unittest.TestCase):
    def test_normal_var_es_and_validation(self) -> None:
        date = pd.Timestamp("2026-08-14")
        weights = pd.DataFrame(
            {
                "signal_date": date,
                "stock_code": ["A", "B"],
                "target_weight": [0.30, 0.15],
            }
        )
        covariance = pd.DataFrame(
            {
                "asset_i": ["A", "A", "B", "B"],
                "asset_j": ["A", "B", "A", "B"],
                "covariance_5d": [0.04, 0.01, 0.01, 0.09],
            }
        )
        validator = RiskValidator()
        weight, matrix, input_check = validator.validate_inputs(
            weights,
            covariance,
            date,
            0.45,
            covariance_column="covariance_5d",
            horizon_days=5,
        )
        result = ParametricNormalVarEs([0.95, 0.99]).calculate(weight, matrix)
        output_check = validator.validate_results(result)
        self.assertEqual(input_check["status"], "passed")
        self.assertEqual(output_check["status"], "passed")
        self.assertGreater(result["metrics"]["95%"]["expected_shortfall"], result["metrics"]["95%"]["value_at_risk"])
        self.assertGreater(result["metrics"]["99%"]["value_at_risk"], result["metrics"]["95%"]["value_at_risk"])


if __name__ == "__main__":
    unittest.main()
