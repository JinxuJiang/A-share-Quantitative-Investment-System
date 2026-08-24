from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "01数据"
FACTOR_ENGINE_DIR = PROJECT_ROOT / "02因子库" / "src" / "data_engine"
for path in (DATA_DIR, FACTOR_ENGINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Base_TushareEngine import TushareDataEngine  # noqa: E402
from financial_data_loader import FinancialDataLoader  # noqa: E402
from pit_aligner import PITAligner  # noqa: E402


def test_financial_refresh_preserves_old_versions() -> None:
    original = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20240331"],
            "f_ann_date": ["20240425"],
            "report_type": ["5"],
            "value": [100.0],
            "query_period": ["20240331"],
        }
    )
    refreshed = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20240331", "20240331"],
            "f_ann_date": ["20240425", "20240801"],
            "report_type": ["5", "1"],
            "value": [100.0, 95.0],
            "query_period": ["20240331", "20240331"],
        }
    )
    result = TushareDataEngine._merge_financial_versions(original, refreshed)
    assert result["value"].tolist() == [100.0, 95.0]


def test_same_day_conflict_prefers_adjusted_before_original() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3,
            "end_date": ["20240331"] * 3,
            "ann_date": ["20240425"] * 3,
            "f_ann_date": ["20240425", "20240425", "20240801"],
            "report_type": ["1", "5", "1"],
            "update_flag": ["1", "0", "1"],
            "value": [98.0, 100.0, 95.0],
        }
    )
    result = FinancialDataLoader._select_statement_versions(frame)
    assert result["f_ann_date"].tolist() == ["20240425", "20240801"]
    assert result["report_type"].tolist() == ["5", "1"]
    assert result["value"].tolist() == [100.0, 95.0]


def test_late_type5_archive_does_not_revert_a_revision() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3,
            "end_date": ["20240331"] * 3,
            "ann_date": ["20240425"] * 3,
            "f_ann_date": ["20240425", "20240801", "20240901"],
            "report_type": ["1", "1", "5"],
            "update_flag": ["0", "1", "1"],
            "value": [100.0, 95.0, 100.0],
        }
    )
    result = FinancialDataLoader._select_statement_versions(frame)
    assert result["f_ann_date"].tolist() == ["20240425", "20240801"]
    assert result["value"].tolist() == [100.0, 95.0]


def test_ttm_changes_only_when_revised_annual_value_arrives() -> None:
    records = [
        {"report_date": "20230331", "m_anntime": "20230425", "profit": 10.0},
        {"report_date": "20230630", "m_anntime": "20230825", "profit": 30.0},
        {"report_date": "20230930", "m_anntime": "20231025", "profit": 60.0},
        {"report_date": "20231231", "m_anntime": "20240225", "profit": 100.0},
        {"report_date": "20231231", "m_anntime": "20240801", "profit": 95.0},
    ]
    loader = FinancialDataLoader.__new__(FinancialDataLoader)
    events = loader._build_ttm_events(records, ["profit"])
    finite = [event for event in events if not np.isnan(event["profit_ttm"])]
    assert [(event["m_anntime"], event["profit_ttm"]) for event in finite] == [
        ("20240225", 100.0),
        ("20240801", 95.0),
    ]


def test_revision_activates_on_next_trading_session() -> None:
    calendar = [
        dt.date(2024, 4, 25),
        dt.date(2024, 4, 26),
        dt.date(2024, 8, 1),
        dt.date(2024, 8, 2),
    ]
    events = [
        {"m_anntime": "20240425", "value": 100.0},
        {"m_anntime": "20240801", "value": 95.0},
    ]
    result = PITAligner(calendar).align(events, "m_anntime", ["value"])
    values = {date: value for date, value in result}
    assert np.isnan(values[dt.date(2024, 4, 25)])
    assert values[dt.date(2024, 4, 26)] == 100.0
    assert values[dt.date(2024, 8, 1)] == 100.0
    assert values[dt.date(2024, 8, 2)] == 95.0
