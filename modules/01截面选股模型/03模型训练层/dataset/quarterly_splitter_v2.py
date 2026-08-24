# -*- coding: utf-8 -*-
"""
Quarterly splitter V2.

Each natural quarter is a fixed prediction period. The model for a quarter is
trained once from data that ends before the quarter starts, with a label gap to
avoid using returns that overlap the prediction period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional
import re

import pandas as pd


@dataclass
class QuarterlySplit:
    fold_id: int
    model_period: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    train_dates: List[pd.Timestamp]
    valid_dates: List[pd.Timestamp]
    pred_dates: List[pd.Timestamp]


class QuarterlySplitterV2:
    """Fixed natural-quarter splitter for PIT signal generation."""

    def __init__(
        self,
        dates: List[pd.Timestamp],
        train_window: str = "3Y",
        valid_window: str = "2M",
        label_horizon: int = 20,
        gap: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.all_dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))
        self.train_window = train_window
        self.valid_window = valid_window
        self.label_horizon = label_horizon
        self.gap = gap if gap is not None else label_horizon + 1
        self.pred_start_date = pd.Timestamp(start_date) if start_date is not None else None
        self.asof_end_date = pd.Timestamp(end_date) if end_date is not None else None

        self.dates = self.all_dates
        if end_date is not None:
            self.dates = self.dates[self.dates <= self.asof_end_date]
        if len(self.dates) == 0:
            raise ValueError("No trading dates after applying end_date.")

        self.train_days = self._window_to_trading_days(train_window)
        self.valid_days = self._window_to_trading_days(valid_window)
        self._splits = self._compute_splits()

    @staticmethod
    def _window_to_trading_days(window: str) -> int:
        match = re.fullmatch(r"(\d+)([YyMmDd])", str(window).strip())
        if not match:
            raise ValueError(f"Invalid window format: {window}; expected e.g. 3Y, 2M, 60D")
        value = int(match.group(1))
        unit = match.group(2).upper()
        if unit == "Y":
            return int(value * 252)
        if unit == "M":
            return int(value * 21)
        return value

    @staticmethod
    def _period_label(date: pd.Timestamp) -> str:
        quarter = ((date.month - 1) // 3) + 1
        return f"{date.year}Q{quarter}"

    def _compute_splits(self) -> List[QuarterlySplit]:
        splits: List[QuarterlySplit] = []
        periods = pd.Series(self.dates, index=self.dates).groupby(self.dates.to_period("Q"))

        for _, period_dates_series in periods:
            pred_dates = list(period_dates_series.values)
            if not pred_dates:
                continue

            period_start = pd.Timestamp(pred_dates[0])
            period_end = pd.Timestamp(pred_dates[-1])
            if self.pred_start_date is not None and period_end < self.pred_start_date:
                continue
            pred_start_idx = self.dates.get_loc(period_start)

            valid_end_idx = pred_start_idx - self.gap
            valid_start_idx = valid_end_idx - self.valid_days
            train_end_idx = valid_start_idx - self.gap
            train_start_idx = train_end_idx - self.train_days

            if train_start_idx < 0 or train_end_idx <= train_start_idx or valid_end_idx <= valid_start_idx:
                continue

            train_dates = self.dates[train_start_idx:train_end_idx].tolist()
            valid_dates = self.dates[valid_start_idx:valid_end_idx].tolist()
            if len(train_dates) == 0 or len(valid_dates) == 0:
                continue

            splits.append(
                QuarterlySplit(
                    fold_id=len(splits),
                    model_period=self._period_label(period_start),
                    period_start=period_start,
                    period_end=period_end,
                    train_dates=train_dates,
                    valid_dates=valid_dates,
                    pred_dates=[pd.Timestamp(d) for d in pred_dates],
                )
            )

        return splits

    def get_splits(self) -> Iterator[QuarterlySplit]:
        yield from self._splits

    def get_n_splits(self) -> int:
        return len(self._splits)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for split in self._splits:
            rows.append(
                {
                    "fold_id": split.fold_id,
                    "model_period": split.model_period,
                    "period_start": split.period_start,
                    "period_end": split.period_end,
                    "train_start": split.train_dates[0],
                    "train_end": split.train_dates[-1],
                    "valid_start": split.valid_dates[0],
                    "valid_end": split.valid_dates[-1],
                    "n_train_dates": len(split.train_dates),
                    "n_valid_dates": len(split.valid_dates),
                    "n_pred_dates": len(split.pred_dates),
                    "gap": self.gap,
                }
            )
        return pd.DataFrame(rows)
