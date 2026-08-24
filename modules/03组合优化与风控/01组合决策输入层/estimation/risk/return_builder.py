from __future__ import annotations

import pandas as pd


class ReturnMatrixBuilder:
    def __init__(self, lookback_returns: int = 252):
        self.lookback_returns = int(lookback_returns)

    def build(
        self,
        close: pd.DataFrame,
        stock_codes: list[str],
        signal_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        date = pd.Timestamp(signal_date).normalize()
        missing = sorted(set(stock_codes) - set(close.columns))
        if missing:
            raise ValueError(f"close缺少选中股票: {missing}")
        price_window = close.loc[close.index <= date, stock_codes].tail(self.lookback_returns + 1)
        if len(price_window) < self.lookback_returns + 1:
            raise ValueError(
                f"价格窗口只有{len(price_window)}行，需要{self.lookback_returns + 1}行"
            )
        if price_window.isna().any().any():
            bad = price_window.columns[price_window.isna().any()].tolist()
            raise ValueError(f"选中股票价格窗口存在缺失: {bad}")
        returns = price_window.pct_change(fill_method=None).iloc[1:]
        if len(returns) != self.lookback_returns or returns.isna().any().any():
            raise ValueError("历史收益矩阵不完整")
        return price_window, returns
