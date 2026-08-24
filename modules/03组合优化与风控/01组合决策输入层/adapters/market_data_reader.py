from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow.parquet as pq


class LocalMarketDataReader:
    def __init__(self, local_data_root: Path):
        self.root = Path(local_data_root)
        self.market_root = self.root / "market_data"
        self.status_root = self.root / "status"
        self.metadata_root = self.root / "metadata"

    @staticmethod
    def _read_wide(path: Path, stock_codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        names = pq.ParquetFile(path).schema_arrow.names
        if stock_codes is None:
            columns = names
        else:
            requested = list(dict.fromkeys(str(code) for code in stock_codes))
            columns = [name for name in names if name == "time" or name in requested]
            missing = sorted(set(requested) - set(columns))
            if missing:
                raise ValueError(f"{path.name}缺少{len(missing)}只股票，示例: {missing[:5]}")
        frame = pq.read_table(path, columns=columns).to_pandas()
        if "time" in frame.columns:
            frame = frame.set_index("time")
        frame.index = pd.to_datetime(frame.index).normalize()
        frame.index.name = "time"
        return frame.sort_index()

    def read_market_field(self, field: str, stock_codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self._read_wide(self.market_root / f"{field}.parquet", stock_codes)

    def read_status(self, name: str, stock_codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self._read_wide(self.status_root / f"{name}.parquet", stock_codes)

    def read_stock_info(self) -> pd.DataFrame:
        path = self.metadata_root / "stock_info.parquet"
        frame = pd.read_parquet(path)
        frame["order_book_id"] = frame["order_book_id"].astype("string")
        return frame

    def read_trade_schedule(self) -> pd.DataFrame:
        path = self.metadata_root / "trade_schedule.parquet"
        frame = pd.read_parquet(path)
        return frame

