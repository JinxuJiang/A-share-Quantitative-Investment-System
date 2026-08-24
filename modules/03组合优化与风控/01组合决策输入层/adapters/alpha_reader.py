from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from adapters.common import read_json, sha256_file


class AlphaReleaseReader:
    REQUIRED_COLUMNS = {
        "signal_date",
        "stock_code",
        "alpha_score",
        "alpha_rank",
        "horizon_days",
    }

    def __init__(self, local_alpha_root: Path):
        self.root = Path(local_alpha_root)
        self.current_path = self.root / "current.json"
        self.manifest_path = self.root / "manifest.json"
        self.data_path = self.root / "stock_alpha.parquet"

    def validate(self) -> dict:
        for path in (self.current_path, self.manifest_path, self.data_path):
            if not path.exists():
                raise FileNotFoundError(f"缺少本地Alpha同步文件: {path}")
        current = read_json(self.current_path)
        manifest = read_json(self.manifest_path)
        if current.get("schema_version") != "stock_alpha_current_v1":
            raise ValueError("Alpha current schema_version不是stock_alpha_current_v1")
        if manifest.get("schema_version") != "stock_alpha_v1":
            raise ValueError("Alpha manifest schema_version不是stock_alpha_v1")
        if current.get("release_id") != manifest.get("release_id"):
            raise ValueError("Alpha current与manifest的release_id不一致")
        actual_hash = sha256_file(self.data_path)
        if actual_hash != manifest.get("output_sha256"):
            raise ValueError("本地Alpha文件SHA256与上游manifest不一致")
        names = set(pq.ParquetFile(self.data_path).schema_arrow.names)
        missing = self.REQUIRED_COLUMNS - names
        if missing:
            raise ValueError(f"Alpha文件缺少字段: {sorted(missing)}")
        return manifest

    def available_dates(self) -> pd.DatetimeIndex:
        values = pq.read_table(self.data_path, columns=["signal_date"])["signal_date"]
        return pd.DatetimeIndex(pd.to_datetime(values.to_pandas()).unique()).sort_values()

    def read_date(self, signal_date: pd.Timestamp) -> pd.DataFrame:
        date = pd.Timestamp(signal_date).normalize()
        table = pq.read_table(
            self.data_path,
            filters=[("signal_date", "=", date.to_pydatetime())],
        )
        frame = table.to_pandas()
        if frame.empty:
            raise ValueError(f"Alpha文件没有信号日: {date.date()}")
        frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
        frame["stock_code"] = frame["stock_code"].astype("string")
        frame = frame.rename(columns={"alpha_rank": "source_alpha_rank"})
        if frame.duplicated(["signal_date", "stock_code"]).any():
            raise ValueError("Alpha截面存在重复的signal_date + stock_code")
        return frame.sort_values("stock_code").reset_index(drop=True)

