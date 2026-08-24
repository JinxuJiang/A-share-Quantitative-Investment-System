from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from adapters.common import read_json, sha256_file


class MarketSignalReleaseReader:
    DECISION_COLUMNS = [
        "signal_date",
        "market_code",
        "forecast_return",
        "signal_zscore",
        "market_state",
        "target_equity_exposure",
        "horizon_days",
    ]
    REQUIRED_COLUMNS = set(DECISION_COLUMNS)

    def __init__(self, local_market_root: Path):
        self.root = Path(local_market_root)
        self.current_path = self.root / "current.json"
        self.manifest_path = self.root / "manifest.json"
        self.data_path = self.root / "market_signal.parquet"

    def validate(self) -> dict:
        for path in (self.current_path, self.manifest_path, self.data_path):
            if not path.exists():
                raise FileNotFoundError(f"缺少本地市场信号同步文件: {path}")
        current = read_json(self.current_path)
        manifest = read_json(self.manifest_path)
        if current.get("schema_version") != "market_signal_current_v1":
            raise ValueError("市场current schema_version不是market_signal_current_v1")
        if manifest.get("schema_version") != "market_signal_v1":
            raise ValueError("市场manifest schema_version不是market_signal_v1")
        if current.get("release_id") != manifest.get("release_id"):
            raise ValueError("市场current与manifest的release_id不一致")
        if sha256_file(self.data_path) != manifest.get("output_sha256"):
            raise ValueError("本地市场信号SHA256与上游manifest不一致")
        names = set(pq.ParquetFile(self.data_path).schema_arrow.names)
        missing = self.REQUIRED_COLUMNS - names
        if missing:
            raise ValueError(f"市场信号缺少字段: {sorted(missing)}")
        return manifest

    def read_all(self) -> pd.DataFrame:
        frame = pq.read_table(self.data_path).to_pandas()
        frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
        return frame.loc[:, self.DECISION_COLUMNS].sort_values("signal_date").reset_index(drop=True)

    def read_date(self, signal_date: pd.Timestamp) -> pd.Series:
        date = pd.Timestamp(signal_date).normalize()
        frame = self.read_all()
        selected = frame.loc[frame["signal_date"].eq(date)]
        if len(selected) != 1:
            raise ValueError(f"市场信号日{date.date()}应有且只有一行，实际{len(selected)}行")
        return selected.iloc[0]
