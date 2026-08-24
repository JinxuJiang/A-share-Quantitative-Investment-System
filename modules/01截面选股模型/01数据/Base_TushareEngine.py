# -*- coding: utf-8 -*-
"""
Tushare 数据引擎
================

与 Base_DataEngine.py（QMT）对应的核心数据引擎，一个类承载全部核心逻辑：

- 连接层：Tushare Pro API 限速 / 重试 / 分页
- 元数据：stock_basic、交易日历、申万行业 → stock_info.parquet + industry_map.csv
- 行情：按交易日抓取 daily / adj_factor / daily_basic → 等比前复权
       → market_data/{code}.parquet（与 QMT 旧格式一致）
- 财务：四表全原生字段按季度分区 → financial_full/{table}/{period}.parquet
       （原始层不改字段名、不做 PIT 清洗，PIT 对齐留给因子层）
- 状态：stock_st / suspend_d 事件表 → st_status.parquet + suspend_status.parquet 宽表

存储布局（data_path 默认为 01数据/data/tushare_data）：

    tushare_data/
    ├── market_data/{code}.parquet          # 最终行情（等比前复权）
    ├── financial_full/{table}/{YYYYMMDD}.parquet  # 最终财务（全字段季度分区）
    ├── st_status.parquet                   # ST 状态宽表（0/1/2）
    ├── suspend_status.parquet              # 停牌状态宽表（0/1）
    ├── stock_info.parquet                  # 股票基础信息
    ├── industry_map.csv                    # 申万一级行业映射
    ├── raw/                                # 中间层（支持断点续跑与月度重建）
    │   ├── market/{daily,adj_factor,daily_basic}.parquet
    │   └── metadata/{stock_basic,trade_cal,stock_st,suspend_d,...}.parquet
    └── logs/                               # 校验报告与更新日志
"""

import datetime as dt
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# 常量
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_START_DATE = "20100101"
DEFAULT_FINANCIAL_START_PERIOD = "20100331"

# 财务四表（全原生字段口径，唯一保留的财务口径）
FINANCIAL_TABLE_APIS = {
    "income": "income_vip",
    "balancesheet": "balancesheet_vip",
    "cashflow": "cashflow_vip",
    "fina_indicator": "fina_indicator_vip",
}

FINANCIAL_REQUIRED_COLUMNS = {
    "income": {"ts_code", "ann_date", "f_ann_date", "end_date",
               "report_type", "comp_type", "update_flag"},
    "balancesheet": {"ts_code", "ann_date", "f_ann_date", "end_date",
                     "report_type", "comp_type", "update_flag"},
    "cashflow": {"ts_code", "ann_date", "f_ann_date", "end_date",
                 "report_type", "comp_type", "update_flag"},
    "fina_indicator": {"ts_code", "ann_date", "end_date", "update_flag"},
}

FINANCIAL_MIN_COLUMNS = {
    "income": 80,
    "balancesheet": 145,
    "cashflow": 90,
    "fina_indicator": 100,
}

# 行情字段
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
ADJ_FIELDS = "ts_code,trade_date,adj_factor"
DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
    "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
    "free_share,total_mv,circ_mv"
)
PRICE_COLS = ["open", "high", "low", "close", "pre_close"]

# 行情抓取每 N 个交易日分批落盘一次（中断最多损失一批，内存占用可控）
MARKET_FLUSH_BATCH = 250


@dataclass
class PartitionCheck:
    """财务单分区校验结果"""
    table: str
    period: str
    rows: int
    columns: int
    unique_stocks: int
    duplicate_rows: int
    schema_hash: str
    status: str
    detail: str = ""


class TushareDataEngine:
    """Tushare 数据引擎：连接 + 抓取 + 清洗 + 存储，全部核心逻辑在此。"""

    def __init__(self, data_path=None, token=None):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "tushare_data"
        self.root_path = Path(data_path)

        # 最终输出
        self.market_path = self.root_path / "market_data"
        self.fin_path = self.root_path / "financial_full"
        self.st_file = self.root_path / "st_status.parquet"
        self.suspend_file = self.root_path / "suspend_status.parquet"
        self.stock_info_file = self.root_path / "stock_info.parquet"
        self.industry_map_file = self.root_path / "industry_map.csv"
        self.benchmark_dir = self.root_path / "benchmark"

        # 中间层与日志
        self.raw_market_dir = self.root_path / "raw" / "market"
        self.raw_metadata_dir = self.root_path / "raw" / "metadata"
        self.log_dir = self.root_path / "logs"

        # 连接参数（token 优先级：构造参数 > 环境变量 > 本地非 git 文件）
        self.token = token or self._read_token()
        self.request_interval_sec = float(os.environ.get("TUSHARE_INTERVAL_SEC", "0.15"))
        self.max_retries = int(os.environ.get("TUSHARE_MAX_RETRIES", "3"))
        self.retry_sleep_sec = float(os.environ.get("TUSHARE_RETRY_SLEEP_SEC", "2.0"))

        self._pro = None
        self._last_call_at = 0.0

        self._ensure_directories()

    def _ensure_directories(self):
        for path in [self.market_path, self.fin_path, self.benchmark_dir,
                     self.raw_market_dir, self.raw_metadata_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_token() -> str:
        """读取 Tushare token：环境变量 TUSHARE_TOKEN > 01数据/tushare_token.txt"""
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if token:
            return token
        token_file = Path(__file__).parent / "tushare_token.txt"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                return token
        raise RuntimeError(
            "缺少 Tushare token：请创建 01数据/tushare_token.txt（已 gitignore），"
            "或设置环境变量 TUSHARE_TOKEN"
        )

    # ═════════════════════════════════════════════════════════════════════
    # 连接层：限速 / 重试 / 分页
    # ═════════════════════════════════════════════════════════════════════

    @property
    def pro(self):
        """惰性初始化 Tushare Pro API"""
        if self._pro is None:
            try:
                import tushare as ts
            except ImportError as exc:
                raise RuntimeError(
                    "当前环境未安装 tushare，请先执行: pip install tushare"
                ) from exc
            ts.set_token(self.token)
            self._pro = ts.pro_api()
        return self._pro

    def _pace(self):
        elapsed = time.monotonic() - self._last_call_at
        wait_for = self.request_interval_sec - elapsed
        if wait_for > 0:
            time.sleep(wait_for)

    def call(self, api_name: str, **kwargs: Any) -> pd.DataFrame:
        """调用 Tushare Pro API（带限速与重试）"""
        api_func: Callable[..., pd.DataFrame] = getattr(self.pro, api_name)
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self._pace()
                result = api_func(**kwargs)
                self._last_call_at = time.monotonic()
                if result is None:
                    return pd.DataFrame()
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_sec * attempt)

        raise RuntimeError(
            f"Tushare API 调用失败: {api_name}, kwargs={kwargs}, error={last_error}"
        )

    def call_paged(self, api_name: str, page_size: int = 5000,
                   max_pages: int = 1000, **kwargs: Any) -> pd.DataFrame:
        """分页调用 Tushare Pro API（limit/offset），直到返回短页为止"""
        frames = []
        offset = 0
        for _ in range(max_pages):
            page = self.call(api_name, limit=page_size, offset=offset, **kwargs)
            if page.empty:
                break
            frames.append(page)
            if len(page) < page_size:
                break
            offset += page_size
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ═════════════════════════════════════════════════════════════════════
    # 通用工具
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def is_target_stock_code(series: pd.Series) -> pd.Series:
        """只保留普通沪深股票代码（6位数字.SH/.SZ）"""
        return series.astype(str).str.match(r"^\d{6}\.(SH|SZ)$", na=False)

    @staticmethod
    def _save_frame(df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"saved {path} rows={len(df)}")

    @staticmethod
    def _write_atomic(df: pd.DataFrame, path: Path):
        """原子写入：先写临时文件再替换，防止中断产生半个文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp.parquet")
        df.to_parquet(temp_path, index=False, compression="zstd")
        temp_path.replace(path)

    def read_open_trade_dates(self, start_date: str, end_date: str,
                              max_dates: int | None = None) -> list[str]:
        """从本地交易日历读取开市日期"""
        trade_cal_path = self.raw_metadata_dir / "trade_cal.parquet"
        if not trade_cal_path.exists():
            raise FileNotFoundError(
                f"缺少交易日历: {trade_cal_path}，请先运行 download_metadata()"
            )
        cal = pd.read_parquet(trade_cal_path)
        cal = cal[
            (cal["cal_date"].astype(str) >= start_date)
            & (cal["cal_date"].astype(str) <= end_date)
            & (cal["is_open"].astype(int) == 1)
        ]
        dates = sorted(cal["cal_date"].astype(str).tolist())
        if max_dates is not None:
            dates = dates[:max_dates]
        return dates

    def get_stock_list(self) -> list[str]:
        """从本地 stock_basic 获取沪深 A 股代码列表（含已退市）"""
        stock_basic_path = self.raw_metadata_dir / "stock_basic.parquet"
        if not stock_basic_path.exists():
            raise FileNotFoundError(
                f"缺少股票列表: {stock_basic_path}，请先运行 download_metadata()"
            )
        stock_basic = pd.read_parquet(stock_basic_path)
        stock_basic = stock_basic[self.is_target_stock_code(stock_basic["ts_code"])]
        return sorted(stock_basic["ts_code"].dropna().astype(str).unique().tolist())

    # ═════════════════════════════════════════════════════════════════════
    # 元数据：stock_basic / 交易日历 / 申万行业 → stock_info + industry_map
    # ═════════════════════════════════════════════════════════════════════

    def download_metadata(self, start_date: str = DEFAULT_START_DATE,
                          end_date: str | None = None):
        """抓取全部元数据原表，并构建 stock_info.parquet 与 industry_map.csv"""
        if end_date is None:
            end_date = dt.datetime.now().strftime("%Y%m%d")

        print("📁 抓取元数据原表...")
        self._save_frame(self._fetch_stock_basic(),
                         self.raw_metadata_dir / "stock_basic.parquet")
        self._save_frame(
            self.call("trade_cal", exchange="", start_date=start_date, end_date=end_date,
                      fields="exchange,cal_date,is_open,pretrade_date"),
            self.raw_metadata_dir / "trade_cal.parquet")
        # 单独保存当年完整交易安排，供月末信号识别使用。trade_cal 仍只到
        # end_date，避免状态宽表提前出现尚未发生的交易日。
        schedule_end = f"{end_date[:4]}1231"
        self._save_frame(
            self.call("trade_cal", exchange="", start_date=start_date, end_date=schedule_end,
                      fields="exchange,cal_date,is_open,pretrade_date"),
            self.raw_metadata_dir / "trade_schedule.parquet")
        self._save_frame(
            self.call_paged("namechange",
                            fields="ts_code,name,start_date,end_date,ann_date,change_reason"),
            self.raw_metadata_dir / "namechange.parquet")
        self._save_frame(self._fetch_stock_company(),
                         self.raw_metadata_dir / "stock_company.parquet")
        index_classify = self._fetch_index_classify()
        self._save_frame(index_classify,
                         self.raw_metadata_dir / "index_classify_sw2021.parquet")
        self._save_frame(self._fetch_index_member_all(index_classify),
                         self.raw_metadata_dir / "index_member_all_sw2021.parquet")

        print("\n📁 构建 stock_info / industry_map ...")
        self.build_stock_info()
        self.build_industry_map()
        print("✅ 元数据完成")

    def download_benchmark_index(self, start_date: str = DEFAULT_START_DATE,
                                 end_date: str | None = None,
                                 ts_code: str = "000852.SH") -> Path:
        """下载并原子保存基准指数日行情（默认中证1000）。"""
        if end_date is None:
            end_date = dt.datetime.now().strftime("%Y%m%d")
        fields = (
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        )
        data = self.call(
            "index_daily",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )
        if data.empty:
            raise RuntimeError(f"基准指数无数据: {ts_code} {start_date}~{end_date}")
        data = (
            data.drop_duplicates(["ts_code", "trade_date"], keep="last")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        path = self.benchmark_dir / f"{ts_code}.parquet"
        self._write_atomic(data, path)
        print(f"saved {path} rows={len(data)}")
        return path

    def _fetch_stock_basic(self) -> pd.DataFrame:
        fields = ("ts_code,symbol,name,area,industry,fullname,enname,cnspell,"
                  "market,exchange,curr_type,list_status,list_date,delist_date,is_hs")
        frames = []
        for status in ["L", "D", "P"]:  # 上市 / 退市 / 暂停上市
            df = self.call("stock_basic", exchange="", list_status=status, fields=fields)
            if not df.empty:
                df["query_list_status"] = status
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")

    def _fetch_stock_company(self) -> pd.DataFrame:
        fields = ("ts_code,chairman,manager,secretary,reg_capital,setup_date,province,"
                  "city,website,email,employees,main_business,business_scope")
        frames = []
        for exchange in ["SSE", "SZSE", "BSE"]:
            df = self.call("stock_company", exchange=exchange, fields=fields)
            if not df.empty:
                df["query_exchange"] = exchange
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="last")

    def _fetch_index_classify(self) -> pd.DataFrame:
        fields = "index_code,industry_name,level,industry_code,is_pub,parent_code,src"
        frames = []
        for level in ["L1", "L2", "L3"]:
            df = self.call("index_classify", level=level, src="SW2021", fields=fields)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(
            ["index_code", "level"], keep="last")

    def _fetch_index_member_all(self, index_classify: pd.DataFrame) -> pd.DataFrame:
        fields = ("l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
                  "ts_code,name,in_date,out_date,is_new")
        if index_classify.empty:
            return self.call_paged("index_member_all", fields=fields, is_new="Y")
        l1_codes = (
            index_classify.loc[index_classify["level"].astype(str) == "L1", "index_code"]
            .dropna().astype(str).sort_values().unique().tolist()
        )
        frames = []
        for l1_code in l1_codes:
            df = self.call_paged("index_member_all", fields=fields,
                                 l1_code=l1_code, is_new="Y")
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(
            ["ts_code", "l1_code", "l2_code", "l3_code"], keep="last")

    def _read_raw_metadata(self, name: str) -> pd.DataFrame:
        path = self.raw_metadata_dir / name
        if not path.exists():
            raise FileNotFoundError(f"缺少元数据原表: {path}")
        return pd.read_parquet(path)

    def build_stock_info(self) -> Path:
        """构建 stock_info.parquet（股票基础信息，字段对齐旧 QMT 口径）"""
        stock_basic = self._read_raw_metadata("stock_basic.parquet")
        company_path = self.raw_metadata_dir / "stock_company.parquet"
        stock_company = pd.read_parquet(company_path) if company_path.exists() else pd.DataFrame()

        out = stock_basic[self.is_target_stock_code(stock_basic["ts_code"])].copy()
        out = out.rename(columns={
            "ts_code": "order_book_id",
            "symbol": "stock_symbol",
            "name": "symbol",
            "fullname": "listed_company_name",
        })
        if not stock_company.empty:
            company = stock_company.rename(columns={"ts_code": "order_book_id"})
            keep_cols = [c for c in ["order_book_id", "province", "city", "reg_capital",
                                     "setup_date", "employees", "main_business",
                                     "business_scope"] if c in company.columns]
            out = out.merge(company[keep_cols], on="order_book_id", how="left")

        out.to_parquet(self.stock_info_file, index=False)
        print(f"saved {self.stock_info_file} rows={len(out)}")
        return self.stock_info_file

    def build_industry_map(self) -> Path:
        """构建 industry_map.csv（申万一级行业映射，对齐旧 QMT 口径）"""
        member = self._read_raw_metadata("index_member_all_sw2021.parquet")
        stock_basic = self._read_raw_metadata("stock_basic.parquet")
        target_codes = set(
            stock_basic.loc[self.is_target_stock_code(stock_basic["ts_code"]), "ts_code"]
            .astype(str)
        )
        current = member.copy()
        if "is_new" in current.columns:
            current = current[current["is_new"].astype(str).isin(["1", "Y", "True", "true"])]
        elif "out_date" in current.columns:
            current = current[current["out_date"].isna() | (current["out_date"].astype(str) == "")]
        if current.empty:
            current = member.copy()

        industry_col = "l1_name" if "l1_name" in current.columns else "industry_name"
        current = current[
            self.is_target_stock_code(current["ts_code"])
            & current["ts_code"].astype(str).isin(target_codes)
        ].copy()
        out = current[["ts_code", industry_col]].copy()
        out = out.rename(columns={"ts_code": "order_book_id", industry_col: "industry_name"})
        out = out.dropna(subset=["order_book_id"]).drop_duplicates("order_book_id", keep="last")

        out.to_csv(self.industry_map_file, index=False, encoding="utf-8-sig")
        print(f"saved {self.industry_map_file} rows={len(out)}")
        return self.industry_map_file

    # ═════════════════════════════════════════════════════════════════════
    # 行情：按交易日抓取 → 等比前复权 → market_data/{code}.parquet
    # ═════════════════════════════════════════════════════════════════════

    def download_market_data(self, start_date: str = DEFAULT_START_DATE,
                             end_date: str | None = None,
                             missing_only: bool = False,
                             skip_daily_basic: bool = False,
                             build: bool = True,
                             max_dates: int | None = None):
        """
        下载行情数据（两步）

        第一步：按交易日抓取 daily / adj_factor / daily_basic 存入 raw/（断点续跑）
        第二步：等比前复权后构建 market_data/{code}.parquet

        参数:
        -----
        start_date / end_date : str, 'YYYYMMDD'，end_date 默认今天
        missing_only : 只抓取本地 raw 缺失的交易日（月度增量用）
        skip_daily_basic : 跳过 daily_basic（调试用）
        build : 抓取完成后是否重建 per-stock 文件
        max_dates : 调试参数，只取前 N 个交易日
        """
        if end_date is None:
            end_date = dt.datetime.now().strftime("%Y%m%d")

        trade_dates = self.read_open_trade_dates(start_date, end_date, max_dates)
        if not trade_dates:
            raise RuntimeError("指定区间内没有开市交易日")

        datasets = [("daily", "daily", DAILY_FIELDS), ("adj_factor", "adj_factor", ADJ_FIELDS)]
        if not skip_daily_basic:
            datasets.append(("daily_basic", "daily_basic", DAILY_BASIC_FIELDS))

        for name, api_name, fields in datasets:
            dates = trade_dates
            if missing_only:
                known = self._market_dataset_dates(name)
                dates = [d for d in trade_dates if d not in known]
            if dates:
                print(f"{name} range: {dates[0]} ~ {dates[-1]} ({len(dates)} open days)")
            else:
                print(f"{name}: no missing dates")
            self._fetch_market_dataset(name, api_name, dates, fields)

        if build:
            self.build_market_files(start_date, end_date)

    def _fetch_market_dataset(self, name: str, api_name: str,
                              trade_dates: list[str], fields: str):
        """按交易日抓取，每 MARKET_FLUSH_BATCH 天原子落盘一个分片

        中断最多损失当前一批；内存占用与全历史规模无关。
        """
        frames = []
        batch_dates = []
        total = len(trade_dates)
        for i, trade_date in enumerate(trade_dates, 1):
            df = self.call(api_name, trade_date=trade_date, fields=fields)
            if not df.empty:
                frames.append(df)
            batch_dates.append(trade_date)
            if len(batch_dates) >= MARKET_FLUSH_BATCH or i == total:
                if frames:
                    self._save_market_shard(name, pd.concat(frames, ignore_index=True),
                                            batch_dates)
                frames, batch_dates = [], []
            if i % 20 == 0 or i == total:
                print(f"{api_name}: {i}/{total} dates")

    def _market_shard_dir(self, name: str) -> Path:
        return self.raw_market_dir / f"{name}_shards"

    def _save_market_shard(self, name: str, df: pd.DataFrame,
                           batch_dates: list[str]) -> Path:
        """原子写入一个行情分片，分片按日期区间命名（重跑同区间安全覆盖）"""
        key_cols = [c for c in ["ts_code", "trade_date"] if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(key_cols, keep="last")
        shard = self._market_shard_dir(name) / f"{min(batch_dates)}_{max(batch_dates)}.parquet"
        self._write_atomic(df, shard)
        print(f"saved shard {shard.name} rows={len(df)}")
        return shard

    def _market_dataset_dates(self, name: str) -> set[str]:
        """数据集已覆盖的交易日（旧单文件 + 新分片目录的并集）"""
        dates = set()
        legacy = self.raw_market_dir / f"{name}.parquet"
        if legacy.exists():
            df = pd.read_parquet(legacy, columns=["trade_date"])
            dates |= set(df["trade_date"].dropna().astype(str).unique().tolist())
        shard_dir = self._market_shard_dir(name)
        if shard_dir.exists():
            for shard in sorted(shard_dir.glob("*.parquet")):
                df = pd.read_parquet(shard, columns=["trade_date"])
                dates |= set(df["trade_date"].dropna().astype(str).unique().tolist())
        return dates

    def _read_market_dataset(self, name: str) -> pd.DataFrame:
        """读取完整行情数据集（旧单文件 + 新分片合并去重）"""
        frames = []
        legacy = self.raw_market_dir / f"{name}.parquet"
        if legacy.exists():
            frames.append(pd.read_parquet(legacy))
        shard_dir = self._market_shard_dir(name)
        if shard_dir.exists():
            for shard in sorted(shard_dir.glob("*.parquet")):
                frames.append(pd.read_parquet(shard))
        if not frames:
            raise FileNotFoundError(
                f"缺少行情数据集 {name}，请先运行 download_market_data()"
            )
        df = pd.concat(frames, ignore_index=True)
        key_cols = [c for c in ["ts_code", "trade_date"] if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(key_cols, keep="last")
            df = df.sort_values(key_cols).reset_index(drop=True)
        return df

    # ── 第二步：构建 per-stock 文件 ────────────────────────────────────────

    @staticmethod
    def _parse_date_col(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")

    @staticmethod
    def _local_midnight_ms(value) -> int:
        """本地午夜的 epoch 毫秒（对齐旧 MarketDataLoader 的 time 字段）"""
        date_value = value.date() if isinstance(value, pd.Timestamp) else value
        local_dt = dt.datetime.combine(date_value, dt.time.min)
        return int(local_dt.timestamp() * 1000)

    def _prepare_daily(self, daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
        """等比前复权：qfq_ratio = adj_factor / 最新 adj_factor，避免传统前复权负数问题"""
        df = daily.copy()
        factors = adj.copy()
        df["trade_dt"] = self._parse_date_col(df["trade_date"])
        factors["trade_dt"] = self._parse_date_col(factors["trade_date"])

        df = df.merge(factors[["ts_code", "trade_date", "adj_factor"]],
                      on=["ts_code", "trade_date"], how="left")
        latest_factor = (
            factors.dropna(subset=["adj_factor"])
            .sort_values(["ts_code", "trade_dt"])
            .groupby("ts_code", observed=True)["adj_factor"].last()
        )
        df["latest_adj_factor"] = df["ts_code"].map(latest_factor)
        df["qfq_ratio"] = df["adj_factor"] / df["latest_adj_factor"]

        for col in PRICE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce") * df["qfq_ratio"]

        df["volume"] = pd.to_numeric(df["vol"], errors="coerce") * 100.0   # 手 → 股
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000.0  # 千元 → 元
        df = df.rename(columns={"pre_close": "preClose", "ts_code": "stock_code"})
        return df

    def _build_one_stock(self, stock_code: str, stock_row: pd.Series,
                         stock_df: pd.DataFrame | None,
                         trade_days: pd.DatetimeIndex) -> pd.DataFrame | None:
        if trade_days.empty or stock_df is None or stock_df.empty:
            return None
        bounds = self._stock_date_bounds(stock_row, trade_days.min(), trade_days.max())
        if bounds is None:
            return None
        left, right = bounds
        stock_days = trade_days[(trade_days >= left) & (trade_days <= right)]
        if stock_days.empty:
            return None

        stock_df = stock_df.sort_values("trade_dt").set_index("trade_dt")

        out = pd.DataFrame(index=stock_days)
        out = out.join(stock_df[["open", "high", "low", "close", "volume", "amount", "preClose"]])
        out["suspendFlag"] = out["close"].isna().astype(np.int64)

        for col in ["open", "high", "low", "close", "preClose"]:
            out[col] = out[col].ffill()
        out["volume"] = out["volume"].fillna(0).round().astype(np.int64)
        out["amount"] = out["amount"].fillna(0.0)

        if out["close"].isna().all():
            return None

        out = out.reset_index(names="trade_dt")
        out["time"] = out["trade_dt"].map(self._local_midnight_ms).astype(np.int64)
        return out[["time", "open", "high", "low", "close",
                    "volume", "amount", "preClose", "suspendFlag"]]

    @staticmethod
    def _stock_date_bounds(stock_row: pd.Series, start: pd.Timestamp,
                           end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        list_date = pd.to_datetime(str(stock_row.get("list_date", "")),
                                   format="%Y%m%d", errors="coerce")
        delist_date = pd.to_datetime(str(stock_row.get("delist_date", "")),
                                     format="%Y%m%d", errors="coerce")
        left = max(start, list_date) if not pd.isna(list_date) else start
        right = min(end, delist_date) if not pd.isna(delist_date) else end
        if pd.isna(left) or pd.isna(right) or left > right:
            return None
        return left, right

    def build_market_files(self, start_date: str = DEFAULT_START_DATE,
                           end_date: str | None = None,
                           stock_codes: list[str] | None = None,
                           max_stocks: int | None = None):
        """
        从 raw 行情数据构建 market_data/{code}.parquet（全量重建）

        注意：等比前复权因子随分红除权漂移，每周更新时需要全量重建，
        与旧 QMT 流程"行情每月全量覆盖"的原因一致。
        """
        if end_date is None:
            end_date = dt.datetime.now().strftime("%Y%m%d")

        daily = self._prepare_daily(self._read_market_dataset("daily"),
                                    self._read_market_dataset("adj_factor"))
        trade_days = pd.DatetimeIndex(
            self._parse_date_col(pd.Series(self.read_open_trade_dates(start_date, end_date)))
            .dropna().sort_values().unique()
        )
        stock_basic = self._read_raw_metadata("stock_basic.parquet")

        stocks = stock_basic[self.is_target_stock_code(stock_basic["ts_code"])].copy()
        if stock_codes:
            stocks = stocks[stocks["ts_code"].isin(stock_codes)]
        else:
            stocks = stocks[stocks["list_status"].astype(str).isin(["L", "D"])]
        stocks = stocks.drop_duplicates("ts_code", keep="first").sort_values("ts_code")
        if max_stocks is not None:
            stocks = stocks.head(max_stocks)

        # 一次性按股票分组，避免每只股票全表扫描（全历史 2000万+ 行时必须）
        daily_by_code = {code: g for code, g in daily.groupby("stock_code", sort=False)}

        self.market_path.mkdir(parents=True, exist_ok=True)
        written, skipped = 0, 0
        for i, row in enumerate(stocks.itertuples(index=False), 1):
            stock_row = pd.Series(row._asdict())
            stock_code = stock_row["ts_code"]
            out = self._build_one_stock(stock_code, stock_row,
                                        daily_by_code.get(stock_code), trade_days)
            if out is None or out.empty:
                skipped += 1
                continue
            out.to_parquet(self.market_path / f"{stock_code}.parquet", index=False)
            written += 1
            if i % 200 == 0 or i == len(stocks):
                print(f"progress {i}/{len(stocks)} written={written} skipped={skipped}")

        print(f"done written={written} skipped={skipped} output={self.market_path}")

    # ═════════════════════════════════════════════════════════════════════
    # 财务：四表全原生字段，按季度分区（唯一保留的财务口径）
    # 原始层不改字段名、不做 QMT 兼容映射、不执行 PIT 清洗
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def quarter_periods(start_period: str, end_period: str,
                        max_periods: int | None = None) -> list[str]:
        start_year = int(start_period[:4])
        end_year = int(end_period[:4])
        periods = [
            f"{year}{suffix}"
            for year in range(start_year, end_year + 1)
            for suffix in ("0331", "0630", "0930", "1231")
            if start_period <= f"{year}{suffix}" <= end_period
        ]
        return periods[:max_periods] if max_periods is not None else periods

    @staticmethod
    def latest_quarter_period(reference: str | None = None) -> str:
        """最近一个已到来的季度报告期（避免把未来季度纳入抓取/校验范围）"""
        ref = reference or dt.datetime.now().strftime("%Y%m%d")
        periods = TushareDataEngine.quarter_periods(
            DEFAULT_FINANCIAL_START_PERIOD, ref)
        if not periods:
            raise RuntimeError(f"{ref} 之前没有可用的季度报告期")
        return periods[-1]

    def financial_partition_path(self, table: str, period: str) -> Path:
        return self.fin_path / table / f"{period}.parquet"

    @staticmethod
    def _merge_financial_versions(
        existing: pd.DataFrame,
        refreshed: pd.DataFrame,
    ) -> pd.DataFrame:
        """Append newly observed versions without deleting prior vendor rows."""
        return (
            pd.concat([existing, refreshed], ignore_index=True, sort=False)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def download_financial_data(self,
                                start_period: str = DEFAULT_FINANCIAL_START_PERIOD,
                                end_period: str | None = None,
                                tables: list[str] | None = None,
                                max_periods: int | None = None,
                                overwrite: bool = False):
        """
        下载财务数据：四表全原生字段，按 表/季度 分区原子写入，断点续跑

        参数:
        -----
        start_period / end_period : str, 'YYYYMMDD' 季度末日期（如 20100331）
        tables : 默认全部四表
        overwrite : 重新抓取已有分区，并与旧版本追加合并而非替换
            （披露季内刷新未完整季度用；旧版本行保留，保证 PIT 可复现）
        """
        if end_period is None:
            end_period = self.latest_quarter_period()
        tables = tables or list(FINANCIAL_TABLE_APIS)
        periods = self.quarter_periods(start_period, end_period, max_periods)
        if not periods:
            raise RuntimeError("指定区间内没有季度报告期")

        total = len(tables) * len(periods)
        completed = 0
        for table in tables:
            for period in periods:
                self._fetch_financial_partition(table, period, overwrite)
                completed += 1
                print(f"progress {completed}/{total}")

    def _fetch_financial_partition(self, table: str, period: str,
                                   overwrite: bool) -> Path:
        path = self.financial_partition_path(table, period)
        if path.exists() and not overwrite:
            print(f"skip existing {table}/{period}")
            return path

        api_name = FINANCIAL_TABLE_APIS[table]
        if table in {"income", "balancesheet", "cashflow"}:
            version_frames = []
            version_counts = {}
            for report_type in ("1", "5"):
                version_df = self.call_paged(
                    api_name,
                    page_size=5000,
                    period=period,
                    report_type=report_type,
                )
                if not version_df.empty:
                    actual_type = (
                        version_df["report_type"]
                        .astype(str)
                        .str.replace(r"\.0$", "", regex=True)
                    )
                    version_df = version_df.loc[actual_type.eq(report_type)].copy()
                version_counts[report_type] = len(version_df)
                if not version_df.empty:
                    version_frames.append(version_df)

            if version_counts.get("1", 0) == 0:
                raise RuntimeError(
                    f"{api_name} 在 {period} 的 report_type=1 返回空数据"
                )
            df = pd.concat(version_frames, ignore_index=True, sort=False)
        else:
            df = self.call_paged(api_name, page_size=5000, period=period)
            version_counts = {}

        if df.empty:
            raise RuntimeError(f"{api_name} 在 {period} 返回空数据")

        missing = sorted(FINANCIAL_REQUIRED_COLUMNS[table] - set(df.columns))
        if missing:
            raise RuntimeError(f"{table}/{period} 缺少必要列: {missing}")
        if len(df.columns) < FINANCIAL_MIN_COLUMNS[table]:
            raise RuntimeError(
                f"{table}/{period} 列数异常: {len(df.columns)} < {FINANCIAL_MIN_COLUMNS[table]}"
            )

        df = df.drop_duplicates().copy()
        df["query_period"] = period
        if path.exists():
            existing = pd.read_parquet(path)
            # Financial refreshes are append-only at the version level. A later
            # f_ann_date or an adjusted-before row must not erase a version that
            # was used by an earlier point-in-time date.
            df = self._merge_financial_versions(existing, df)
        sort_cols = [c for c in ["ts_code", "end_date", "f_ann_date", "ann_date",
                                 "report_type", "update_flag"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        self._write_atomic(df, path)
        version_detail = ""
        if version_counts:
            version_detail = (
                f" type1={version_counts.get('1', 0)}"
                f" type5={version_counts.get('5', 0)}"
            )
        print(f"saved {table}/{period}: rows={len(df)} "
              f"columns={len(df.columns)} stocks={df['ts_code'].nunique()}"
              f"{version_detail}")
        return path

    # ── 财务校验 ──────────────────────────────────────────────────────────

    @staticmethod
    def _schema_hash(columns: list[str]) -> str:
        return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()[:16]

    def _inspect_financial_partition(self, table: str, period: str) -> PartitionCheck:
        path = self.financial_partition_path(table, period)
        if not path.exists():
            return PartitionCheck(table, period, 0, 0, 0, 0, "", "FAIL", f"missing {path}")

        df = pd.read_parquet(path)
        missing = sorted(FINANCIAL_REQUIRED_COLUMNS[table] - set(df.columns))
        wrong_period = 0
        if "end_date" in df.columns:
            wrong_period = int((df["end_date"].astype(str) != period).sum())
        duplicate_rows = int(df.duplicated().sum())

        problems = []
        if df.empty:
            problems.append("empty")
        if missing:
            problems.append(f"missing_columns={missing}")
        if len(df.columns) < FINANCIAL_MIN_COLUMNS[table]:
            problems.append(f"column_count={len(df.columns)}<{FINANCIAL_MIN_COLUMNS[table]}")
        if wrong_period:
            problems.append(f"wrong_end_date_rows={wrong_period}")
        if duplicate_rows:
            problems.append(f"duplicate_rows={duplicate_rows}")

        return PartitionCheck(
            table=table, period=period, rows=len(df), columns=len(df.columns),
            unique_stocks=int(df["ts_code"].nunique()) if "ts_code" in df.columns else 0,
            duplicate_rows=duplicate_rows,
            schema_hash=self._schema_hash(df.columns.tolist()),
            status="FAIL" if problems else "PASS",
            detail="; ".join(problems),
        )

    def validate_financial_data(self,
                                start_period: str = DEFAULT_FINANCIAL_START_PERIOD,
                                end_period: str | None = None,
                                tables: list[str] | None = None,
                                max_periods: int | None = None) -> list[PartitionCheck]:
        """
        校验财务分区：schema 一致性 / 必要列 / 重复行 / 披露季完整性
        报告保存到 logs/financial_full_validation.json
        """
        if end_period is None:
            end_period = self.latest_quarter_period()
        tables = tables or list(FINANCIAL_TABLE_APIS)
        periods = self.quarter_periods(start_period, end_period, max_periods)

        results = []
        for table in tables:
            expected_schema = None
            table_results = []
            for period in periods:
                result = self._inspect_financial_partition(table, period)
                if result.status == "PASS":
                    if expected_schema is None:
                        expected_schema = result.schema_hash
                    elif result.schema_hash != expected_schema:
                        result.status = "FAIL"
                        result.detail = (
                            f"{result.detail}; " if result.detail else ""
                        ) + f"schema_hash={result.schema_hash}, expected={expected_schema}"
                table_results.append(result)

            # 披露季内的分区结构合法但行数可能远低于正常水平，标 WARN 而非 FAIL
            for i, result in enumerate(table_results):
                prior_rows = [item.rows for item in table_results[max(0, i - 4):i]
                              if item.status != "FAIL" and item.rows > 0]
                if result.status == "PASS" and len(prior_rows) >= 2:
                    baseline = float(pd.Series(prior_rows).median())
                    if baseline > 0 and result.rows < baseline * 0.2:
                        result.status = "WARN"
                        result.detail = (
                            f"possible incomplete disclosure period: rows={result.rows}, "
                            f"recent_median={baseline:.0f}"
                        )

            for result in table_results:
                results.append(result)
                print(f"[{result.status}] {table}/{result.period}: rows={result.rows} "
                      f"columns={result.columns} stocks={result.unique_stocks} {result.detail}")

        report_path = self.log_dir / "financial_full_validation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"validation report: {report_path}")
        return results

    # ═════════════════════════════════════════════════════════════════════
    # 状态：stock_st / suspend_d 事件表 → ST/停牌宽表
    # ═════════════════════════════════════════════════════════════════════

    @property
    def _status_fetch_log_path(self) -> Path:
        return self.raw_metadata_dir / "status_fetch_log.parquet"

    def download_status_data(self, start_date: str = DEFAULT_START_DATE,
                             end_date: str | None = None,
                             missing_only: bool = True,
                             tables: list[str] | None = None,
                             build: bool = True,
                             max_dates: int | None = None):
        """
        下载 ST / 停牌状态数据

        stock_st 和 suspend_d 是事件表：某天无事件则该天无行，
        因此用 fetch_log 记录"已请求过的日期"，missing_only 时不会重复请求。

        参数:
        -----
        tables : 默认 ["stock_st", "suspend_d"]
        build : 抓取完成后是否重建 st_status / suspend_status 宽表
        """
        if end_date is None:
            end_date = dt.datetime.now().strftime("%Y%m%d")
        tables = tables or ["stock_st", "suspend_d"]

        trade_dates = self.read_open_trade_dates(start_date, end_date, max_dates)
        if not trade_dates:
            raise RuntimeError("指定区间内没有开市交易日")
        print(f"date range: {trade_dates[0]} ~ {trade_dates[-1]} ({len(trade_dates)} open days)")

        if "stock_st" in tables:
            dates = trade_dates
            if missing_only:
                known = (self._status_requested_dates("stock_st")
                         or self._existing_status_dates("stock_st", "trade_date"))
                dates = [d for d in trade_dates if d not in known]
            print(f"stock_st: {len(dates)} days to fetch" if dates else "stock_st: no missing dates")
            stock_st = self._fetch_stock_st(dates)
            self._save_status_dataset("stock_st", stock_st)
            self._save_status_fetch_log("stock_st", dates, stock_st)

        if "suspend_d" in tables:
            dates = trade_dates
            if missing_only:
                known = (self._status_requested_dates("suspend_d")
                         or self._existing_status_dates("suspend_d", "trade_date")
                         or self._existing_status_dates("suspend_d", "suspend_date"))
                dates = [d for d in trade_dates if d not in known]
            print(f"suspend_d: {len(dates)} days to fetch" if dates else "suspend_d: no missing dates")
            suspend_d = self._fetch_suspend_d(dates)
            self._save_status_dataset("suspend_d", suspend_d)
            self._save_status_fetch_log("suspend_d", dates, suspend_d)

        if build:
            self.build_status_files(tables=tables)

    def _fetch_stock_st(self, trade_dates: list[str]) -> pd.DataFrame:
        frames = []
        fields = "ts_code,name,trade_date,type,type_name"
        total = len(trade_dates)
        for i, trade_date in enumerate(trade_dates, 1):
            df = self.call("stock_st", trade_date=trade_date, fields=fields)
            if not df.empty:
                frames.append(df)
            if i % 20 == 0 or i == total:
                print(f"stock_st: {i}/{total} dates")
        if not frames:
            return pd.DataFrame(columns=["ts_code", "name", "trade_date", "type", "type_name"])
        return pd.concat(frames, ignore_index=True)

    def _fetch_suspend_d(self, trade_dates: list[str]) -> pd.DataFrame:
        frames = []
        total = len(trade_dates)
        for i, trade_date in enumerate(trade_dates, 1):
            df = self.call("suspend_d", trade_date=trade_date)
            if not df.empty:
                frames.append(df)
            if i % 20 == 0 or i == total:
                print(f"suspend_d: {i}/{total} dates")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _save_status_dataset(self, name: str, df: pd.DataFrame) -> Path:
        path = self.raw_metadata_dir / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df], ignore_index=True)
        if not df.empty:
            if name == "suspend_d":
                key_cols = [c for c in ["ts_code", "suspend_date", "resume_date", "ann_date"]
                            if c in df.columns]
            else:
                key_cols = [c for c in ["ts_code", "trade_date", "type"] if c in df.columns]
            if key_cols:
                df = df.drop_duplicates(key_cols, keep="last")
                df = df.sort_values(key_cols).reset_index(drop=True)
        df.to_parquet(path, index=False)
        print(f"saved {path} rows={len(df)}")
        return path

    def _save_status_fetch_log(self, table: str, trade_dates: list[str], df: pd.DataFrame):
        """记录已请求的日期（包括返回空的事件日），保证 missing_only 不重复请求"""
        if not trade_dates:
            return
        date_col = "trade_date"
        if table == "suspend_d" and "trade_date" not in df.columns and "suspend_date" in df.columns:
            date_col = "suspend_date"
        if df.empty or date_col not in df.columns:
            row_counts = pd.Series(dtype="int64")
        else:
            row_counts = df[date_col].dropna().astype(str).value_counts()

        log = pd.DataFrame({
            "table": table,
            "trade_date": trade_dates,
            "rows": [int(row_counts.get(d, 0)) for d in trade_dates],
            "fetched_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        if self._status_fetch_log_path.exists():
            old = pd.read_parquet(self._status_fetch_log_path)
            log = pd.concat([old, log], ignore_index=True)
        log = (log.drop_duplicates(["table", "trade_date"], keep="last")
               .sort_values(["table", "trade_date"]).reset_index(drop=True))
        log.to_parquet(self._status_fetch_log_path, index=False)

    def _status_requested_dates(self, table: str) -> set[str]:
        if not self._status_fetch_log_path.exists():
            return set()
        log = pd.read_parquet(self._status_fetch_log_path)
        if log.empty or "table" not in log.columns:
            return set()
        log = log[log["table"].astype(str) == table]
        return set(log["trade_date"].dropna().astype(str).tolist())

    def _existing_status_dates(self, name: str, date_col: str) -> set[str]:
        path = self.raw_metadata_dir / f"{name}.parquet"
        if not path.exists():
            return set()
        df = pd.read_parquet(path)
        if date_col not in df.columns:
            return set()
        return set(df[date_col].dropna().astype(str).unique().tolist())

    # ── 宽表构建 ──────────────────────────────────────────────────────────

    def build_status_files(self, tables: list[str] | None = None):
        """从事件表构建 st_status.parquet（0/1/2）与 suspend_status.parquet（0/1）"""
        tables = tables or ["stock_st", "suspend_d"]
        if "stock_st" in tables:
            self._build_st_status()
        if "suspend_d" in tables:
            self._build_suspend_status()

    def _status_base_frame(self) -> pd.DataFrame:
        """行=全部开市日，列=全部沪深股票，值=0 的基底宽表"""
        trade_cal = self._read_raw_metadata("trade_cal.parquet")
        stock_basic = self._read_raw_metadata("stock_basic.parquet")
        open_days = trade_cal[trade_cal["is_open"].astype(int) == 1]
        dates = pd.to_datetime(open_days["cal_date"].astype(str), format="%Y%m%d")
        # 索引必须排序且去重，下游按时间顺序读取
        dates = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
        stock_basic = stock_basic[self.is_target_stock_code(stock_basic["ts_code"])]
        stocks = sorted(stock_basic["ts_code"].dropna().astype(str).unique().tolist())
        return pd.DataFrame(0, index=dates, columns=stocks, dtype=np.int8)

    def _build_st_status(self) -> Path | None:
        stock_st_path = self.raw_metadata_dir / "stock_st.parquet"
        if not stock_st_path.exists():
            print("skip st_status: 未找到 stock_st.parquet")
            return None

        status = self._status_base_frame()
        stock_st = pd.read_parquet(stock_st_path)
        if not stock_st.empty:
            data = stock_st.copy()
            data["time"] = pd.to_datetime(data["trade_date"].astype(str), format="%Y%m%d")
            if "type_name" not in data.columns:
                data["type_name"] = ""
            # *ST = 2，ST = 1
            data["status_value"] = np.where(
                data["type_name"].astype(str).str.contains(r"\*ST", regex=True), 2, 1)
            for row in data[["time", "ts_code", "status_value"]].itertuples(index=False):
                if row.ts_code in status.columns and row.time in status.index:
                    status.at[row.time, row.ts_code] = max(
                        int(status.at[row.time, row.ts_code]), int(row.status_value))

        status.index.name = "time"
        status.to_parquet(self.st_file, compression="zstd")
        print(f"saved {self.st_file} shape={status.shape}")
        return self.st_file

    def _build_suspend_status(self) -> Path | None:
        suspend_path = self.raw_metadata_dir / "suspend_d.parquet"
        if not suspend_path.exists():
            print("skip suspend_status: 未找到 suspend_d.parquet")
            return None

        status = self._status_base_frame()
        suspend_d = pd.read_parquet(suspend_path)
        if not suspend_d.empty:
            data = suspend_d.copy()
            date_col = "suspend_date" if "suspend_date" in data.columns else "trade_date"
            if date_col not in data.columns:
                print("skip suspend_status: suspend_d 缺少 suspend_date/trade_date 列")
                return None
            data["time"] = pd.to_datetime(data[date_col].astype(str),
                                          format="%Y%m%d", errors="coerce")
            for row in data[["time", "ts_code"]].dropna().itertuples(index=False):
                if row.ts_code in status.columns and row.time in status.index:
                    status.at[row.time, row.ts_code] = 1

        status.index.name = "time"
        status.to_parquet(self.suspend_file, compression="zstd")
        print(f"saved {self.suspend_file} shape={status.shape}")
        return self.suspend_file

    # ═════════════════════════════════════════════════════════════════════
    # 总验证：行情覆盖 / 财务分区 / 状态宽表 / 元数据完整性
    # ═════════════════════════════════════════════════════════════════════

    def validate_all(self, end_date: str | None = None,
                     sample_stocks: int = 20) -> dict:
        """
        全量数据验收：行情 / 财务 / 状态 / 元数据 四类检查
        报告保存到 logs/validation_report.json，返回报告 dict
        （报告内 summary.fail > 0 表示验收未通过）
        """
        end = end_date or dt.datetime.now().strftime("%Y%m%d")
        checks: list[dict] = []

        def add(category: str, item: str, status: str, detail: str = ""):
            checks.append({"category": category, "item": item,
                           "status": status, "detail": detail})
            print(f"[{status}] {category}/{item} {detail}")

        # ── 1. 元数据 ──
        stock_basic_path = self.raw_metadata_dir / "stock_basic.parquet"
        stock_basic = pd.DataFrame()
        if not stock_basic_path.exists():
            add("metadata", "stock_basic", "FAIL", f"missing {stock_basic_path}")
        else:
            stock_basic = pd.read_parquet(stock_basic_path)
            n = int(stock_basic["ts_code"].nunique())
            add("metadata", "stock_basic", "PASS" if n > 4000 else "WARN",
                f"stocks={n}")

        trade_cal_path = self.raw_metadata_dir / "trade_cal.parquet"
        if not trade_cal_path.exists():
            add("metadata", "trade_cal", "FAIL", f"missing {trade_cal_path}")
        else:
            cal = pd.read_parquet(trade_cal_path)
            max_date = str(cal["cal_date"].astype(str).max())
            add("metadata", "trade_cal", "PASS" if max_date >= end else "WARN",
                f"cal_max={max_date} end={end}")

        schedule_path = self.raw_metadata_dir / "trade_schedule.parquet"
        if not schedule_path.exists():
            add("metadata", "trade_schedule", "FAIL", f"missing {schedule_path}")
        else:
            schedule = pd.read_parquet(schedule_path)
            schedule_max = str(schedule["cal_date"].astype(str).max())
            expected_schedule_end = f"{end[:4]}1231"
            add(
                "metadata", "trade_schedule",
                "PASS" if schedule_max >= expected_schedule_end else "WARN",
                f"schedule_max={schedule_max} expected={expected_schedule_end}",
            )

        for f, name in [(self.stock_info_file, "stock_info"),
                        (self.industry_map_file, "industry_map")]:
            add("metadata", name, "PASS" if f.exists() else "FAIL",
                "exists" if f.exists() else f"missing {f}")

        open_dates: list[str] = []
        last_open = ""
        try:
            open_dates = self.read_open_trade_dates(DEFAULT_START_DATE, end)
            last_open = open_dates[-1] if open_dates else ""
        except FileNotFoundError:
            pass

        benchmark_path = self.benchmark_dir / "000852.SH.parquet"
        if not benchmark_path.exists():
            add("benchmark", "000852.SH", "FAIL", f"missing {benchmark_path}")
        else:
            benchmark = pd.read_parquet(benchmark_path, columns=["trade_date"])
            benchmark_max = str(benchmark["trade_date"].astype(str).max())
            add(
                "benchmark", "000852.SH",
                "PASS" if not last_open or benchmark_max >= last_open else "WARN",
                f"benchmark_max={benchmark_max} last_open={last_open}",
            )

        # ── 2. 行情覆盖 ──
        for name in ["daily", "adj_factor", "daily_basic"]:
            have = self._market_dataset_dates(name)
            if not have:
                add("market", name, "FAIL", "no data")
                continue
            missing = [d for d in open_dates if d not in have]
            if missing:
                add("market", name, "FAIL",
                    f"missing {len(missing)} dates, e.g. {missing[:5]}")
            else:
                add("market", name, "PASS", f"covers {len(have)} open days")

        stock_list: list[str] = []
        try:
            stock_list = self.get_stock_list()
        except FileNotFoundError:
            pass
        files = list(self.market_path.glob("*.parquet")) if self.market_path.exists() else []
        if not files:
            add("market", "market_data_files", "FAIL",
                f"no files in {self.market_path}")
        else:
            ratio = len(files) / max(len(stock_list), 1)
            add("market", "market_data_files",
                "PASS" if ratio > 0.8 else "WARN",
                f"files={len(files)} stocks={len(stock_list)}")

            # 抽查在市股票的最后交易日是否跟上最新开市日
            if last_open and not stock_basic.empty:
                listed = stock_basic[
                    stock_basic["list_status"].astype(str) == "L"
                ]["ts_code"].astype(str).tolist()
                listed = [c for c in listed
                          if (self.market_path / f"{c}.parquet").exists()]
                if listed:
                    step = max(len(listed) // sample_stocks, 1)
                    sample = listed[::step][:sample_stocks]
                    stale = []
                    for code in sample:
                        df = pd.read_parquet(self.market_path / f"{code}.parquet",
                                             columns=["time"])
                        # time 是本地（北京）午夜毫秒，按 UTC 解读会差 8 小时
                        last_dt = pd.to_datetime(df["time"].max(), unit="ms", utc=True) \
                            .tz_convert("Asia/Shanghai").strftime("%Y%m%d")
                        if last_dt < last_open:
                            stale.append(f"{code}:{last_dt}")
                    add("market", "sample_last_date",
                        "PASS" if not stale else "FAIL",
                        f"sample={len(sample)} last_open={last_open} stale={stale[:5]}")

        # ── 3. 状态宽表 ──
        for f, name in [(self.st_file, "st_status"),
                        (self.suspend_file, "suspend_status")]:
            if not f.exists():
                add("status", name, "FAIL", f"missing {f}")
                continue
            df = pd.read_parquet(f)
            max_d = pd.to_datetime(df.index).max().strftime("%Y%m%d")
            ok = bool(last_open) and max_d >= last_open
            add("status", name, "PASS" if ok else "WARN",
                f"max_date={max_d} last_open={last_open} shape={df.shape}")

        # ── 4. 财务分区 ──
        try:
            fin_results = self.validate_financial_data(
                end_period=self.latest_quarter_period(end))
            fin_fail = sum(1 for r in fin_results if r.status == "FAIL")
            fin_warn = sum(1 for r in fin_results if r.status == "WARN")
            add("financial", "partitions",
                "PASS" if fin_fail == 0 else "FAIL",
                f"fail={fin_fail} warn={fin_warn} total={len(fin_results)}")
        except Exception as exc:
            add("financial", "partitions", "FAIL", str(exc))

        summary = {
            "fail": sum(1 for c in checks if c["status"] == "FAIL"),
            "warn": sum(1 for c in checks if c["status"] == "WARN"),
            "pass": sum(1 for c in checks if c["status"] == "PASS"),
        }
        report = {"end_date": end, "summary": summary, "checks": checks}
        report_path = self.log_dir / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"validation report: {report_path}")
        print(f"summary: FAIL={summary['fail']} WARN={summary['warn']} "
              f"PASS={summary['pass']}")
        return report
