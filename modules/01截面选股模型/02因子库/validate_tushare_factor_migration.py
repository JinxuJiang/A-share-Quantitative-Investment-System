# -*- coding: utf-8 -*-
"""
Tushare 因子层迁移验收脚本（只读）。

检查范围：
1. 正式因子文件是否齐全，日期轴和股票轴是否一致。
2. 因子值是否包含 inf，以及完整清洗后的截面均值/标准差是否正常。
3. Tushare 三张财务表独立版本选择后的公告日期差异统计。
4. 抽样逐日重建资产负债表 PIT 结果，并与正式基础宽表精确比较。
5. processed_data/financial_data 中是否残留旧版基础字段。

脚本不会修改任何数据，也不会生成报告文件。

用法：
    python validate_tushare_factor_migration.py
    python validate_tushare_factor_migration.py --full-values
    python validate_tushare_factor_migration.py --full-values --pit-samples 30

退出码：
    0 = 无 FAIL
    1 = 存在 FAIL
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FACTOR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FACTOR_ROOT.parent
PROCESSED_ROOT = FACTOR_ROOT / "processed_data"
FINANCIAL_BASE_ROOT = PROCESSED_ROOT / "financial_data"
TUSHARE_FINANCIAL_ROOT = (
    PROJECT_ROOT / "01数据" / "data" / "tushare_data" / "financial_full"
)
sys.path.insert(0, str(FACTOR_ROOT / "src" / "data_engine"))
from financial_data_loader import FinancialDataLoader  # noqa: E402
from pit_aligner import PITAligner  # noqa: E402

TECHNICAL_FACTORS = [
    "ret1", "ret5", "ret20", "ret60", "ret120", "ret20_60",
    "std20", "std60", "atr20", "volatility_regime",
    "amihud", "pv_corr20", "vol_trend", "amount_ratio",
    "close_position", "intraday_return_ma5", "intraday_return_ma20",
    "close_position_ma5", "close_position_ma20", "skew20", "kurt20",
]

FINANCIAL_FACTORS = [
    "pe", "pb", "ps", "ey",
    "roe", "roa", "roe_growth", "opm",
    "profit_growth", "revenue_growth", "oper_profit_growth",
    "financial_leverage", "profit_quality", "current_asset_ratio",
    "accrual", "cashflow_to_profit", "ocf_to_revenue",
    "debt_to_equity", "current_ratio", "cash_ratio",
    "asset_growth", "capex_to_assets",
    "asset_turnover", "working_capital_ratio",
]

BALANCE_FIELD_MAP = {
    "total_share": "cap_stk",
    "total_assets": "tot_assets",
    "total_hldr_eqy_exc_min_int": "tot_shrhldr_eqy",
    "total_cur_assets": "total_current_assets",
    "total_liab": "tot_liab",
    "total_cur_liab": "total_current_liability",
    "money_cap": "cash_equivalents",
}

CASHFLOW_FIELD_MAP = {
    "n_cashflow_act": "operating_cash_flow_ttm",
    "c_pay_acq_const_fiolta": "capex_ttm",
}

CURRENT_FINANCIAL_BASE_FILES = {
    "industry",
    "cap_stk", "tot_assets", "tot_shrhldr_eqy", "total_current_assets",
    "net_profit_ttm", "revenue_ttm", "oper_profit_ttm",
    "tot_liab", "total_current_liability", "cash_equivalents",
    "operating_cash_flow_ttm", "capex_ttm",
}


@dataclass
class Finding:
    level: str
    check: str
    detail: str


class Reporter:
    def __init__(self) -> None:
        self.findings: List[Finding] = []

    def add(self, level: str, check: str, detail: str) -> None:
        finding = Finding(level, check, detail)
        self.findings.append(finding)
        print(f"[{level}] {check}: {detail}")

    def pass_(self, check: str, detail: str) -> None:
        self.add("PASS", check, detail)

    def warn(self, check: str, detail: str) -> None:
        self.add("WARN", check, detail)

    def fail(self, check: str, detail: str) -> None:
        self.add("FAIL", check, detail)

    def summary(self) -> int:
        counts = {
            level: sum(item.level == level for item in self.findings)
            for level in ("PASS", "WARN", "FAIL")
        }
        print("\n" + "=" * 78)
        print(
            "验收汇总: "
            f"PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}"
        )
        print("=" * 78)
        if counts["FAIL"]:
            print("结论：因子产物可读取，但 Tushare 迁移尚不能关闭。")
            return 1
        print("结论：未发现阻断迁移关闭的问题。")
        return 0


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ensure_paths(reporter: Reporter) -> bool:
    required = [
        PROCESSED_ROOT / "market_data" / "close.parquet",
        PROCESSED_ROOT / "factors" / "technical",
        PROCESSED_ROOT / "factors" / "financial",
        FINANCIAL_BASE_ROOT,
        TUSHARE_FINANCIAL_ROOT / "income",
        TUSHARE_FINANCIAL_ROOT / "balancesheet",
        TUSHARE_FINANCIAL_ROOT / "cashflow",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        reporter.fail("required_paths", f"缺少路径: {missing}")
        return False
    reporter.pass_("required_paths", "正式产物和 Tushare 财务分区均存在")
    return True


def factor_files() -> List[Path]:
    technical_root = PROCESSED_ROOT / "factors" / "technical"
    financial_root = PROCESSED_ROOT / "factors" / "financial"
    return (
        [technical_root / f"{name}.parquet" for name in TECHNICAL_FACTORS]
        + [financial_root / f"{name}.parquet" for name in FINANCIAL_FACTORS]
    )


def check_factor_inventory_and_axes(reporter: Reporter) -> None:
    section("1. 因子文件、日期轴与股票轴")
    expected = factor_files()
    missing = [path.name for path in expected if not path.exists()]
    if missing:
        reporter.fail("factor_inventory", f"缺少因子文件: {missing}")
        return

    actual_technical = {
        path.stem
        for path in (PROCESSED_ROOT / "factors" / "technical").glob("*.parquet")
    }
    actual_financial = {
        path.stem
        for path in (PROCESSED_ROOT / "factors" / "financial").glob("*.parquet")
    }
    extras = sorted(
        (actual_technical - set(TECHNICAL_FACTORS))
        | (actual_financial - set(FINANCIAL_FACTORS))
    )
    reporter.pass_(
        "factor_inventory",
        f"技术={len(actual_technical)} 财务={len(actual_financial)} "
        f"合计={len(actual_technical) + len(actual_financial)}",
    )
    if extras:
        reporter.warn("factor_inventory_extra", f"发现未注册因子: {extras}")

    reference_columns: Sequence[str] | None = None
    reference_times: pd.DatetimeIndex | None = None
    column_mismatch: List[str] = []
    time_mismatch: List[str] = []

    for path in expected:
        parquet_file = pq.ParquetFile(path)
        columns = parquet_file.schema_arrow.names
        if not columns or columns[0] != "time":
            reporter.fail("factor_schema", f"{path.name} 首列不是 time")
            continue

        stock_columns = columns[1:]
        times = pd.DatetimeIndex(
            pq.read_table(path, columns=["time"]).column("time").to_pylist()
        )
        if reference_columns is None:
            reference_columns = stock_columns
            reference_times = times
        elif stock_columns != reference_columns:
            column_mismatch.append(path.name)
        if reference_times is not None and not times.equals(reference_times):
            time_mismatch.append(path.name)

    if column_mismatch:
        reporter.fail("factor_stock_axis", f"股票列不一致: {column_mismatch}")
    else:
        reporter.pass_(
            "factor_stock_axis",
            f"45 个因子股票列完全一致，共 {len(reference_columns or [])} 只",
        )

    if time_mismatch:
        reporter.fail("factor_time_axis", f"日期轴不一致: {time_mismatch}")
    elif reference_times is not None:
        if reference_times.has_duplicates or not reference_times.is_monotonic_increasing:
            reporter.fail("factor_time_axis", "日期重复或未升序")
        else:
            reporter.pass_(
                "factor_time_axis",
                f"共 {len(reference_times)} 日，"
                f"{reference_times.min().date()} ~ {reference_times.max().date()}",
            )

    close_file = PROCESSED_ROOT / "market_data" / "close.parquet"
    market_columns = set(pq.ParquetFile(close_file).schema_arrow.names[1:])
    factor_columns = set(reference_columns or [])
    market_only = sorted(market_columns - factor_columns)
    factor_only = sorted(factor_columns - market_columns)
    if factor_only:
        reporter.fail("factor_market_universe", f"因子独有股票: {factor_only}")
    elif market_only:
        reporter.warn(
            "factor_market_universe",
            f"行情比因子多 {len(market_only)} 只: {market_only}",
        )
    else:
        reporter.pass_("factor_market_universe", "行情与因子股票全集一致")


def check_factor_values(reporter: Reporter, full_values: bool) -> None:
    section("2. 清洗后因子值")
    files = factor_files()
    if not full_values:
        sample_names = {"ret20", "pe", "pb", "revenue_growth"}
        files = [path for path in files if path.stem in sample_names]
        reporter.warn(
            "factor_value_scope",
            "当前为抽样值检查；使用 --full-values 检查全部 45 个因子",
        )

    issues: List[str] = []
    for index, path in enumerate(files, 1):
        frame = pd.read_parquet(path).set_index("time")
        values = frame.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        infinite_count = int(np.isinf(values).sum())
        valid_counts = finite.sum(axis=1)
        safe_values = np.where(finite, values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(safe_values, axis=1)
            stds = np.nanstd(safe_values, axis=1, ddof=0)

        valid_days = valid_counts >= 100
        if valid_days.any():
            max_abs_mean = float(np.nanmax(np.abs(means[valid_days])))
            std_p05 = float(np.nanpercentile(stds[valid_days], 5))
            std_p95 = float(np.nanpercentile(stds[valid_days], 95))
        else:
            max_abs_mean = float("nan")
            std_p05 = float("nan")
            std_p95 = float("nan")

        print(
            f"  [{index:02d}/{len(files)}] {path.stem:<24s} "
            f"inf={infinite_count:<4d} last_valid={int(valid_counts[-1]):<5d} "
            f"max|mean|={max_abs_mean:.2e} "
            f"std05/95={std_p05:.6f}/{std_p95:.6f}"
        )
        if infinite_count:
            issues.append(f"{path.name}: inf={infinite_count}")
        if valid_days.any() and (
            max_abs_mean > 1e-8 or std_p05 < 0.99 or std_p95 > 1.01
        ):
            issues.append(
                f"{path.name}: max|mean|={max_abs_mean:.3g}, "
                f"std05/95={std_p05:.6f}/{std_p95:.6f}"
            )

        del frame, values, finite, valid_counts, safe_values, means, stds
        gc.collect()

    if issues:
        reporter.fail("factor_value_quality", "; ".join(issues))
    else:
        reporter.pass_(
            "factor_value_quality",
            f"{len(files)} 个因子无 inf，截面均值和标准差符合标准化预期",
        )


def read_version_selected_table(
    table_name: str,
    value_columns: Iterable[str] = (),
) -> pd.DataFrame:
    columns = [
        "ts_code", "end_date", "ann_date", "f_ann_date",
        "report_type", "update_flag",
        *value_columns,
    ]
    frames = [
        pd.read_parquet(path, columns=columns)
        for path in sorted((TUSHARE_FINANCIAL_ROOT / table_name).glob("*.parquet"))
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame = FinancialDataLoader._select_statement_versions(frame)
    frame["_ann"] = frame["f_ann_date"]
    return frame


def merge_announcement_dates() -> pd.DataFrame:
    frames: Dict[str, pd.DataFrame] = {}
    for table_name in ("income", "balancesheet", "cashflow"):
        frame = read_version_selected_table(table_name)
        frame = (
            frame.sort_values(["ts_code", "end_date", "_ann"], kind="mergesort")
            .drop_duplicates(["ts_code", "end_date"], keep="first")
        )
        frames[table_name] = frame[
            ["ts_code", "end_date", "_ann"]
        ].rename(columns={"_ann": f"ann_{table_name}"})

    return (
        frames["income"]
        .merge(frames["balancesheet"], on=["ts_code", "end_date"], how="outer")
        .merge(frames["cashflow"], on=["ts_code", "end_date"], how="outer")
    )


def check_cross_table_announcements(
    reporter: Reporter,
) -> Dict[str, pd.DataFrame]:
    section("3. 财务三表公告日期")
    merged = merge_announcement_dates()
    late_frames: Dict[str, pd.DataFrame] = {}

    income_date = pd.to_datetime(
        merged["ann_income"], format="%Y%m%d", errors="coerce"
    )
    for table_name in ("balancesheet", "cashflow"):
        source_date = pd.to_datetime(
            merged[f"ann_{table_name}"], format="%Y%m%d", errors="coerce"
        )
        both = income_date.notna() & source_date.notna()
        later = both & (source_date > income_date)
        earlier = both & (source_date < income_date)
        late_frame = merged.loc[later].copy()
        late_frame["income_date"] = income_date[later]
        late_frame["source_date"] = source_date[later]
        late_frame["lag_days"] = (
            late_frame["source_date"] - late_frame["income_date"]
        ).dt.days
        late_frames[table_name] = late_frame

        same_count = int((both & (source_date == income_date)).sum())
        later_count = int(later.sum())
        earlier_count = int(earlier.sum())
        max_lag = int(late_frame["lag_days"].max()) if later_count else 0
        detail = (
            f"共同记录={int(both.sum())}, 同日={same_count}, "
            f"晚于利润表={later_count}, 早于利润表={earlier_count}, "
            f"最大滞后={max_lag}天"
        )
        reporter.pass_(
            f"{table_name}_independent_announcements",
            detail,
        )
        if later_count:
            print(
                late_frame[
                    [
                        "ts_code", "end_date", "ann_income",
                        f"ann_{table_name}", "lag_days",
                    ]
                ].head(10).to_string(index=False)
            )

    return late_frames


def values_close(left: float, right: float) -> bool:
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    return bool(np.isclose(left, right, rtol=1e-9, atol=1e-6))


def verify_balance_sheet_pit_reconstruction(
    reporter: Reporter,
    late_dates: pd.DataFrame,
    sample_limit: int,
) -> None:
    section("4. 资产负债表 PIT 逐日重建抽样")
    if late_dates.empty:
        reporter.pass_("balance_sheet_pit_reconstruction", "没有待验证记录")
        return

    balance = read_version_selected_table(
        "balancesheet", BALANCE_FIELD_MAP.keys()
    )
    sample_codes = (
        late_dates.sort_values(
            ["lag_days", "ts_code"], ascending=[False, True]
        )["ts_code"]
        .drop_duplicates()
        .head(sample_limit)
        .tolist()
    )
    if not sample_codes:
        reporter.pass_("balance_sheet_pit_reconstruction", "抽样数为0")
        return

    calendar = pd.DatetimeIndex(
        pq.read_table(
            PROCESSED_ROOT / "market_data" / "close.parquet",
            columns=["time"],
        ).column("time").to_pylist()
    )
    aligner = PITAligner(calendar.to_list())
    loader = FinancialDataLoader.__new__(FinancialDataLoader)
    loader.aligner = aligner
    output_fields = list(BALANCE_FIELD_MAP.values())

    actual_by_field = {}
    available_codes = set(sample_codes)
    for output_field in output_fields:
        output_path = FINANCIAL_BASE_ROOT / f"{output_field}.parquet"
        schema_codes = set(pq.ParquetFile(output_path).schema_arrow.names[1:])
        available_codes &= schema_codes
    checked_codes = sorted(available_codes)
    for output_field in output_fields:
        actual_by_field[output_field] = pd.read_parquet(
            FINANCIAL_BASE_ROOT / f"{output_field}.parquet",
            columns=["time", *checked_codes],
        ).set_index("time")

    mismatches = []
    for code in checked_codes:
        stock = balance.loc[balance["ts_code"].eq(code)].copy()
        stock = stock.rename(
            columns={
                "end_date": "report_date",
                "f_ann_date": "m_anntime",
                **BALANCE_FIELD_MAP,
            }
        )
        records = stock[
            ["report_date", "m_anntime", *output_fields]
        ].to_dict("records")
        events = loader._build_latest_period_events(records, output_fields)
        aligned = aligner.align(
            events, "m_anntime", output_fields, code
        )
        for field_index, output_field in enumerate(output_fields):
            expected = np.fromiter(
                (row[field_index + 1] for row in aligned),
                dtype=np.float64,
                count=len(aligned),
            )
            actual = (
                actual_by_field[output_field][code]
                .reindex(calendar)
                .to_numpy(dtype=np.float64)
            )
            unequal = ~np.isclose(
                expected, actual, rtol=1e-9, atol=1e-6, equal_nan=True
            )
            if unequal.any():
                first = int(np.flatnonzero(unequal)[0])
                mismatches.append(
                    {
                        "ts_code": code,
                        "field": output_field,
                        "date": calendar[first].date(),
                        "expected": expected[first],
                        "actual": actual[first],
                        "mismatch_days": int(unequal.sum()),
                    }
                )

    if mismatches:
        reporter.fail(
            "balance_sheet_pit_reconstruction",
            f"抽查 {len(checked_codes)} 只股票，发现 "
            f"{len(mismatches)} 个股票字段与逐日重建不一致",
        )
        print(pd.DataFrame(mismatches).head(20).to_string(index=False))
    else:
        reporter.pass_(
            "balance_sheet_pit_reconstruction",
            f"抽查 {len(checked_codes)} 只公告日期冲突股票，"
            f"{len(output_fields)} 个字段均与新事件算法逐日一致",
        )


def check_stale_financial_base_files(reporter: Reporter) -> None:
    section("5. 正式基础财务目录残留")
    actual = {
        path.stem for path in FINANCIAL_BASE_ROOT.glob("*.parquet")
    }
    stale = sorted(actual - CURRENT_FINANCIAL_BASE_FILES)
    missing = sorted(CURRENT_FINANCIAL_BASE_FILES - actual)
    if missing:
        reporter.fail("financial_base_inventory", f"缺少当前字段: {missing}")
    else:
        reporter.pass_(
            "financial_base_inventory",
            f"当前所需 {len(CURRENT_FINANCIAL_BASE_FILES)} 个基础文件齐全",
        )
    if stale:
        reporter.warn(
            "financial_base_stale_files",
            f"发现 {len(stale)} 个旧字段文件: {stale}",
        )
    else:
        reporter.pass_("financial_base_stale_files", "没有旧字段残留")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读验收 Tushare 因子层迁移"
    )
    parser.add_argument(
        "--full-values",
        action="store_true",
        help="读取并检查全部 45 个正式因子的 inf、均值和标准差",
    )
    parser.add_argument(
        "--pit-samples",
        type=int,
        default=20,
        help="资产负债表公告日前值验证的最大样本数，默认20",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter()

    print("=" * 78)
    print("Tushare 因子层迁移验收（只读）")
    print(f"项目路径: {PROJECT_ROOT}")
    print(f"正式因子: {PROCESSED_ROOT / 'factors'}")
    print("=" * 78)

    if not ensure_paths(reporter):
        return reporter.summary()

    check_factor_inventory_and_axes(reporter)
    check_factor_values(reporter, full_values=args.full_values)
    late_frames = check_cross_table_announcements(reporter)
    verify_balance_sheet_pit_reconstruction(
        reporter,
        late_frames["balancesheet"],
        max(args.pit_samples, 0),
    )
    check_stale_financial_base_files(reporter)
    return reporter.summary()


if __name__ == "__main__":
    sys.exit(main())
