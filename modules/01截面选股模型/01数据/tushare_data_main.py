# -*- coding: utf-8 -*-
"""
Tushare 数据下载主入口
======================

与旧 QMT 的 data_main.py 对应，一键获取全部数据：
元数据 → 行情 → 财务 → 状态（ST/停牌）

使用示例:
  # 全量下载（首次使用）
  python tushare_data_main.py --full

  # 全量下载但只到指定日期（避免盘中获取未收盘数据）
  python tushare_data_main.py --full --end-date 20260727

  # 每周更新（推荐每周收盘后运行）
  python tushare_data_main.py --weekly

  # 一次性重抓三张财务报表的 type 1 + type 5 历史版本
  python tushare_data_main.py --refresh-financial-versions

模式说明:
  --full:    全量下载（行情/状态从2010年，财务从2010Q1，已抓分区自动跳过）
  --weekly:  增量更新（行情补缺+全量重建，财务重抓最近12个季度，状态补缺，元数据刷新）
  --refresh-financial-versions: 重抓三张财务报表的全部历史分区，与已有版本追加合并（不删除旧行）
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Base_TushareEngine import TushareDataEngine


def full_download(engine: TushareDataEngine, end_date: str = ""):
    """全量下载（断点续跑，已抓分区自动跳过）"""
    print("🚀 执行 Tushare 全量下载模式...")
    end = end_date or datetime.now().strftime("%Y%m%d")
    # 只覆盖已到来的季度报告期，避免抓取未来季度返回空数据
    end_period = TushareDataEngine.latest_quarter_period(end)

    # 元数据（含交易日历，是其他数据的前提）
    print("\n📁 下载元数据...")
    engine.download_metadata(end_date=end)

    # 回测主基准：中证1000
    print("\n📉 下载基准指数数据...")
    engine.download_benchmark_index(start_date="20100101", end_date=end)

    # 行情（抓取 + 构建 per-stock 文件，已抓交易日自动跳过）
    print("\n📈 下载行情数据...")
    engine.download_market_data(start_date="20100101", end_date=end, missing_only=True)

    # 财务（四表全字段，按季度分区）
    print("\n📊 下载财务数据...")
    engine.download_financial_data(start_period="20100331", end_period=end_period)

    # 状态（ST/停牌事件表 + 宽表）
    print("\n🚨 下载状态数据...")
    engine.download_status_data(start_date="20100101", end_date=end)

    # 总验证（行情覆盖 / 财务分区 / 状态宽表 / 元数据）
    print("\n✔️ 总验证...")
    report = engine.validate_all(end_date=end)
    n_fail = report["summary"]["fail"]
    if n_fail:
        print(f"\n⚠️ 全量下载完成，但验证有 {n_fail} 项 FAIL，请检查报告！")
        raise SystemExit(1)
    print("\n✅ 全量下载完成！")


def refresh_financial_versions(
    engine: TushareDataEngine,
    end_date: str = "",
):
    """重抓三张财务报表全部历史分区（type 1 + type 5），与已有版本追加合并。

    注意：这是追加合并而非覆盖替换，旧版本行不会被删除，
    不能用于清洗历史分区中的错误行（需要时请手动删除对应 parquet）。
    """
    end = end_date or datetime.now().strftime("%Y%m%d")
    end_period = TushareDataEngine.latest_quarter_period(end)
    print(
        "📊 重抓财务历史版本（追加合并） "
        f"20100331 ~ {end_period}（income/balancesheet/cashflow）..."
    )
    engine.download_financial_data(
        start_period="20100331",
        end_period=end_period,
        tables=["income", "balancesheet", "cashflow"],
        overwrite=True,
    )
    print("\n✅ 三张财务报表的 type 1 + type 5 历史版本重抓完成！")


def main():
    parser = argparse.ArgumentParser(
        description="Tushare 数据下载工具 - 支持全量下载和每周增量更新",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python tushare_data_main.py --full                    # 全量下载
  python tushare_data_main.py --full --end-date 20260727  # 避开盘中未收盘数据
  python tushare_data_main.py --weekly                  # 每周增量更新
  python tushare_data_main.py --refresh-financial-versions
        """,
    )
    parser.add_argument("--full", action="store_true",
                        help="全量下载模式（断点续跑，已有分区自动跳过）")
    parser.add_argument("--weekly", action="store_true",
                        help="每周更新模式（增量抓取 + 必要重建）")
    parser.add_argument(
        "--refresh-financial-versions",
        action="store_true",
        help="重抓三张财务报表全部历史分区（type 1+5），与已有版本追加合并",
    )
    parser.add_argument("--end-date", default="",
                        help="结束日期 YYYYMMDD，默认今天。盘中运行建议指定昨天日期")
    args = parser.parse_args()

    if args.refresh_financial_versions:
        engine = TushareDataEngine()
        refresh_financial_versions(engine, end_date=args.end_date)
    elif args.weekly:
        from tushare_weekly_update import WeeklyTushareUpdater
        updater = WeeklyTushareUpdater()
        updater.weekly_update(end_date=args.end_date or None)
    elif args.full:
        engine = TushareDataEngine()
        full_download(engine, end_date=args.end_date)
    else:
        parser.print_help()
        print("\n💡 请选择运行模式:")
        print("   python tushare_data_main.py --full    # 全量下载")
        print("   python tushare_data_main.py --weekly # 每周更新")
        print("   python tushare_data_main.py --refresh-financial-versions")


if __name__ == "__main__":
    main()
