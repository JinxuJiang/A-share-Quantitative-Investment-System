from __future__ import annotations

"""读取组合优化历史权重，使用 Backtrader 进行周度账户回测。

运行：
    python .\04组合回测层\backtrader.eval.py
"""

import argparse
import base64
import html
import io
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import backtrader as bt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


LAYER_ROOT = Path(__file__).resolve().parent
BENCHMARK_DATA_NAME = "__BENCHMARK__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回测完整历史优化权重")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--portfolio-release", help="默认读取02层current历史版本")
    parser.add_argument("--report-id", help="默认根据组合版本和成本场景生成")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="报告目录已存在时，明确允许替换该报告",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (LAYER_ROOT / path).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_wide(path: Path, stock_codes: list[str]) -> pd.DataFrame:
    names = pq.ParquetFile(path).schema_arrow.names
    columns = [name for name in names if name == "time" or name in stock_codes]
    missing = sorted(set(stock_codes) - set(columns))
    if missing:
        raise ValueError(f"{path.name}缺少股票，示例: {missing[:5]}")
    frame = pq.read_table(path, columns=columns).to_pandas()
    if "time" in frame.columns:
        frame = frame.set_index("time")
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame.sort_index()


def should_skip_minimum_rebalance(
    *,
    target_weight: float,
    current_weight: float,
    current_shares: int,
    desired_target_shares: int,
    threshold: float,
    target_exposure: float,
    previous_target_exposure: float | None,
    exposure_tolerance: float = 1.0e-8,
) -> bool:
    """过滤小额存量调整，但不阻止建仓、清仓或市场仓位档位切换。"""
    if threshold <= 0.0 or previous_target_exposure is None:
        return False
    if abs(target_exposure - previous_target_exposure) > exposure_tolerance:
        return False
    if current_shares <= 0 or desired_target_shares <= 0:
        return False
    return abs(target_weight - current_weight) < threshold


def next_trade_dates(
    signal_dates: list[pd.Timestamp],
    schedule: pd.DataFrame,
    last_market_date: pd.Timestamp,
) -> tuple[dict[pd.Timestamp, pd.Timestamp], list[pd.Timestamp]]:
    dates = pd.to_datetime(
        schedule["cal_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    open_days = pd.DatetimeIndex(
        dates.loc[pd.to_numeric(schedule["is_open"], errors="coerce").eq(1)].dropna()
    ).sort_values()
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    pending: list[pd.Timestamp] = []
    for signal_date in signal_dates:
        later = open_days[open_days > signal_date]
        if len(later) == 0 or later[0] > last_market_date:
            pending.append(signal_date)
        else:
            mapping[signal_date] = pd.Timestamp(later[0]).normalize()
    return mapping, pending


class ConservativeAStockCommission(bt.CommInfoBase):
    params = (
        ("buy_rate", 0.003),
        ("sell_rate", 0.004),
        ("minimum_fee", 5.0),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        if size == 0:
            return 0.0
        rate = self.p.buy_rate if size > 0 else self.p.sell_rate
        return max(float(self.p.minimum_fee), abs(float(size)) * float(price) * float(rate))


class PortfolioWeightStrategy(bt.Strategy):
    params = (
        ("targets_by_trade_date", None),
        ("suspend_status", None),
        ("board_lot", 100),
        ("minimum_rebalance_weight", 0.0),
        ("rebalance_records", None),
        ("trade_records", None),
    )

    def __init__(self):
        self.data_by_name = {
            data._name: data for data in self.datas if data._name != BENCHMARK_DATA_NAME
        }
        self.order_record: dict[int, dict] = {}
        self.nav_records: list[dict] = []
        self.total_transaction_cost = 0.0
        self.total_traded_amount = 0.0
        self.previous_target_exposure: float | None = None

    @staticmethod
    def _bar_is_current(data, current_date) -> bool:
        return len(data) > 0 and data.datetime.date(0) == current_date

    def _is_suspended(self, date: pd.Timestamp, stock_code: str) -> bool:
        status = self.p.suspend_status
        if status is None or date not in status.index or stock_code not in status.columns:
            return True
        value = pd.to_numeric(pd.Series([status.at[date, stock_code]]), errors="coerce").iloc[0]
        return not np.isfinite(value) or float(value) != 0.0

    def _equity_at_open(self, current_date) -> float:
        value = float(self.broker.getcash())
        for stock_code, data in self.data_by_name.items():
            position = self.getposition(data)
            if position.size == 0 or len(data) == 0:
                continue
            price = float(data.open[0]) if self._bar_is_current(data, current_date) else float(data.close[0])
            if np.isfinite(price) and price > 0:
                value += float(position.size) * price
        return value

    def prenext_open(self):
        """部分股票尚未上市时，仍处理已经存在行情的股票和调仓信号。"""
        self._process_open()

    def nextstart_open(self):
        self._process_open()

    def next_open(self):
        self._process_open()

    def _process_open(self):
        current_date = self.datas[0].datetime.date(0)
        trade_date = pd.Timestamp(current_date).normalize()
        event = self.p.targets_by_trade_date.get(trade_date)
        if event is None:
            return

        signal_date: pd.Timestamp = event["signal_date"]
        target_frame: pd.DataFrame = event["targets"]
        target_weights = dict(
            zip(target_frame["stock_code"].astype(str), target_frame["target_weight"].astype(float))
        )
        target_exposure = float(sum(target_weights.values()))
        target_meta = target_frame.set_index("stock_code").to_dict("index")
        held_codes = {
            name for name, data in self.data_by_name.items() if self.getposition(data).size != 0
        }
        all_codes = sorted(set(target_weights) | held_codes)
        account_value = self._equity_at_open(current_date)
        intents: list[tuple[str, bt.LineSeries, int, dict]] = []

        for stock_code in all_codes:
            data = self.data_by_name.get(stock_code)
            target_weight = float(target_weights.get(stock_code, 0.0))
            meta = target_meta.get(stock_code, {})
            current_shares = int(self.getposition(data).size) if data is not None else 0
            record = {
                "signal_date": signal_date.date().isoformat(),
                "trade_date": trade_date.date().isoformat(),
                "stock_code": stock_code,
                "selection_rank": meta.get("selection_rank"),
                "target_weight": target_weight,
                "account_value_at_open": account_value,
                "open_price": None,
                "current_shares": current_shares,
                "target_shares": current_shares,
                "desired_target_shares": current_shares,
                "order_shares": 0,
                "actual_shares": current_shares,
                "actual_weight_at_open": None,
                "transaction_cost": 0.0,
                "status": "",
            }
            self.p.rebalance_records.append(record)

            if data is None or not self._bar_is_current(data, current_date):
                record["status"] = "NO_OPEN_PRICE"
                continue
            open_price = float(data.open[0])
            record["open_price"] = open_price if np.isfinite(open_price) else None
            if not np.isfinite(open_price) or open_price <= 0:
                record["status"] = "NO_OPEN_PRICE"
                continue
            if self._is_suspended(trade_date, stock_code):
                record["status"] = "SUSPENDED"
                record["actual_weight_at_open"] = current_shares * open_price / account_value
                continue

            target_amount = account_value * target_weight
            target_shares = int(math.floor(target_amount / open_price / self.p.board_lot)) * self.p.board_lot
            record["desired_target_shares"] = target_shares
            record["target_shares"] = target_shares
            current_weight = current_shares * open_price / account_value
            record["actual_weight_at_open"] = current_weight
            if should_skip_minimum_rebalance(
                target_weight=target_weight,
                current_weight=current_weight,
                current_shares=current_shares,
                desired_target_shares=target_shares,
                threshold=float(self.p.minimum_rebalance_weight),
                target_exposure=target_exposure,
                previous_target_exposure=self.previous_target_exposure,
            ):
                record["target_shares"] = current_shares
                record["status"] = "MIN_REBALANCE_THRESHOLD"
                continue
            delta = target_shares - current_shares
            record["order_shares"] = delta
            if delta == 0:
                record["status"] = "LOT_TOO_SMALL" if target_weight > 0 and target_shares == 0 else "NO_TRADE"
                continue
            intents.append((stock_code, data, delta, record))

        self.previous_target_exposure = target_exposure

        # 先卖后买，让卖出回款尽量供同一开盘时点的买单使用。
        intents.sort(key=lambda item: 0 if item[2] < 0 else 1)
        for _, data, delta, record in intents:
            order = self.sell(data=data, size=abs(delta)) if delta < 0 else self.buy(data=data, size=delta)
            record["status"] = "SUBMITTED"
            self.order_record[order.ref] = record

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        record = self.order_record.get(order.ref)
        if record is None:
            return
        if order.status == order.Completed:
            action = "BUY" if order.isbuy() else "SELL"
            shares = abs(int(order.executed.size))
            gross_amount = shares * float(order.executed.price)
            transaction_cost = float(order.executed.comm)
            signed_shares = shares if order.isbuy() else -shares
            record["actual_shares"] = int(record["current_shares"] + signed_shares)
            record["actual_weight_at_open"] = (
                record["actual_shares"] * float(order.executed.price) / record["account_value_at_open"]
            )
            record["transaction_cost"] = transaction_cost
            record["status"] = "FILLED"
            self.total_transaction_cost += transaction_cost
            self.total_traded_amount += gross_amount
            self.p.trade_records.append(
                {
                    "trade_date": bt.num2date(order.executed.dt).date().isoformat(),
                    "stock_code": order.data._name,
                    "action": action,
                    "price": float(order.executed.price),
                    "shares": shares,
                    "gross_amount": gross_amount,
                    "transaction_cost": transaction_cost,
                    "cash_after": float(self.broker.getcash()),
                }
            )
        else:
            status_names = {
                order.Canceled: "CANCELED",
                order.Margin: "INSUFFICIENT_CASH",
                order.Rejected: "REJECTED",
                order.Expired: "EXPIRED",
            }
            record["status"] = status_names.get(order.status, "FAILED")

    def prenext(self):
        """部分数据源尚未开始时也记录组合净值。"""
        self._record_nav()

    def nextstart(self):
        self._record_nav()

    def next(self):
        self._record_nav()

    def _record_nav(self):
        current_date = self.datas[0].datetime.date(0)
        value = float(self.broker.getvalue())
        cash = float(self.broker.getcash())
        self.nav_records.append(
            {
                "date": pd.Timestamp(current_date),
                "value": value,
                "cash": cash,
                "stock_exposure": 0.0 if value <= 0 else 1.0 - cash / value,
            }
        )


def calculate_metrics(nav: pd.DataFrame, initial_cash: float, risk_free_rate: float) -> dict:
    nav = nav.sort_values("date").drop_duplicates("date", keep="last").copy()
    nav["return"] = nav["value"].pct_change().fillna(0.0)
    nav["nav"] = nav["value"] / float(initial_cash)
    nav["peak"] = nav["nav"].cummax()
    nav["drawdown"] = nav["nav"] / nav["peak"] - 1.0
    n_returns = max(len(nav) - 1, 1)
    years = n_returns / 252.0
    total_return = float(nav["nav"].iloc[-1] - 1.0)
    annual_return = float(nav["nav"].iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    annual_volatility = float(nav["return"].std(ddof=1) * np.sqrt(252))
    daily_rf = float(risk_free_rate) / 252.0
    excess = nav["return"] - daily_rf
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) > 0 else 0.0
    return nav, {
        "initial_cash": float(initial_cash),
        "final_value": float(nav["value"].iloc[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": float(nav["drawdown"].min()),
        "trading_days": int(len(nav)),
    }


def benchmark_series(path: Path, dates: pd.DatetimeIndex, initial_cash: float) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
    frame = frame.set_index("date").sort_index().reindex(dates).ffill()
    base = float(frame["close"].iloc[0])
    frame["benchmark_nav"] = frame["close"] / base
    frame["benchmark_value"] = frame["benchmark_nav"] * initial_cash
    return frame[["benchmark_nav", "benchmark_value"]]


def figure_to_base64(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_report(
    report_dir: Path,
    nav: pd.DataFrame,
    benchmark: pd.DataFrame,
    metrics: dict,
    benchmark_metrics: dict,
    config: dict,
    portfolio_release: str,
    total_cost: float,
    total_traded_amount: float,
    rebalance_records: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    combined = nav.set_index("date").join(benchmark, how="left")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axes[0].plot(combined.index, combined["nav"], label="Optimized portfolio", linewidth=2.0)
    axes[0].plot(combined.index, combined["benchmark_nav"], label="CSI 1000", linewidth=1.8)
    axes[0].axhline(1.0, color="#999999", linestyle="--", linewidth=1.2, alpha=0.7)
    axes[0].set_title(f"Portfolio vs CSI 1000 - {portfolio_release}", fontsize=16)
    axes[0].set_ylabel("Normalized NAV")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(
        combined.index,
        combined["drawdown"] * 100.0,
        0,
        alpha=0.35,
        color="#c0392b",
    )
    axes[1].set_title("Portfolio drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("%")
    axes[1].grid(alpha=0.25)
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    figure.savefig(report_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    image_b64 = figure_to_base64(figure)
    plt.close(figure)

    blocked = int(rebalance_records["status"].isin(["SUSPENDED", "NO_OPEN_PRICE", "INSUFFICIENT_CASH"]).sum())
    turnover = total_traded_amount / max(float(nav["value"].mean()), 1.0)
    cost_cfg = config["costs"]
    total_return_class = "good" if metrics["total_return"] > 0 else "bad"
    annual_return_class = "good" if metrics["annual_return"] > 0 else "bad"
    sharpe_class = "good" if metrics["sharpe"] > 1 else "warning" if metrics["sharpe"] > 0.5 else "bad"
    drawdown_class = "bad" if abs(metrics["max_drawdown"]) > 0.20 else "warning" if abs(metrics["max_drawdown"]) > 0.10 else "good"
    report_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Backtest Report - {html.escape(portfolio_release)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Microsoft YaHei', sans-serif;
       margin: 40px; background: #f5f5f5; color: #333; }}
.container {{ max-width: 1200px; margin: 0 auto; background: white;
              padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
h2 {{ color: #555; margin-top: 30px; }}
.metrics {{ display: flex; flex-wrap: wrap; justify-content: space-around; margin: 20px 0; }}
.metric-box {{ text-align: center; padding: 20px; background: #f8f9fa;
               border-radius: 8px; min-width: 150px; margin: 10px; flex: 1; }}
.metric-value {{ font-size: 30px; font-weight: bold; white-space: nowrap; }}
.metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
.good {{ color: #4CAF50; }} .warning {{ color: #FF9800; }} .bad {{ color: #f44336; }}
img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
.summary {{ background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
.summary li {{ margin: 8px 0; }}
.note {{ color: #777; font-size: 13px; margin-top: 30px; line-height: 1.7; }}
</style>
</head>
<body>
<div class="container">
<h1>📈 组合优化历史回测报告</h1>
<p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>📊 关键指标</h2>
<div class="metrics">
  <div class="metric-box"><div class="metric-value {total_return_class}">{metrics['total_return']:.2%}</div><div class="metric-label">总收益率</div></div>
  <div class="metric-box"><div class="metric-value {annual_return_class}">{metrics['annual_return']:.2%}</div><div class="metric-label">年化收益率</div></div>
  <div class="metric-box"><div class="metric-value {sharpe_class}">{metrics['sharpe']:.2f}</div><div class="metric-label">夏普比率</div></div>
  <div class="metric-box"><div class="metric-value {drawdown_class}">{metrics['max_drawdown']:.2%}</div><div class="metric-label">最大回撤</div></div>
  <div class="metric-box"><div class="metric-value">{metrics['annual_volatility']:.2%}</div><div class="metric-label">年化波动率</div></div>
  <div class="metric-box"><div class="metric-value">{metrics['final_value']:,.0f}</div><div class="metric-label">期末资金</div></div>
  <div class="metric-box"><div class="metric-value warning">{total_cost:,.0f}</div><div class="metric-label">总交易成本</div></div>
  <div class="metric-box"><div class="metric-value">{len(trades)}</div><div class="metric-label">成交次数</div></div>
</div>

<h2>📉 {html.escape(config['report']['benchmark_name'])}基准</h2>
<div class="metrics">
  <div class="metric-box"><div class="metric-value">{benchmark_metrics['total_return']:.2%}</div><div class="metric-label">累计收益率</div></div>
  <div class="metric-box"><div class="metric-value">{benchmark_metrics['annual_return']:.2%}</div><div class="metric-label">年化收益率</div></div>
  <div class="metric-box"><div class="metric-value">{benchmark_metrics['annual_volatility']:.2%}</div><div class="metric-label">年化波动率</div></div>
  <div class="metric-box"><div class="metric-value">{benchmark_metrics['sharpe']:.2f}</div><div class="metric-label">夏普比率</div></div>
  <div class="metric-box"><div class="metric-value">{benchmark_metrics['max_drawdown']:.2%}</div><div class="metric-label">最大回撤</div></div>
</div>

<div class="summary">
<h3>📋 回测设置</h3>
<ul>
  <li><strong>组合版本：</strong>{html.escape(portfolio_release)}</li>
  <li><strong>回测区间：</strong>{combined.index.min().date()} ～ {combined.index.max().date()}</li>
  <li><strong>调仓方式：</strong>每周最后一个交易日收盘生成信号，下一交易日开盘成交</li>
  <li><strong>目标仓位：</strong>由上游市场模型输出 0% / 45% / 90% 三档股票仓位</li>
  <li><strong>交易单位：</strong>100 股整数倍</li>
  <li><strong>初始资金：</strong>{metrics['initial_cash']:,.2f} 元</li>
  <li><strong>交易成本：</strong>买入 {cost_cfg['buy_cost_rate']:.2%}，卖出 {cost_cfg['sell_cost_rate']:.2%}，最低 {cost_cfg['minimum_fee']:.2f} 元/笔</li>
  <li><strong>累计换手：</strong>{turnover:.2f} 倍；<strong>阻塞委托：</strong>{blocked} 条</li>
</ul>
</div>

<h2>📈 净值曲线</h2>
<img src="data:image/png;base64,{image_b64}" alt="Equity Curve">

<p class="note">行情使用等比前复权价格；100 股约束为研究回测中的近似交易单位。最后一个没有下一交易日行情的信号记为 PENDING，不假设成交。<br>
Generated by Backtrader | Data: {datetime.now().strftime('%Y-%m-%d')}</p>
</div>
</body>
</html>"""
    (report_dir / "backtest_report.html").write_text(report_html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    if config.get("schema_version") != "portfolio_backtest_config_v1":
        raise ValueError("配置schema_version不是portfolio_backtest_config_v1")
    if config["execution"]["price_mode"] != "forward_adjusted":
        raise ValueError("第一版只支持前复权行情")
    if int(config["execution"]["board_lot"]) != 100:
        raise ValueError("第一版固定使用100股交易单位")

    portfolio_output = resolve_path(config["paths"]["portfolio_output_root"])
    portfolio_release = args.portfolio_release
    if not portfolio_release:
        current = read_json(portfolio_output / "current.json")
        if current.get("schema_version") != "portfolio_history_current_v1":
            raise ValueError("02层current不是完整历史版本")
        portfolio_release = str(current["release_id"])
    portfolio_dir = portfolio_output / "releases" / portfolio_release
    portfolio_manifest = read_json(portfolio_dir / "manifest.json")
    if portfolio_manifest.get("schema_version") != "portfolio_history_v1":
        raise ValueError("02层manifest不是portfolio_history_v1")
    weights = pd.read_parquet(portfolio_dir / "weights.parquet")
    weights["signal_date"] = pd.to_datetime(weights["signal_date"]).dt.normalize()
    if weights.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("weights存在重复日期+股票")

    data_root = resolve_path(config["paths"]["decision_data_root"])
    stock_codes = sorted(weights["stock_code"].astype(str).unique())
    print(f"加载 {len(stock_codes)} 只历史候选股票的前复权行情...", flush=True)
    market_root = data_root / "market_data"
    price_fields = {
        name: read_wide(market_root / f"{name}.parquet", stock_codes)
        for name in ["open", "high", "low", "close", "volume"]
    }
    suspend_status = read_wide(data_root / "status" / "suspend_status.parquet", stock_codes)
    schedule = pd.read_parquet(data_root / "metadata" / "trade_schedule.parquet")
    first_signal = weights["signal_date"].min()
    last_market_date = price_fields["close"].index.max()
    signal_dates = sorted(weights["signal_date"].unique())
    execution_map, pending_dates = next_trade_dates(signal_dates, schedule, last_market_date)

    targets_by_trade_date: dict[pd.Timestamp, dict] = {}
    for signal_date, trade_date in execution_map.items():
        targets = weights.loc[weights["signal_date"].eq(signal_date)].copy()
        targets_by_trade_date[trade_date] = {"signal_date": signal_date, "targets": targets}

    rebalance_records: list[dict] = []
    for signal_date in pending_dates:
        for row in weights.loc[weights["signal_date"].eq(signal_date)].itertuples(index=False):
            rebalance_records.append(
                {
                    "signal_date": signal_date.date().isoformat(),
                    "trade_date": None,
                    "stock_code": str(row.stock_code),
                    "selection_rank": int(row.selection_rank),
                    "target_weight": float(row.target_weight),
                    "account_value_at_open": None,
                    "open_price": None,
                    "current_shares": None,
                    "target_shares": None,
                    "desired_target_shares": None,
                    "order_shares": None,
                    "actual_shares": None,
                    "actual_weight_at_open": None,
                    "transaction_cost": 0.0,
                    "status": "PENDING",
                }
            )
    trade_records: list[dict] = []

    benchmark_path = resolve_path(config["paths"]["benchmark_file"])
    benchmark_raw = pd.read_parquet(benchmark_path)
    benchmark_raw["date"] = pd.to_datetime(
        benchmark_raw["trade_date"].astype(str), format="%Y%m%d"
    )
    benchmark_feed = benchmark_raw.set_index("date")[["open", "high", "low", "close", "vol"]].rename(
        columns={"vol": "volume"}
    )
    benchmark_feed["openinterest"] = 0.0
    benchmark_feed = benchmark_feed.loc[
        (benchmark_feed.index >= first_signal) & (benchmark_feed.index <= last_market_date)
    ]

    cerebro = bt.Cerebro(runonce=False, cheat_on_open=True)
    cerebro.adddata(
        bt.feeds.PandasData(dataname=benchmark_feed, fromdate=first_signal.to_pydatetime()),
        name=BENCHMARK_DATA_NAME,
    )
    for number, stock_code in enumerate(stock_codes, 1):
        frame = pd.DataFrame(
            {
                "open": price_fields["open"][stock_code],
                "high": price_fields["high"][stock_code],
                "low": price_fields["low"][stock_code],
                "close": price_fields["close"][stock_code],
                "volume": price_fields["volume"][stock_code],
            }
        )
        frame["openinterest"] = 0.0
        frame = frame.loc[(frame.index >= first_signal) & (frame.index <= last_market_date)]
        frame = frame.dropna(subset=["close"])
        if frame.empty:
            raise ValueError(f"{stock_code}回测区间没有行情")
        cerebro.adddata(
            bt.feeds.PandasData(dataname=frame, fromdate=first_signal.to_pydatetime()),
            name=stock_code,
        )
        if number % 100 == 0 or number == len(stock_codes):
            print(f"加载进度: {number}/{len(stock_codes)}", flush=True)

    cerebro.addstrategy(
        PortfolioWeightStrategy,
        targets_by_trade_date=targets_by_trade_date,
        suspend_status=suspend_status,
        board_lot=int(config["execution"]["board_lot"]),
        minimum_rebalance_weight=float(
            config["execution"].get("minimum_rebalance_weight", 0.0)
        ),
        rebalance_records=rebalance_records,
        trade_records=trade_records,
    )
    cerebro.broker.setcash(float(config["account"]["initial_cash"]))
    cerebro.broker.set_checksubmit(False)
    cerebro.broker.set_coo(True)
    cerebro.broker.addcommissioninfo(
        ConservativeAStockCommission(
            buy_rate=float(config["costs"]["buy_cost_rate"]),
            sell_rate=float(config["costs"]["sell_cost_rate"]),
            minimum_fee=float(config["costs"]["minimum_fee"]),
        )
    )
    print("启动 Backtrader...", flush=True)
    strategies = cerebro.run(runonce=False)
    strategy = strategies[0]
    nav = pd.DataFrame(strategy.nav_records)
    nav, metrics = calculate_metrics(
        nav,
        initial_cash=float(config["account"]["initial_cash"]),
        risk_free_rate=float(config["report"]["risk_free_rate"]),
    )
    benchmark = benchmark_series(
        benchmark_path, pd.DatetimeIndex(nav["date"]), float(config["account"]["initial_cash"])
    )
    benchmark_nav = pd.DataFrame(
        {"date": benchmark.index, "value": benchmark["benchmark_value"].to_numpy()}
    )
    _, benchmark_metrics = calculate_metrics(
        benchmark_nav,
        initial_cash=float(config["account"]["initial_cash"]),
        risk_free_rate=float(config["report"]["risk_free_rate"]),
    )

    report_id = args.report_id or portfolio_release.replace("portfolio_", "portfolio_") + "_high_cost"
    reports_root = resolve_path(config["paths"]["reports_root"])
    report_dir = reports_root / report_id
    if report_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"报告目录已存在，不允许覆盖: {report_dir}；如需重跑请添加 --overwrite"
            )
        shutil.rmtree(report_dir)
    temp_dir = reports_root / f".{report_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    rebalance_frame = pd.DataFrame(rebalance_records).sort_values(
        ["signal_date", "selection_rank", "stock_code"], na_position="last"
    )
    trade_frame = pd.DataFrame(trade_records)
    rebalance_frame.to_csv(temp_dir / "rebalance_signals.csv", index=False, encoding="utf-8-sig")
    trade_frame.to_csv(temp_dir / "trades.csv", index=False, encoding="utf-8-sig")
    (temp_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    create_report(
        temp_dir,
        nav,
        benchmark,
        metrics,
        benchmark_metrics,
        config,
        portfolio_release,
        strategy.total_transaction_cost,
        strategy.total_traded_amount,
        rebalance_frame,
        trade_frame,
    )
    os.replace(temp_dir, report_dir)
    print(f"回测报告完成: {report_dir / 'backtest_report.html'}")
    print(
        json.dumps(
            {
                "final_value": metrics["final_value"],
                "total_return": metrics["total_return"],
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "transaction_cost": strategy.total_transaction_cost,
                "trade_count": len(trade_frame),
                "pending_signal_dates": [date.date().isoformat() for date in pending_dates],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
