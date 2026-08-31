from __future__ import annotations

r"""读取组合优化层最近两期权重，生成理论调仓报告。

运行：
    python .\05调仓输出层\generate_rebalance_report.py

重复生成同一期报告：
    python .\05调仓输出层\generate_rebalance_report.py --overwrite
"""

import argparse
import html
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


LAYER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LAYER_ROOT.parent
PORTFOLIO_OUTPUT_ROOT = PROJECT_ROOT / "02组合优化层" / "outputs"
RISK_OUTPUT_ROOT = PROJECT_ROOT / "03组合风控层" / "outputs"
STOCK_INFO_PATH = PROJECT_ROOT / "01组合决策输入层" / "data" / "metadata" / "stock_info.parquet"
TRADE_SCHEDULE_PATH = (
    PROJECT_ROOT / "01组合决策输入层" / "data" / "metadata" / "trade_schedule.parquet"
)
REPORTS_ROOT = LAYER_ROOT / "reports"
WEIGHT_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成最近两期目标组合的调仓报告")
    parser.add_argument("--portfolio-release", help="默认读取02组合优化层对应频率的 current.json")
    parser.add_argument("--risk-release", help="默认读取与组合匹配的03组合风控层版本")
    parser.add_argument(
        "--frequency", choices=("weekly", "monthly"), default="weekly"
    )
    parser.add_argument("--previous-date", help="上期信号日期，格式 YYYY-MM-DD")
    parser.add_argument("--current-date", help="本期信号日期，格式 YYYY-MM-DD")
    parser.add_argument("--report-id", help="默认使用 rebalance_本期日期")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名报告目录")
    parser.add_argument(
        "--minimum-rebalance-weight",
        type=float,
        default=0.005,
        help="同一市场仓位档位下继续持有股票的最小调仓权重，默认0.005",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_release(
    release_id: str | None, frequency: str = "weekly"
) -> tuple[str, Path]:
    track_root = PORTFOLIO_OUTPUT_ROOT / frequency
    if release_id is None:
        current_path = track_root / "current.json"
        if not current_path.exists():
            raise FileNotFoundError(f"找不到组合优化层当前版本: {current_path}")
        current = read_json(current_path)
        release_id = current.get("release_id")
    if not release_id:
        raise ValueError("组合优化层 current.json 缺少 release_id")
    release_dir = track_root / "releases" / release_id
    if not release_dir.is_dir():
        raise FileNotFoundError(f"找不到组合优化版本: {release_dir}")
    return release_id, release_dir


def load_risk_history(
    release_id: str | None, portfolio_release: str, frequency: str
) -> tuple[str, pd.DataFrame]:
    track_root = RISK_OUTPUT_ROOT / frequency
    if release_id is None:
        current = read_json(track_root / "current.json")
        release_id = current.get("release_id")
    if not release_id:
        raise ValueError("03组合风控层 current.json 缺少 release_id")
    release_dir = track_root / "releases" / release_id
    manifest = read_json(release_dir / "manifest.json")
    if manifest.get("source_portfolio_release") != portfolio_release:
        raise ValueError("风险版本不是由所选组合版本计算得到")
    if manifest.get("rebalance_frequency") != frequency:
        raise ValueError("风险版本频率与调仓报告频率不一致")
    risk = pd.read_parquet(release_dir / "risk.parquet")
    risk["signal_date"] = pd.to_datetime(risk["signal_date"]).dt.normalize()
    return str(release_id), risk


def apply_risk_scaling(
    weights: pd.DataFrame, summary: pd.DataFrame, risk: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "signal_date",
        "risk_scale",
        "scaled_stock_exposure",
        "scaled_cash_weight",
        "var_budget",
    }
    missing = required - set(risk.columns)
    if missing:
        raise ValueError(f"风险历史缺少缩放字段，请重跑03层: {sorted(missing)}")
    risk_fields = risk[list(required)].copy()
    if risk_fields["signal_date"].duplicated().any():
        raise ValueError("风险历史存在重复 signal_date")
    scaled_weights = weights.merge(
        risk_fields[["signal_date", "risk_scale"]],
        on="signal_date",
        how="left",
        validate="many_to_one",
    )
    if scaled_weights["risk_scale"].isna().any():
        raise ValueError("风险历史没有覆盖全部组合信号日期")
    scaled_weights["unscaled_target_weight"] = scaled_weights["target_weight"]
    scaled_weights["target_weight"] = (
        scaled_weights["target_weight"] * scaled_weights["risk_scale"]
    )
    scaled_summary = summary.merge(
        risk_fields,
        on="signal_date",
        how="left",
        validate="one_to_one",
    )
    if scaled_summary["risk_scale"].isna().any():
        raise ValueError("风险历史没有覆盖全部组合汇总日期")
    scaled_summary["unscaled_stock_exposure"] = scaled_summary["stock_exposure"]
    scaled_summary["unscaled_cash_weight"] = scaled_summary["cash_weight"]
    scaled_summary["stock_exposure"] = scaled_summary["scaled_stock_exposure"]
    scaled_summary["cash_weight"] = scaled_summary["scaled_cash_weight"]
    return scaled_weights, scaled_summary


def choose_dates(
    available_dates: pd.DatetimeIndex,
    previous_date: str | None,
    current_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.DatetimeIndex(available_dates).normalize().sort_values().unique()
    if len(dates) < 2:
        raise ValueError("至少需要两期组合权重才能生成调仓报告")

    current = pd.Timestamp(current_date).normalize() if current_date else pd.Timestamp(dates[-1])
    if current not in dates:
        raise ValueError(f"本期日期不在组合权重中: {current.date()}")

    if previous_date:
        previous = pd.Timestamp(previous_date).normalize()
    else:
        earlier = dates[dates < current]
        if len(earlier) == 0:
            raise ValueError(f"本期日期之前没有可比较的组合: {current.date()}")
        previous = pd.Timestamp(earlier[-1])
    if previous not in dates:
        raise ValueError(f"上期日期不在组合权重中: {previous.date()}")
    if previous >= current:
        raise ValueError("上期日期必须早于本期日期")
    return previous, current


def load_stock_names() -> pd.DataFrame:
    if not STOCK_INFO_PATH.exists():
        return pd.DataFrame(columns=["stock_code", "stock_name"])
    info = pd.read_parquet(STOCK_INFO_PATH, columns=["order_book_id", "symbol"])
    return (
        info.rename(columns={"order_book_id": "stock_code", "symbol": "stock_name"})
        .dropna(subset=["stock_code"])
        .drop_duplicates("stock_code", keep="last")
    )


def next_planned_trade_date(signal_date: pd.Timestamp) -> pd.Timestamp | None:
    if not TRADE_SCHEDULE_PATH.exists():
        return None
    schedule = pd.read_parquet(TRADE_SCHEDULE_PATH)
    dates = pd.to_datetime(
        schedule["cal_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    open_dates = pd.DatetimeIndex(
        dates.loc[pd.to_numeric(schedule["is_open"], errors="coerce").eq(1)].dropna()
    ).sort_values()
    later = open_dates[open_dates > signal_date]
    return pd.Timestamp(later[0]).normalize() if len(later) else None


def build_actions(
    weights: pd.DataFrame,
    previous_date: pd.Timestamp,
    current_date: pd.Timestamp,
    minimum_rebalance_weight: float = 0.005,
    exposure_changed: bool = False,
) -> pd.DataFrame:
    if minimum_rebalance_weight < 0.0:
        raise ValueError("minimum_rebalance_weight不能为负")
    required = {"signal_date", "stock_code", "industry", "selection_rank", "target_weight"}
    missing = sorted(required - set(weights.columns))
    if missing:
        raise ValueError(f"weights.parquet 缺少字段: {missing}")

    previous = weights.loc[
        weights["signal_date"].eq(previous_date),
        ["stock_code", "industry", "selection_rank", "target_weight"],
    ].rename(
        columns={
            "industry": "previous_industry",
            "selection_rank": "previous_rank",
            "target_weight": "previous_weight",
        }
    )
    current = weights.loc[
        weights["signal_date"].eq(current_date),
        ["stock_code", "industry", "selection_rank", "target_weight"],
    ].rename(
        columns={
            "industry": "current_industry",
            "selection_rank": "current_rank",
            "target_weight": "current_weight",
        }
    )
    if previous["stock_code"].duplicated().any() or current["stock_code"].duplicated().any():
        raise ValueError("同一期 weights.parquet 中存在重复股票")

    actions = previous.merge(current, on="stock_code", how="outer")
    actions[["previous_weight", "current_weight"]] = actions[
        ["previous_weight", "current_weight"]
    ].fillna(0.0)
    actions = actions.loc[
        actions[["previous_weight", "current_weight"]].max(axis=1).gt(WEIGHT_EPSILON)
    ].copy()
    actions["industry"] = actions["current_industry"].fillna(actions["previous_industry"])
    actions = actions.merge(load_stock_names(), on="stock_code", how="left")
    actions["stock_name"] = actions["stock_name"].fillna("")
    actions["target_weight_change"] = actions["current_weight"] - actions["previous_weight"]

    previous_active = actions["previous_weight"].gt(WEIGHT_EPSILON)
    current_active = actions["current_weight"].gt(WEIGHT_EPSILON)
    held = previous_active & current_active
    actions["threshold_skipped"] = (
        held
        & (not exposure_changed)
        & actions["target_weight_change"].abs().lt(minimum_rebalance_weight)
    )
    actions["execution_weight"] = actions["current_weight"]
    actions.loc[actions["threshold_skipped"], "execution_weight"] = actions.loc[
        actions["threshold_skipped"], "previous_weight"
    ]
    actions["weight_change"] = actions["execution_weight"] - actions["previous_weight"]
    actions["action_group"] = "继续持有"
    actions.loc[previous_active & ~current_active, "action_group"] = "卖出"
    actions.loc[~previous_active & current_active, "action_group"] = "新买入"
    actions["action"] = actions["action_group"]
    actions.loc[held & actions["weight_change"].gt(WEIGHT_EPSILON), "action"] = "继续持有-加仓"
    actions.loc[held & actions["weight_change"].lt(-WEIGHT_EPSILON), "action"] = "继续持有-减仓"
    actions.loc[held & actions["weight_change"].abs().le(WEIGHT_EPSILON), "action"] = "继续持有-不变"
    actions.loc[actions["threshold_skipped"], "action"] = "继续持有-阈值内不调仓"

    order = {"卖出": 0, "继续持有": 1, "新买入": 2}
    actions["action_order"] = actions["action_group"].map(order)
    actions = actions.sort_values(
        ["action_order", "weight_change", "current_rank", "previous_rank", "stock_code"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return actions


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_rank(value: object, candidate_count: int) -> str:
    if candidate_count <= 0:
        raise ValueError("candidate_stock_count必须为正整数")
    return f"{candidate_count}名外" if pd.isna(value) else str(int(value))


def action_table(
    frame: pd.DataFrame,
    empty_text: str,
    previous_candidate_count: int,
    current_candidate_count: int,
) -> str:
    if frame.empty:
        return f'<p class="empty">{html.escape(empty_text)}</p>'
    rows = []
    for row in frame.itertuples(index=False):
        delta_class = "positive" if row.weight_change > 0 else "negative" if row.weight_change < 0 else ""
        rows.append(
            "<tr>"
            f"<td><span class=\"tag {html.escape(row.action_group)}\">{html.escape(row.action)}</span></td>"
            f"<td>{html.escape(str(row.stock_code))}</td>"
            f"<td>{html.escape(str(row.stock_name))}</td>"
            f"<td>{html.escape(str(row.industry))}</td>"
            f"<td>{format_rank(row.previous_rank, previous_candidate_count)}</td>"
            f"<td>{format_rank(row.current_rank, current_candidate_count)}</td>"
            f"<td>{pct(row.previous_weight)}</td>"
            f"<td>{pct(row.current_weight)}</td>"
            f"<td>{pct(row.execution_weight)}</td>"
            f"<td class=\"{delta_class}\">{row.weight_change * 100:+.2f}%</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>操作</th><th>股票代码</th><th>股票名称</th><th>行业</th>"
        "<th>上期排名</th><th>本期排名</th><th>上期权重</th><th>模型目标权重</th>"
        "<th>理论执行权重</th><th>执行权重变化</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def create_html(
    release_id: str,
    previous_date: pd.Timestamp,
    current_date: pd.Timestamp,
    previous_summary: pd.Series,
    current_summary: pd.Series,
    actions: pd.DataFrame,
    minimum_rebalance_weight: float,
    planned_execution_date: pd.Timestamp | None,
    frequency: str = "weekly",
) -> str:
    sell = actions.loc[actions["action_group"].eq("卖出")]
    keep = actions.loc[actions["action_group"].eq("继续持有")]
    buy = actions.loc[actions["action_group"].eq("新买入")]
    exposure_change = float(current_summary["stock_exposure"] - previous_summary["stock_exposure"])
    exposure_class = "positive" if exposure_change > 0 else "negative" if exposure_change < 0 else ""
    threshold_skipped = int(actions["threshold_skipped"].sum())
    previous_candidate_count = int(previous_summary["candidate_stock_count"])
    current_candidate_count = int(current_summary["candidate_stock_count"])
    execution_text = (
        planned_execution_date.date().isoformat()
        if planned_execution_date is not None
        else "交易日历暂不可用"
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>调仓报告 - {current_date.date()}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Microsoft YaHei', sans-serif;
       margin: 40px; background: #f5f5f5; color: #333; }}
.container {{ max-width: 1280px; margin: 0 auto; background: white; padding: 30px;
              border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
h1 {{ border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
h2 {{ color: #555; margin-top: 32px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 16px; margin: 22px 0; }}
.card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; }}
.value {{ font-size: 28px; font-weight: 700; }} .label {{ color: #666; font-size: 14px; margin-top: 6px; }}
.summary {{ background: #e8f5e9; border-radius: 8px; padding: 20px; margin: 20px 0; line-height: 1.8; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; }}
th,td {{ padding: 10px 9px; border-bottom: 1px solid #e5e7eb; text-align: right; white-space: nowrap; }}
th {{ background: #f8f9fa; color: #555; }} th:nth-child(-n+4),td:nth-child(-n+4) {{ text-align: left; }}
.tag {{ display: inline-block; border-radius: 999px; padding: 4px 9px; font-size: 13px; font-weight: 600; }}
.tag.卖出 {{ color: #b42318; background: #fee4e2; }}
.tag.新买入 {{ color: #027a48; background: #d1fadf; }}
.tag.继续持有 {{ color: #175cd3; background: #dbeafe; }}
.positive {{ color: #16803c; font-weight: 600; }} .negative {{ color: #d92d20; font-weight: 600; }}
.empty {{ color: #777; background: #f8f9fa; padding: 16px; border-radius: 8px; }}
.note {{ color: #777; font-size: 13px; line-height: 1.7; margin-top: 28px; }}
@media (max-width: 700px) {{ body {{ margin: 12px; }} .container {{ padding: 18px; }} }}
</style>
</head>
<body><div class="container">
<h1>📋 本期调仓报告</h1>
<p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="summary">
  <strong>组合版本：</strong>{html.escape(release_id)}<br>
  <strong>调仓频率：</strong>{'周度' if frequency == 'weekly' else '月度'}<br>
  <strong>上期信号日：</strong>{previous_date.date()}<br>
  <strong>本期信号日：</strong>{current_date.date()}（收盘后形成）<br>
  <strong>计划执行日：</strong>{execution_text}（开盘执行）<br>
  <strong>候选池规模：</strong>{previous_candidate_count} → {current_candidate_count} 只<br>
  <strong>最小调仓阈值：</strong>{minimum_rebalance_weight:.2%}（市场仓位档位切换时不启用）<br>
  <strong>说明：</strong>本报告比较最近两期目标权重，属于理论调仓清单。
</div>

<h2>组合整体变化</h2>
<div class="cards">
  <div class="card"><div class="value">{pct(float(previous_summary['stock_exposure']))}</div><div class="label">上期股票仓位</div></div>
  <div class="card"><div class="value">{pct(float(current_summary['stock_exposure']))}</div><div class="label">本期股票仓位</div></div>
  <div class="card"><div class="value {exposure_class}">{exposure_change * 100:+.2f}%</div><div class="label">股票仓位变化</div></div>
  <div class="card"><div class="value">{pct(float(previous_summary['cash_weight']))} → {pct(float(current_summary['cash_weight']))}</div><div class="label">现金仓位</div></div>
  <div class="card"><div class="value">{float(current_summary['risk_scale']):.2f}x</div><div class="label">当前VaR缩放</div></div>
  <div class="card"><div class="value">{pct(float(current_summary['var_budget']))}</div><div class="label">VaR预算</div></div>
</div>
<div class="cards">
  <div class="card"><div class="value negative">{len(sell)}</div><div class="label">卖出</div></div>
  <div class="card"><div class="value">{len(keep)}</div><div class="label">继续持有</div></div>
  <div class="card"><div class="value positive">{len(buy)}</div><div class="label">新买入</div></div>
  <div class="card"><div class="value">{int(previous_summary['active_stock_count'])} → {int(current_summary['active_stock_count'])}</div><div class="label">目标持仓数量</div></div>
  <div class="card"><div class="value">{threshold_skipped}</div><div class="label">阈值内不调仓</div></div>
</div>

<h2>🔴 卖出</h2>{action_table(sell, '本期没有需要卖出的股票。', previous_candidate_count, current_candidate_count)}
<h2>🔵 继续持有</h2>{action_table(keep, '本期没有继续持有的股票。', previous_candidate_count, current_candidate_count)}
<h2>🟢 新买入</h2>{action_table(buy, '本期没有新买入的股票。', previous_candidate_count, current_candidate_count)}

<p class="note">重要：本报告没有接入真实券商账户，因此“卖出、继续持有、新买入”均指目标权重变化，不代表真实成交。实际股数需要结合真实持仓、账户资金和成交价格另行计算。</p>
</div></body></html>"""


def main() -> None:
    args = parse_args()
    release_id, release_dir = resolve_release(args.portfolio_release, args.frequency)
    weights_path = release_dir / "weights.parquet"
    summary_path = release_dir / "portfolio_summary.parquet"
    if not weights_path.exists() or not summary_path.exists():
        raise FileNotFoundError("组合优化版本缺少 weights.parquet 或 portfolio_summary.parquet")

    weights = pd.read_parquet(weights_path)
    summary = pd.read_parquet(summary_path)
    weights["signal_date"] = pd.to_datetime(weights["signal_date"]).dt.normalize()
    summary["signal_date"] = pd.to_datetime(summary["signal_date"]).dt.normalize()
    risk_release, risk = load_risk_history(
        args.risk_release, release_id, args.frequency
    )
    weights, summary = apply_risk_scaling(weights, summary, risk)
    previous_date, current_date = choose_dates(
        pd.DatetimeIndex(weights["signal_date"].unique()),
        args.previous_date,
        args.current_date,
    )
    summary_by_date = summary.set_index("signal_date")
    if previous_date not in summary_by_date.index or current_date not in summary_by_date.index:
        raise ValueError("portfolio_summary.parquet 缺少所选日期")
    previous_summary = summary_by_date.loc[previous_date]
    current_summary = summary_by_date.loc[current_date]
    planned_execution_date = next_planned_trade_date(current_date)
    exposure_changed = not abs(
        float(current_summary["stock_exposure"]) - float(previous_summary["stock_exposure"])
    ) <= WEIGHT_EPSILON
    actions = build_actions(
        weights,
        previous_date,
        current_date,
        minimum_rebalance_weight=float(args.minimum_rebalance_weight),
        exposure_changed=exposure_changed,
    )
    actions["planned_execution_date"] = (
        planned_execution_date.date().isoformat()
        if planned_execution_date is not None
        else None
    )

    report_id = args.report_id or f"rebalance_{current_date:%Y%m%d}"
    track_reports_root = REPORTS_ROOT / args.frequency
    report_dir = track_reports_root / report_id
    if report_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"报告已存在: {report_dir}；如需重生成请添加 --overwrite")
        shutil.rmtree(report_dir)
    temp_dir = track_reports_root / f".{report_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    csv_frame = actions[
        [
            "action",
            "stock_code",
            "stock_name",
            "industry",
            "previous_rank",
            "current_rank",
            "previous_weight",
            "current_weight",
            "execution_weight",
            "target_weight_change",
            "weight_change",
            "threshold_skipped",
            "planned_execution_date",
        ]
    ].copy()
    csv_frame = csv_frame.rename(
        columns={
            "previous_weight": "previous_weight_pct",
            "current_weight": "current_weight_pct",
            "execution_weight": "execution_weight_pct",
            "target_weight_change": "target_weight_change_pct",
            "weight_change": "weight_change_pct",
        }
    )
    for column in [
        "previous_weight_pct",
        "current_weight_pct",
        "execution_weight_pct",
        "target_weight_change_pct",
        "weight_change_pct",
    ]:
        csv_frame[column] = (csv_frame[column] * 100.0).round(2)
    previous_candidate_count = int(previous_summary["candidate_stock_count"])
    current_candidate_count = int(current_summary["candidate_stock_count"])
    csv_frame["previous_rank"] = csv_frame["previous_rank"].map(
        lambda value: format_rank(value, previous_candidate_count)
    )
    csv_frame["current_rank"] = csv_frame["current_rank"].map(
        lambda value: format_rank(value, current_candidate_count)
    )
    csv_frame.to_csv(temp_dir / "rebalance_actions.csv", index=False, encoding="utf-8-sig")
    report_html = create_html(
        release_id,
        previous_date,
        current_date,
        previous_summary,
        current_summary,
        actions,
        float(args.minimum_rebalance_weight),
        planned_execution_date,
        args.frequency,
    )
    (temp_dir / "rebalance_report.html").write_text(report_html, encoding="utf-8")
    temp_dir.replace(report_dir)

    counts = actions["action_group"].value_counts().to_dict()
    print(f"调仓报告完成: {report_dir / 'rebalance_report.html'}")
    print(
        json.dumps(
            {
                "previous_date": previous_date.date().isoformat(),
                "current_date": current_date.date().isoformat(),
                "planned_execution_date": (
                    planned_execution_date.date().isoformat()
                    if planned_execution_date is not None
                    else None
                ),
                "previous_stock_exposure": float(previous_summary["stock_exposure"]),
                "current_stock_exposure": float(current_summary["stock_exposure"]),
                "sell": int(counts.get("卖出", 0)),
                "keep": int(counts.get("继续持有", 0)),
                "buy": int(counts.get("新买入", 0)),
                "minimum_rebalance_weight": float(args.minimum_rebalance_weight),
                "threshold_skipped": int(actions["threshold_skipped"].sum()),
                "rebalance_frequency": args.frequency,
                "risk_release": risk_release,
                "current_risk_scale": float(current_summary["risk_scale"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
