# 05 调仓输出层

本层读取指定频率的组合优化和配套风险发布版本，将可选 VaR 缩放应用到目标权重后，生成供人工复核的理论调仓报告。报告不会连接券商账户，也不会直接提交订单。

## 输出内容

每个版本化报告目录包含：

- `rebalance_report.html`：适合人工查看的调仓报告；
- `rebalance_actions.csv`：卖出、继续持有和新买入的结构化明细。

报告同时展示信号日、计划执行日、目标股票仓位、现金仓位、候选池规模、目标持仓数量和最小调仓阈值。

## 排名口径

排名来自组合优化版本的 `weights.parquet`：

- 候选池内股票显示精确 `selection_rank`；
- 当股票不在对应日期的候选池记录中时，显示“候选池规模 + 名外”；
- 候选池规模从 `portfolio_summary.parquet` 的 `candidate_stock_count` 动态读取，不硬编码 Top20 或 Top30。

例如当前候选池为 Top30 时，第 21～30 名显示精确排名，排名缺失的股票显示“30名外”。

## 最小调仓阈值

同一市场仓位档位下，继续持有股票的目标权重变化绝对值低于阈值时保持上期执行权重。市场仓位档位发生切换时不启用该阈值。

默认阈值为 0.5%：

```powershell
python .\05调仓输出层\generate_rebalance_report.py --frequency weekly --minimum-rebalance-weight 0.005

python .\05调仓输出层\generate_rebalance_report.py --frequency monthly --minimum-rebalance-weight 0.005
```

覆盖生成指定版本报告：

```powershell
python .\05调仓输出层\generate_rebalance_report.py `
  --portfolio-release <组合发布版本> `
  --risk-release <风险发布版本> `
  --frequency weekly `
  --previous-date YYYY-MM-DD `
  --current-date YYYY-MM-DD `
  --report-id <报告版本> `
  --minimum-rebalance-weight 0.005 `
  --overwrite
```

## 注意事项

报告比较的是两期理论目标权重，不代表账户真实成交。正式下单前仍需结合真实持仓、可用资金、最新成交价格、停牌与涨跌停状态，将目标权重换算为实际股数并再次校验。
