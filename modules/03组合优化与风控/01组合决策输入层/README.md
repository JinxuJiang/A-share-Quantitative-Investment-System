# 01 组合决策输入层

本层同步截面 Alpha、市场日度预测和全量行情，然后为全部共同交易日生成可供优化与回测使用的日度历史决策输入。

每一期都按照官方 A 股交易日历确认交易日，只使用该日及以前的数据：过滤主板、ST、停牌及历史不足股票，重算 `eligible_alpha_rank`，选出 Top 30，再用过去 252 日收益估计 Ledoit–Wolf 收缩协方差。周度和月度日期选择留给 02 层处理。

## 运行

```powershell
# 正式运行：先同步上游数据，再重建全部历史
python .\01组合决策输入层\run_pipeline.py

# 调试：复用已经同步的数据
python .\01组合决策输入层\run_pipeline.py --skip-sync

# 可选日期范围
python .\01组合决策输入层\run_pipeline.py --skip-sync --start-date 2024-01-01 --end-date 2025-12-31
```

## 正式输出

```text
outputs/releases/decision_daily_history_起始日_结束日_v1/
├─ decision_inputs.parquet
├─ covariance.parquet
├─ period_summary.parquet
├─ config.yaml
└─ manifest.json
```

- `decision_inputs.parquet`：完整的日期×股票决策表；每期 20 行，包含 Alpha、行业、排名、市场状态、股票总仓位和个股风险。
- `covariance.parquet`：完整的日期×股票 i×股票 j 协方差长表；每期 400 行。
- `period_summary.parquet`：每期一行的股票池数量、市场仓位、收缩系数和校验结果。
- `current.json`：指向当前完整历史版本。

Alpha 与市场模型只按 `signal_date` 合并。该日期表示交易日收盘后形成决策；下一交易日开盘成交由后续回测层处理。
