# 04 回测层

> 对模型预测进行 Alphalens 因子检验和 Backtrader 策略回测。

## 当前迁移状态

截至 2026-07-31，第四层使用 V2 单一 PIT 预测链：

| 组件 | 状态 | 说明 |
|:---|:---|:---|
| `alphalens_analysis.py` | 可用 | 直接读取 V2 的 `predictions.parquet` 或 `smoothed_predictions.parquet` |
| `backtrader.eval.py` | 可用 | 读取 V2 主预测链和 Tushare ST 状态 |

V2 的 `predictions.parquet` 已覆盖最新无标签日期，不再生成或拼接独立的 `live_predictions.parquet`。

## 数据流

```text
03模型训练层/experiments/{exp_id}/
├── predictions.parquet
└── smoothed_predictions.parquet
             │
             ├───────────────┐
             ▼               ▼
       Alphalens         Backtrader
       IC / IR           月末调仓
       分层收益           成本 / 止损
       稳定性             成交过滤
             │               │
             └───────┬───────┘
                     ▼
          04回测层/reports/{exp_id}/
```

## 目录结构

```text
04回测层/
├── alphalens_analysis.py
├── backtrader.eval.py
├── backtrader.weekly.eval.py
├── utils.py
├── 项目要求.md
├── reports/                    # 运行时生成，不进入Git
└── README.md
```

## Alphalens 分析

### 输入

```text
03模型训练层/experiments/{exp_id}/predictions.parquet
03模型训练层/experiments/{exp_id}/smoothed_predictions.parquet
02因子库/processed_data/market_data/close.parquet
```

### 运行

原始预测：

```powershell
python 04回测层/alphalens_analysis.py `
  --exp-id lgbm20_tushare_profit20_v2 `
  --periods 20 `
  --quantiles 10
```

平滑预测：

```powershell
python 04回测层/alphalens_analysis.py `
  --exp-id lgbm20_tushare_profit20_v2 `
  --periods 20 `
  --quantiles 10 `
  --use-smooth
```

### 参数

| 参数 | 含义 |
|:---|:---|
| `--exp-id` | `03模型训练层/experiments/` 下的实验目录名 |
| `--periods` | 远期收益周期，可传一个或多个整数 |
| `--quantiles` | 截面分组数量，默认 10 |
| `--use-smooth` | 使用 `pred_score_smooth` |

### 输出

```text
reports/{exp_id}/
├── alphalens_report.html
├── ic_analysis.png
├── returns_analysis.png
├── turnover_analysis.png
├── alphalens_report_smooth.html
├── ic_analysis_smooth.png
├── returns_analysis_smooth.png
└── turnover_analysis_smooth.png
```

带 `_smooth` 后缀的文件只在使用 `--use-smooth` 时生成。

## Backtrader 回测

### 当前策略参数

参数定义在 `backtrader.eval.py` 的 `STRATEGY_PARAMS`：

| 参数 | 当前值 |
|:---|---:|
| 每次持仓数量 | 20 |
| 回测开始日期 | 2023-10-01 |
| 回测结束日期 | 自动取预测与行情共同覆盖的最新日期 |
| 初始资金 | 50,000 |
| 手续费 | 0.2% |
| 目标总仓位 | 90% |
| 有效止损阈值 | 20% |

当前代码没有显式设置滑点，不应在报告中描述为“已考虑滑点”。

止损参数的实际值为 `0.2`，即 20%。

### 选股和调仓

1. 每月最后一个交易日收盘后生成调仓截面，订单在下一交易日开盘成交。
2. 只保留 `60xxxx.SH` 和 `00xxxx.SZ` 主板股票。
3. 过滤 ST 股票。
4. 只使用截至信号日的历史，过滤有效数据不足 20 天的股票。
5. 按预测分数选择 Top 20；信号生成不读取 T+1 开盘价。
6. 以 90% 总仓位等权配置。
7. 每日检查持仓止损。

月末日期来自 `trade_schedule.parquet` 的正式交易安排，不会把月中最新数据日
误认为月末。当前回测没有模拟涨停无法成交，因此报告不应解释为已处理涨停成交约束。

### 基准

主基准为中证1000（`000852.SH`），数据路径：

```text
01数据/data/tushare_data/benchmark/000852.SH.parquet
```

HTML 和 PNG 都以 1 为起点同时展示策略与中证1000净值；HTML 另列基准累计收益、
年化收益、年化波动、Sharpe 和最大回撤。

### 运行

原始预测：

```powershell
python 04回测层/backtrader.eval.py `
  --exp-id lgbm20_tushare_profit20_v2
```

周频对照（仍使用同一份融合预测与平滑逻辑）：

```powershell
python 04回测层/backtrader.weekly.eval.py `
  --exp-id ensemble_5d_20d_60d_profit20_v2 `
  --use-smooth
```

周频脚本仅把调仓日切换为每周最后一个实际交易日，下一交易日开盘成交；
模型不会每周重新训练或寻参。结果独立写入：

```text
reports/weekly/{exp_id}/
```

月频和周频回测均不再使用硬编码结束日期。每次运行会自动读取预测表最新日期；
如果行情覆盖略短，则安全地使用预测与行情的共同最新日期。

平滑预测：

```powershell
python 04回测层/backtrader.eval.py `
  --exp-id lgbm20_tushare_profit20_v2 `
  --use-smooth
```

V2 不再生成 `live_predictions.parquet`；Backtrader 直接读取单一 PIT 主预测文件。

### 输出

```text
reports/{exp_id}/
├── rebalance_signals.csv
├── trades.csv
├── equity_curve.png
└── backtest_report.html
```

当前默认脚本不会生成 `performance.json`。

### ST 状态路径

```text
01数据/data/tushare_data/st_status.parquet
```

旧 `generate_live_signals.py` 和 `backtrader_eval_有金额上限.py` 已退出主链路并删除。

## 第四层待办

- [x] 将有效止损阈值的注释和日志统一为 20%。
- [ ] 增加 V2 预测文件兼容性测试。
- [x] 完成 Tushare 5d/20d/60d 和三周期融合正式回测验收。

## 绩效展示资产

`assets/performance/` 同时保留当前 Tushare 正式结果和迁移前历史基线：

```text
assets/performance/
├── lgbm60_tushare_profit20_v2/
├── ensemble_5d_20d_60d_profit20_v2/
└── ensemble_5d_20d_60d_v1/          # 迁移前历史对照
```

不同数据源和训练版本的指标不可直接视为同一实验的连续更新。

---

*最后更新：2026-08-01*
*维护者：蒋大王*
