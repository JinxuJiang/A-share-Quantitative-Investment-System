# 04 经济意义与回测层

本层读取模型训练层的正式样本外预测，使用 Backtrader 对中证1000指数代理资产执行多头/现金择时回测。T 日收盘读取信号，T+1 交易日开盘成交。

## 版本化策略实验

每次回测必须指定唯一的 `run-id`。同名实验默认禁止覆盖；确实需要覆盖时必须显式使用 `--overwrite`。

仓位参数放在 `strategy_configs/` 中，与模型预测分开管理。目前提供：

- `bull90_neutral45_bear0.yaml`：牛市90%、震荡45%、熊市0%。
- `bull90_neutral60_bear30.yaml`：牛市90%、震荡60%、熊市30%。

运行一套策略：

```powershell
python 04经济意义与回测层/run_economic_value.py `
  --strategy-config 04经济意义与回测层/strategy_configs/bull90_neutral60_bear30.yaml `
  --run-id bull90_neutral60_bear30_v1 `
  --model all
```

验收指定实验：

```powershell
python 04经济意义与回测层/validate_backtest.py `
  --run-id bull90_neutral60_bear30_v1
```

## 输出结构

```text
data/processed/{run-id}/
├─ ridge/
├─ cnn_gru/
├─ comparison/
└─ logs/backtest_validation_report.json

reports/{run-id}/
├─ config_snapshot.yaml
├─ run_manifest.json
├─ ridge/
├─ cnn_gru/
└─ comparison/
   ├─ benchmark.json
   ├─ equity_comparison.png
   ├─ rebalance_signals.csv
   └─ backtest_report.html

reports/strategy_comparison/
├─ performance_summary.csv
├─ equity_comparison.png
└─ strategy_comparison.html
```

`config_snapshot.yaml` 和 `run_manifest.json` 保存该实验实际使用的策略参数、模型、数据日期和 Git 信息。跨策略总览会在每次成功回测后自动重建，但不会修改任何历史实验。

## 回测口径

- 每周最后一个交易日收盘读取信号，下一个交易日开盘执行。
- Ridge 与 CNN-GRU 分别使用自己的10日半衰期平滑预测。
- 使用当前预测之前的252个交易日预测做滞后标准化。
- 状态改变时调仓；相同状态不重复交易。
- 初始资金100万元，单边手续费0.2%，现金收益和无风险利率均为0。
- 中证1000点位作为代理资产单位价格，不模拟股指期货乘数、保证金或真实指数产品。

本层用于检验模型的历史经济意义，不代表可直接执行的实盘收益。
