# 05 输出层

本层将已经完成训练、版本化回测并通过验收的市场模型发布为稳定的日度仓位信号，供组合优化与风控项目读取。发布频率与模型 20 日预测标签相互独立：模型仍预测未来 20 日，但每个交易日都会生成一条可供下游选择的仓位信号。

## 发布约束

- 必须明确指定第四层的 `run-id`，不能隐式读取“最新报告”。
- 指定实验必须存在 `config_snapshot.yaml`、`run_manifest.json` 和验收报告。
- 验收报告必须 `fail = 0`，且必须包含准备发布的模型。
- `release-id` 一经创建禁止覆盖。
- 只有显式添加 `--set-current` 才会更新默认正式版本。

发布 Ridge：

```powershell
python 05输出层/publish_market_signal.py `
  --model ridge `
  --run-id bull90_neutral60_bear30_v1 `
  --release-id market_ridge_20d_daily_20260814_906030_v1 `
  --set-current
```

发布 CNN-GRU：

```powershell
python 05输出层/publish_market_signal.py `
  --model cnn_gru `
  --run-id bull90_neutral60_bear30_v1 `
  --release-id market_cnn_gru_20d_daily_20260814_906030_v1
```

## 输出结构

```text
05输出层/exports/
├─ current.json
└─ releases/{release-id}/
   ├─ market_signal.parquet
   └─ manifest.json
```

发布清单会记录 `source_backtest_run_id`，并保存回测配置、回测清单、验收报告、模型预测和训练配置的哈希，从而将正式信号绑定到一套可复现的仓位策略。

## 信号协议

`market_signal.parquet` 包含：

- `signal_date`：信号日期。
- `execution_date`：计划执行日期；最新信号可能为空。
- `market_code`：`000852.SH`。
- `forecast_return`：平滑后的未来20日预测收益。
- `signal_zscore`：相对历史预测的标准化强度。
- `market_state`：`bear`、`neutral` 或 `bull`。
- `target_equity_exposure`：指定回测策略对应的目标股票仓位。
- `horizon_days`：预测周期，当前为20。

下游必须遵守 `signal_available_after=market_close` 和 `execution_lag_trading_days=1`，不得将 T 日收盘后信号用于 T 日开盘成交。

下游周度或月度调仓由组合系统根据官方交易日历选择完整结束的交易周或交易月；本层不再提前删除其他交易日信号。
