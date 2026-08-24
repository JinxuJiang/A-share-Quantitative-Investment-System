# 05 输出层

本层把已经完成训练和回测验收的实验发布为稳定的截面 alpha，供后续组合仓库读取。进入本层的实验均视为正式版本，不区分候选和生产版本。

## 当前状态

- 发布脚本和 `stock_alpha_v1` 契约已实现并完成端到端验证。
- 当前正式 release：`alpha_20d_tushare_profit20`
- 来源实验：`lgbm20_tushare_profit20_v2`
- Alpha 日期：2020-01-02 至 2026-07-28
- `current.json` 已指向上述 release。

## 发布

```powershell
python 05输出层\publish_alpha.py --exp-id lgbm20_tushare_profit20_v2 --release-id alpha_20d_tushare_profit20_20260728_v1
```

发布程序读取：

```text
03模型训练层/experiments/{exp_id}/smoothed_predictions.parquet
03模型训练层/experiments/{exp_id}/config.yaml
```

`exp_id` 可以是单模型实验，也可以是 `fuse_predictions.py` 生成的融合实验。融合 Alpha 的 `horizon_days` 采用融合时 `base-idx` 对应模型的 horizon。

每次发布会创建一个不可覆盖的 release，并自动更新 `current.json`：

```text
05输出层/
├── publish_alpha.py
├── README.md
└── exports/
    ├── releases/
    │   └── {release_id}/
    │       ├── stock_alpha.parquet
    │       └── manifest.json
    └── current.json
```

`exports/` 是运行产物，不提交到 Git。历史 release 不允许覆盖；需要发布新结果时必须使用新的 `release-id`。

## stock_alpha.parquet

| 字段 | 含义 |
| --- | --- |
| `signal_date` | 信号日期 |
| `stock_code` | Tushare 股票代码 |
| `alpha_score` | `pred_score_smooth` 发布后的标准字段名 |
| `alpha_rank` | 当日截面百分位，越接近 1 表示模型排名越高 |
| `horizon_days` | 模型预测周期 |

发布文件不包含 `actual_return`、训练区间或验证区间，避免下游误用未来收益和模型内部字段。

## manifest.json

记录 release 来源、模型周期、数据范围、信号执行滞后、Git 状态、行数及文件 SHA256。下游仓库应先校验 `schema_version=stock_alpha_v1`。

信号约定：

- 信号在 `signal_date` 收盘后可用；
- 最早在下一交易日执行；
- `execution_lag_trading_days=1`。

## current.json

`current.json` 不是按时间排序的列表，而是当前默认正式 release 的明确指针。每次成功发布后由脚本自动更新。

组合仓库未指定 `release-id` 时读取 `current.json`；需要复现历史结果时直接指定对应的 release 目录。

## 详细文档

- [设计原理与逻辑架构](../docs/05.1_设计原理与逻辑架构.md)
- [工程实现与规范](../docs/05.2_工程实现与规范.md)
- [运维与变更日志](../docs/05.3_运维与变更日志.md)
