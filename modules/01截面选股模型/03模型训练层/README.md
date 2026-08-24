# 03 模型训练层

> 使用 Tushare 因子宽表构建 open-to-open 标签，通过 LightGBM Quarterly PIT V2 生成可冻结、可增量、可回测的截面预测信号。

## 当前状态

截至 2026-08-01：

- 正式训练入口为 `main_train_v2.py`。
- V1 Walk-forward 入口、切分器和训练器已退役。
- 5 日、20 日和 60 日正式配置统一放在 `configs/production/`。
- LightGBM V2 调参和稳健性验证脚本已纳入本层。
- Tushare 版 5 日、20 日和 60 日模型均已完成 27 个季度的 Freeze 全量训练。
- 三周期滞后 IC 融合已生成并完成回测与正式发布。

`data_constructor_v1.py` 虽保留历史文件名，但仍是 V2 共用的数据构造器，不代表训练流程仍使用 V1。

## 数据流

```text
02因子库/processed_data/
├── factors/technical/*.parquet
├── factors/financial/*.parquet
└── market_data/{open,close,...}.parquet
             │
             ▼
     DataConstructorV1
     特征 + open-to-open 标签
     ST过滤 + 训练集利润过滤
             │
             ▼
     QuarterlySplitterV2
     自然季度固定模型切分
             │
             ▼
     QuarterlyTrainerV2
     LightGBM训练、预测、平滑
             │
             ▼
 experiments/{exp_id}_v2/
 predictions + models + freeze state
```

## 目录结构

```text
03模型训练层/
├── configs/
│   ├── default_config.yaml
│   ├── horizon5_config.yaml
│   ├── horizon20_config.yaml
│   ├── horizon60_config.yaml
│   └── production/
│       ├── horizon5_profit20_tuned_config.yaml
│       ├── horizon20_profit20_tuned_config.yaml
│       └── horizon60_profit20_tuned_config.yaml
├── dataset/
│   ├── data_constructor_v1.py
│   └── quarterly_splitter_v2.py
├── models/
│   ├── base_model.py
│   ├── lightgbm_model.py
│   └── lightgbm_rank_model.py
├── training/
│   └── quarterly_trainer_v2.py
├── tuning/
│   ├── tune_lgbm_v2.py
│   ├── validate_lgbm_v2_params.py
│   ├── tuning_results/
│   └── README.md
├── main_train_v2.py
├── fuse_predictions.py
├── experiments/                 # 运行时生成，不进入Git
└── README.md
```

## 核心逻辑

### Open-to-open 标签

T 日收盘后才能得到完整的 T 日因子，因此不能假设以 T 日收盘价成交。

```text
T日因子 ──→ T+1日开盘买入 ──→ T+(horizon+1)日开盘卖出
```

标签定义：

```text
label(T) = open[T + horizon + 1] / open[T + 1] - 1
```

例如 horizon 为 20 时，以 T+1 日开盘价买入，以 T+21 日开盘价卖出。

### Quarterly PIT V2

每个自然季度对应一个固定模型：

```text
训练窗口        Gap        验证窗口       Gap        预测季度
约3年历史 ───────────→ 约2个月 ───────────→ YYYYQn
```

- 预测季度内所有交易日使用同一个模型。
- 训练窗口默认 3 年，验证窗口默认 2 个月。
- Gap 默认等于 `horizon + 1` 个交易日。
- 预测结果是一条连续 PIT 信号链，不再区分 test 和 live 两套输出。
- `--start-date` 限制预测起点，不会截断模型所需的历史训练窗口。
- `--end-date` 同时作为本次运行的数据截止日和 as-of 日期。

### 股票池过滤

- ST 状态来自 `01数据/data/tushare_data/st_status.parquet`。
- ST 股票不会进入训练、验证和预测截面。
- `profit_filter_pct: 0.2` 表示训练和验证样本过滤净利润 TTM 最低 20% 候选，并排除其中亏损股票。
- 利润过滤不作用于预测截面，避免人为缩小模型实际选股空间。

### Freeze 增量模式

同一个实验需要追加新日期时使用 `--freeze`：

- 读取已有 `predictions.parquet` 和 `summary.parquet`。
- 复用 `state/models/` 中已冻结的季度模型。
- 跳过已经完成的预测日期。
- 只追加缺失预测，不重算历史信号。
- 使用配置哈希阻止不同关键参数误写入同一个实验。

如果更换模型参数、因子或训练逻辑，应该使用新的 `exp-id`。只有明确要重建同一个实验的 Freeze 状态时，才使用 `--reset-freeze`。

### 预测平滑

V2 使用固定半衰期对每只股票的预测序列做指数平滑：

```text
smoothed[t] = alpha × pred[t] + (1-alpha) × smoothed[t-1]
```

默认 `smooth_halflife: 10`。平滑结果写入 `smoothed_predictions.parquet`，用于降低信号波动和换手率。

## 配置管理

### 基准配置

```text
configs/horizon5_config.yaml
configs/horizon20_config.yaml
configs/horizon60_config.yaml
```

用于调参起点、基线实验和开发验证。

### Production 配置

```text
configs/production/horizon5_profit20_tuned_config.yaml
configs/production/horizon20_profit20_tuned_config.yaml
configs/production/horizon60_profit20_tuned_config.yaml
```

Production 配置是正式训练入口：

- 使用项目相对路径。
- 保存已验证并晋升的模型参数。
- 可以追溯到对应的 `tuning_metadata.study_name` 和 trial。

### 调参结果

```text
tuning/tuning_results/{study_name}/final_recommended_config.yaml
```

该文件是调参程序的原始产物和审计记录。正式使用前将结果晋升到 `configs/production/`，不要直接手工修改调参结果目录。

## 正式训练

以下命令均在项目根目录执行。

### 20 日模型

```powershell
python 03模型训练层/main_train_v2.py `
  --config horizon20_profit20_tuned_config.yaml `
  --exp-id lgbm20_tushare_profit20 `
  --start-date 2020-01-01 `
  --freeze -y
```

实际实验目录会自动追加 `_v2`：

```text
03模型训练层/experiments/lgbm20_tushare_profit20_v2/
```

### 5 日模型

```powershell
python 03模型训练层/main_train_v2.py `
  --config configs/production/horizon5_profit20_tuned_config.yaml `
  --exp-id lgbm5_tushare_profit20 `
  --start-date 2020-01-01 `
  --freeze -y
```

### 60 日模型

```powershell
python 03模型训练层/main_train_v2.py `
  --config horizon60_profit20_tuned_config.yaml `
  --exp-id lgbm60_tushare_profit20 `
  --start-date 2020-01-01 `
  --freeze -y
```

### As-of 实验

```powershell
python 03模型训练层/main_train_v2.py `
  --config horizon20_profit20_tuned_config.yaml `
  --exp-id lgbm20_asof_20260701 `
  --start-date 2026-04-01 `
  --end-date 2026-07-01 -y
```

### 月度增量

对同一个实验使用相同配置和 `exp-id`：

```powershell
python 03模型训练层/main_train_v2.py `
  --config horizon20_profit20_tuned_config.yaml `
  --exp-id lgbm20_tushare_profit20 `
  --start-date 2020-01-01 `
  --end-date 2026-08-31 `
  --freeze -y
```

## 训练参数

| 参数 | 含义 |
|:---|:---|
| `--config` | YAML 配置文件 |
| `--exp-id` | 实验 ID；未以 `_v2` 结尾时自动追加 |
| `--start-date` | 预测开始日期 |
| `--end-date` | 数据/as-of 截止日期 |
| `--train-window` | 覆盖 V2 训练窗口，例如 `3Y` |
| `--valid-window` | 覆盖验证窗口，例如 `2M` |
| `--gap` | 覆盖交易日隔离长度 |
| `--smooth-halflife` | 覆盖预测平滑半衰期 |
| `--horizon` | 覆盖标签周期，并自动更新默认 Gap |
| `--freeze` | 复用冻结模型并追加预测 |
| `--reset-freeze` | 重建同一实验的 Freeze 状态 |
| `-y` | 跳过交互确认 |

## 实验输出

```text
experiments/{exp_id}_v2/
├── config.yaml
├── quarterly_splits.parquet
├── predictions.parquet
├── smoothed_predictions.parquet
├── summary.parquet
├── quarterly_rank_ic_v2.png
├── feature_importance_v2.png
├── models/
│   └── model_YYYYQn_fold_NNN.pkl
├── feature_importance/
│   └── importance_YYYYQn_fold_NNN.csv
└── state/
    ├── freeze_manifest.json
    └── models/
        └── model_YYYYQn.pkl
```

主要数据文件：

| 文件 | 用途 |
|:---|:---|
| `predictions.parquet` | 原始 PIT 预测链 |
| `smoothed_predictions.parquet` | 固定半衰期平滑后的预测链 |
| `summary.parquet` | 各季度训练、验证和 Rank IC 摘要 |
| `quarterly_splits.parquet` | 每个季度的训练/验证/预测边界 |
| `freeze_manifest.json` | Freeze 配置哈希、完成季度和日期状态 |

## LightGBM 调参

调参使用代表季度搜索参数，避免每个 trial 执行完整历史训练。

```powershell
python 03模型训练层/tuning/tune_lgbm_v2.py `
  --base-config 03模型训练层/configs/horizon20_config.yaml `
  --study-name lgbm20_v2_example `
  --n-trials 20
```

对 top 参数执行非重合季度验证和邻近扰动检查：

```powershell
python 03模型训练层/tuning/validate_lgbm_v2_params.py `
  --study-name lgbm20_v2_example `
  --base-config 03模型训练层/configs/horizon20_config.yaml `
  --top-k 3
```

调参输出：

```text
tuning/tuning_results/{study_name}/
├── study_results.csv
├── top_trials.csv
├── tuning_summary.md
├── validation_results.csv
└── final_recommended_config.yaml
```

完整说明见 [tuning/README.md](tuning/README.md)。

## 多周期融合

单模型全部完成后，融合它们的 `smoothed_predictions.parquet`：

```powershell
python 03模型训练层/fuse_predictions.py `
  --exps lgbm5_tushare_profit20_v2 lgbm20_tushare_profit20_v2 lgbm60_tushare_profit20_v2 `
  --base-idx 1 `
  --output-exp ensemble_5d_20d_60d_profit20_v2
```

融合流程：

1. 按日期和股票代码取三个模型的共同样本。
2. 每个模型每日做截面 Rank 标准化。
3. 使用滞后的历史 IC 均值生成动态权重。
4. 以 `base-idx` 模型的 `horizon + 1` 作为权重滞后长度；当前 20d 基准使用 21 个交易日。
5. 输出兼容回测层的预测文件，并生成兼容 05 输出层的标准 `config.yaml`。

融合实验的 `data.label.horizon` 取 `base-idx` 对应模型的 horizon。各输入实验可以使用不同 horizon，但标签公式和开盘价口径必须一致。

融合输出：

```text
experiments/{output_exp}/
├── predictions.parquet
├── smoothed_predictions.parquet
├── config.yaml
└── fusion_config.yaml
```

`config.yaml` 供回测层和输出层读取；`fusion_config.yaml` 保存来源实验、IC 滞后和最终权重明细。

## 防泄漏检查清单

- 因子值只使用 T 日及以前可知数据。
- 财务因子在因子层按公告日完成 PIT 对齐。
- 标签从 T+1 日开盘开始计算。
- 训练、验证和预测区间之间保留 `horizon + 1` 个交易日 Gap。
- IC 融合权重只使用已经兑现的历史收益。
- Freeze 模式不得覆盖已经生成的历史预测。
- 更换关键参数时使用新的实验 ID。

## 后续工作

- 按月使用 Freeze 模式追加新交易日预测。
- 持续比较单模型与融合模型的近期回撤和 Top20 稳定性。
- 增加可自动执行的最小训练与 Freeze 回归测试。

---

*最后更新：2026-08-01*
*维护者：蒋大王*
