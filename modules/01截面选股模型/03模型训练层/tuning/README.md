# LightGBM V2 调参与参数晋升

本目录提供两个脚本：

- `tune_lgbm_v2.py`：在固定代表季度上用 Optuna 搜索参数。
- `validate_lgbm_v2_params.py`：对 top 参数做非重合季度验证和邻近扰动稳定性检查。

默认调参季度：

```text
2020Q2, 2021Q4, 2022Q2, 2022Q4, 2024Q1, 2025Q4
```

输出目录：

```text
03模型训练层/tuning/tuning_results/{study_name}/
```

保留文件：

```text
study_results.csv
top_trials.csv
tuning_summary.md
validation_results.csv
final_recommended_config.yaml
```

缓存目录：

```text
03模型训练层/tuning/.cache/
```

缓存默认在脚本结束后删除；需要保留时加 `--keep-cache`。

以下命令均在项目根目录执行。

## 用法

调参：

```powershell
python 03模型训练层/tuning/tune_lgbm_v2.py `
  --base-config 03模型训练层/configs/horizon20_config.yaml `
  --study-name lgbm20_v2_001 `
  --n-trials 20
```

验证：

```powershell
python 03模型训练层/tuning/validate_lgbm_v2_params.py `
  --study-name lgbm20_v2_001 `
  --base-config 03模型训练层/configs/horizon20_config.yaml `
  --top-k 3
```

验证完成后会生成：

```text
03模型训练层/tuning/tuning_results/lgbm20_v2_001/final_recommended_config.yaml
```

该文件是调参原始产物和审计记录，不作为长期正式入口。确认参数后，将其晋升到：

```text
03模型训练层/configs/production/
```

晋升时：

- 保留模型参数、训练参数和 `tuning_metadata`。
- 将数据及输出目录改为项目相对路径。
- 不修改 `tuning_results/` 中的原始文件。
- 正式训练统一使用 `configs/production/` 中的配置。
