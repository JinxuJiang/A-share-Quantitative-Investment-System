# 02 组合优化层

本层读取 01 层日度历史决策输入，并根据官方交易日历分别选择完整结束的周度与月度 `signal_date`。两个频率各自维护上一期目标权重和换手惩罚，互不串用。

目标函数为：

```text
最小化 = 风险权重 × 标准化组合方差 - Alpha权重 × 标准化Alpha效用
```

当前版本只做多，并直接在账户权重口径执行风险约束：单票最高占账户 10%，单一行业最高占账户 25%。市场模型给出的 0% / 45% / 90% 是股票总仓位目标上限。

```text
正常情况：stock_exposure = market_target_exposure
约束不可行：stock_exposure = maximum_feasible_exposure
cash_weight = 1 - stock_exposure
```

优化器内部仍保留合计为 100% 的 `base_weight` 便于比较股票之间的相对配置，但单票和行业上限均在 `target_weight`（账户实际权重）上验收。无法完整部署市场目标时，风险层不改变当期候选名单，而是将不能安全部署的部分留在现金。

## 运行

```powershell
python .\02组合优化层\optimize.py --frequency both

# 也可以只生成一个频率
python .\02组合优化层\optimize.py --frequency weekly
python .\02组合优化层\optimize.py --frequency monthly
```

## 正式输出

```text
outputs/weekly/releases/portfolio_weekly_history_起始日_结束日_v1/
outputs/monthly/releases/portfolio_monthly_history_起始日_结束日_v1/
├─ weights.parquet
├─ portfolio_summary.parquet
├─ config.yaml
└─ manifest.json
```

- `weights.parquet`：最重要的回测输入。每期行数等于上游决策配置的 Top N，`target_weight` 是整个账户的精确股票权重，`is_active` 表示是否为目标活跃持仓。
- `portfolio_summary.parquet`：每期一行，包含市场目标仓位、最大可行仓位、实际股票仓位、仓位缺口、现金、活跃股票数、预测波动率和求解器结果。
- `base_weight_pct` 与 `target_weight_pct` 只供人工查看；回测必须使用未四舍五入的 `target_weight`。

默认配置的 `turnover_penalty` 为 0.05，以相邻两期账户目标权重的绝对变化之和近似预计交易权重；候选池外的上一期持仓计入强制卖出。该惩罚只稳定股票间配置，不放松市场总仓位、单票或行业硬约束。若需复现无惩罚旧基线，可显式将该参数设为 0。
