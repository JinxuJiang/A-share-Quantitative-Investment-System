# 03 组合风控层

本层逐周读取 02 层账户实际权重 `target_weight` 和 01 层对应日期的协方差，生成全部历史时期的 5 日 VaR/ES。5 日代表从本周信号形成后到下一次周度调仓前的风险。

本层只测量风险，不设置新的 VaR/ES 阈值，也不会再次自动减仓。02 层可能因账户级单票或行业硬约束无法满足市场目标而增加现金，本层使用其发布的实际股票仓位计算风险。

## 运行

```powershell
python .\03组合风控层\assess_risk.py
```

## 正式输出

```text
outputs/releases/risk_history_起始日_结束日_v1/
├─ risk.parquet
├─ config.yaml
└─ manifest.json
```

`risk.parquet` 每期一行，只保留 5 日指标：

- `portfolio_volatility_5d`：账户 5 日预测波动率；
- `var_95_5d`、`es_95_5d`：95% VaR/ES；
- `var_99_5d`、`es_99_5d`：99% VaR/ES；
- 带 `_pct` 的字段是便于阅读的百分数。

组合波动率为：

$$
\sigma_p=\sqrt{w^\top\Sigma w}
$$

上游协方差按时间线性规则转换成 5 日口径。现金收益和波动率暂设为 0，组合预期收益暂设为 0；Alpha z-score 不直接当作收益率。

95% VaR 表示约有 5% 的概率在下一次调仓前亏损超过该数值；95% ES 表示进入最差 5% 情形后的平均亏损。ES 不是最大亏损。
