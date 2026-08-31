# 03 组合风控层

本层分别读取 02 层周度和月度账户实际权重 `target_weight`，并使用 01 层同日协方差生成风险历史。周度轨道计算 5 日 VaR/ES，月度轨道计算 20 日 VaR/ES。

本层启用 VaR 风险预算缩放。缩放只允许降低股票总仓位，不改变股票之间的相对权重，也不会把仓位放大到上游目标以上。默认使用周度 7.5% 和月度 15% 的 95% VaR 预算；如需只测量风险，可在配置中将 `risk.scaling.enabled` 设为 `false`。

## 运行

```powershell
python .\03组合风控层\assess_risk.py --frequency both
```

## 正式输出

```text
outputs/weekly/releases/risk_weekly_history_起始日_结束日_v1/
outputs/monthly/releases/risk_monthly_history_起始日_结束日_v1/
├─ risk.parquet
├─ config.yaml
└─ manifest.json
```

`risk.parquet` 每期一行，并记录对应频率的风险期限、原始风险和可选缩放结果：

- `portfolio_volatility_5d`：账户 5 日预测波动率；
- `var_95_5d`、`es_95_5d`：95% VaR/ES；
- `var_99_5d`、`es_99_5d`：99% VaR/ES；
- `risk_scale`：风险预算缩放系数；关闭缩放时恒为 1；
- `scaled_stock_exposure`、`scaled_cash_weight`：缩放后的账户仓位；
- 带 `_pct` 的字段是便于阅读的百分数。

组合波动率为：

$$
\sigma_p=\sqrt{w^\top\Sigma w}
$$

上游协方差按时间线性规则转换成 5 日口径。现金收益和波动率暂设为 0，组合预期收益暂设为 0；Alpha z-score 不直接当作收益率。

95% VaR 表示约有 5% 的概率在下一次调仓前亏损超过该数值；95% ES 表示进入最差 5% 情形后的平均亏损。ES 不是最大亏损。
