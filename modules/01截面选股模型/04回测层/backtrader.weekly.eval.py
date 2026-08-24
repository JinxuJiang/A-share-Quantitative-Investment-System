"""
周频多因子回测入口。

保持 backtrader.eval.py 的模型、预测、选股、仓位、成本和止损逻辑不变，
仅将调仓日切换为每周最后一个实际交易日，并把报告隔离到 reports/weekly。

示例：
    python 04回测层/backtrader.weekly.eval.py \
      --exp-id ensemble_5d_20d_60d_profit20_v2 --use-smooth
"""

import os
import runpy
from pathlib import Path


if __name__ == "__main__":
    os.environ["BACKTEST_REBALANCE_FREQUENCY"] = "weekly_last"
    os.environ["BACKTEST_REPORT_VARIANT"] = "weekly"
    runpy.run_path(
        str(Path(__file__).with_name("backtrader.eval.py")),
        run_name="__main__",
    )
