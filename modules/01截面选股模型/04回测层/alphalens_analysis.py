"""
Alphalens 因子有效性分析

使用 alphalens-reloaded 库进行专业的因子分析

使用方法：
    # 分析原始预测
    python alphalens_analysis.py --exp-id exp_001
    
    # 分析平滑后的预测（输出带_smooth后缀）
    python alphalens_analysis.py --exp-id exp_001 --use-smooth
    
对比效果：
    - 平滑后IC可能略降，但换手率大幅降低，信号更稳定
    - 对比 reports/exp_001/ic_analysis.png vs ic_analysis_smooth.png
"""

import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import alphalens as al
import alphalens.plotting as apl

from utils import load_predictions, load_close_prices, ensure_report_dir

# 忽略警告
warnings.filterwarnings('ignore')


def run_alphalens_analysis(exp_id: str, periods: tuple = (20,), quantiles: int = 10, use_smooth: bool = False):
    """
    运行 Alphalens 因子分析
    
    Args:
        exp_id: 实验ID
        periods: 远期收益周期，默认(20,)表示20日收益
        quantiles: 分位数分组数，默认10组
        use_smooth: 是否使用平滑后的预测（半衰期加权），输出文件带_smooth后缀
    """
    suffix = " (平滑)" if use_smooth else ""
    print("=" * 70)
    print(f"Alphalens 因子分析 - {exp_id}{suffix}")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    predictions = load_predictions(exp_id, use_smooth=use_smooth)
    prices = load_close_prices()
    
    # 2. 准备 Alphalens 格式数据
    print("\n[2/3] 准备 Alphalens 数据格式...")
    
    # factor: MultiIndex Series (date, asset) -> value
    pred_col = 'pred_score_smooth' if use_smooth else 'pred_score'
    factor = predictions.set_index(['date', 'stock_code'])[pred_col]
    
    # prices: DataFrame, index=date, columns=assets
    factor_dates = factor.index.get_level_values('date').unique()
    factor_stocks = factor.index.get_level_values('stock_code').unique()
    
    prices = prices.loc[prices.index.isin(factor_dates), 
                        prices.columns.isin(factor_stocks)]
    
    print(f"  Factor: {len(factor)} 条记录, {len(factor_dates)} 个交易日")
    print(f"  Prices: {prices.shape[0]} 天 x {prices.shape[1]} 只股票")
    
    # 3. 运行 Alphalens 分析
    print("\n[3/3] 运行 Alphalens 分析...")
    
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        periods=periods,
        quantiles=quantiles,
        groupby=None,
        binning_by_group=False,
        filter_zscore=5,
        max_loss=0.35
    )
    
    print(f"  有效因子数据: {len(factor_data)} 条")
    
    # 4. 计算各项指标
    print("\n" + "=" * 70)
    print("分析结果摘要")
    print("=" * 70)
    
    period_str = f'{periods[0]}D'
    
    # Rank IC (Spearman) - Alphalens 默认就是 Rank IC
    ic = al.performance.factor_information_coefficient(factor_data)
    
    # 收益分析
    mean_return_by_q, std_err = al.performance.mean_return_by_quantile(
        factor_data, by_group=False, demeaned=False
    )
    
    # 换手率分析
    turnover = al.performance.factor_rank_autocorrelation(factor_data)
    
    # 5. 打印关键指标
    ic_mean = ic[period_str].mean()
    ic_std = ic[period_str].std()
    ir = ic_mean / ic_std if ic_std > 0 else 0
    
    print(f"\n【Rank IC 分析】(Spearman)")
    print(f"  Rank IC 均值:  {ic_mean:>8.4f}  ({'优秀' if ic_mean > 0.05 else '良好' if ic_mean > 0.03 else '一般'})")
    print(f"  Rank IC 标准差:{ic_std:>8.4f}")
    print(f"  IR:            {ir:>8.4f}  ({'稳定' if ir > 0.5 else '一般'})")
    print(f"  IC > 0 占比:   {ic[period_str].gt(0).mean():>8.2%}")
    
    # 分层收益
    print(f"\n【分层收益】(按 pred_score 分 {quantiles} 组)")
    top_return = mean_return_by_q.loc[quantiles, period_str]
    bottom_return = mean_return_by_q.loc[1, period_str]
    spread = top_return - bottom_return
    
    print(f"  Top组 (Q{quantiles}):    {top_return:>8.4f}")
    print(f"  Bottom组 (Q1): {bottom_return:>8.4f}")
    print(f"  多空收益差:    {spread:>8.4f}  ({'单调性良好' if spread > 0 else '单调性差'})")
    
    print(f"\n  各分位数平均收益:")
    for q in range(1, quantiles + 1):
        ret = mean_return_by_q.loc[q, period_str]
        bar = "█" * int(abs(ret) * 200) if abs(ret) < 0.025 else "█" * 5
        print(f"    Q{q:02d}: {ret:>8.4f} {bar}")
    
    # 换手率
    print(f"\n【换手率分析】")
    print(f"  因子自相关:    {turnover.mean():>8.4f}  (越高因子越稳定)")
    print(f"  日均换手率:    {1 - turnover.mean():>8.4f}")
    
    # 6. 生成报告
    print("\n" + "=" * 70)
    print("生成可视化报告...")
    print("=" * 70)
    
    report_dir = ensure_report_dir(exp_id)
    file_suffix = "_smooth" if use_smooth else ""
    generate_alphalens_report(factor_data, ic, mean_return_by_q, turnover, 
                               report_dir, periods, quantiles, file_suffix)
    
    print(f"\n[完成] 报告已保存至: {report_dir}/")
    
    return factor_data, ic, mean_return_by_q


def generate_alphalens_report(factor_data, ic, mean_return_by_q, turnover,
                               report_dir: str, periods: tuple, quantiles: int, file_suffix: str = ""):
    """生成 Alphalens 可视化报告 - 输出3个PNG"""
    
    period_str = f'{periods[0]}D'
    
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    
    # 1. IC 分析图 - 模仿 notebook 风格
    fig = plt.figure(figsize=(16, 10))
    
    # 图1: IC 时间序列 (使用 Alphalens)
    ax1 = fig.add_subplot(311)
    al.plotting.plot_ic_ts(ic, ax=[ax1])
    ax1.set_title(f'IC Time Series ({period_str})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 图2: IC 分布 (手动绘制)
    ax2 = fig.add_subplot(323)
    ic[period_str].hist(bins=50, ax=ax2, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(ic[period_str].mean(), color='r', linestyle='--', linewidth=2,
                label=f"Mean: {ic[period_str].mean():.4f}")
    ax2.set_title(f'IC Distribution ({period_str})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('IC')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 累计 IC (手动绘制)
    ax3 = fig.add_subplot(324)
    ic_cum = ic.cumsum()
    ic_cum.plot(ax=ax3, color='steelblue', linewidth=1.5)
    ax3.set_title('Cumulative IC', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative IC')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{report_dir}/ic_analysis{file_suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] IC分析图: ic_analysis{file_suffix}.png")
    
    # 2. 收益分析图 - 三个子图：柱状图 + 分组累积 + 多空累积
    fig = plt.figure(figsize=(16, 14))
    
    # 子图1: 分位数平均收益柱状图
    ax1 = fig.add_subplot(311)
    mean_returns = mean_return_by_q[period_str]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(mean_returns)))
    bars = ax1.bar(range(1, len(mean_returns) + 1), mean_returns.values, 
                   color=colors, edgecolor='black')
    ax1.set_title('Mean Return by Quantile', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Quantile')
    ax1.set_ylabel('Mean Return')
    ax1.set_xticks(range(1, len(mean_returns) + 1))
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)
    
    # 子图2: 各分位数累积收益曲线 (使用 Alphalens 标准函数)
    ax2 = fig.add_subplot(312)
    try:
        quantile_returns_ts = al.performance.mean_return_by_quantile(
            factor_data, by_date=True
        )[0]
        
        # 使用 Alphalens 自带的标准绘图函数
        al.plotting.plot_cumulative_returns_by_quantile(quantile_returns_ts, period=period_str, ax=ax2)
        ax2.set_title('Cumulative Returns by Quantile', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax2.transAxes)
    
    # 子图3: 多空收益曲线 (手动绘制 - 使用简单累加避免爆炸)
    ax3 = fig.add_subplot(313)
    try:
        # 计算多空收益: Top组 - Bottom组
        long_returns = quantile_returns_ts.loc[quantiles, period_str]
        short_returns = quantile_returns_ts.loc[1, period_str]
        long_short = long_returns - short_returns
        
        # 累积多空收益 (简单累加 cumsum，不是复利 cumprod)
        long_short_cum = long_short.cumsum()
        
        long_short_cum.plot(ax=ax3, color='red', linewidth=2, label='Long-Short (Q10 - Q01)')
        ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax3.set_title('Cumulative Long-Short Returns (Q10 - Q01)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Return')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 添加统计信息
        final_return = long_short_cum.iloc[-1]
        ax3.text(0.02, 0.95, f'Total: {final_return:.2f} | Daily: {long_short.mean():.4f}', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax3.transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{report_dir}/returns_analysis{file_suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 收益分析图: returns_analysis{file_suffix}.png")
    
    # 3. 换手率分析 (手动绘制 - Alphalens 没有标准 turnover 时间序列图)
    fig, ax = plt.subplots(figsize=(14, 5))
    turnover.plot(ax=ax, color='steelblue', alpha=0.7, linewidth=1, label='Daily')
    turnover.rolling(20).mean().plot(ax=ax, color='red', linewidth=2, label='20-day MA')
    ax.axhline(y=turnover.mean(), color='green', linestyle='--', alpha=0.7,
               label=f'Mean={turnover.mean():.4f}')
    ax.set_title('Factor Rank Autocorrelation (Factor Stability)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加说明文字
    ax.text(0.02, 0.98, 'Higher = More Stable\nLower = Higher Turnover', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{report_dir}/turnover_analysis{file_suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 换手率分析图: turnover_analysis{file_suffix}.png")
    
    # 4. 生成 HTML 报告
    generate_html_report(ic, mean_return_by_q, turnover, report_dir, periods, quantiles, file_suffix)
    print(f"  [OK] 综合报告: alphalens_report{file_suffix}.html")


def generate_html_report(ic, mean_return_by_q, turnover, report_dir, periods, quantiles, file_suffix: str = ""):
    """生成 HTML 摘要报告"""
    
    period_str = f'{periods[0]}D'
    
    # 计算指标
    ic_mean = ic[period_str].mean()
    ic_std = ic[period_str].std()
    ir = ic_mean / ic_std if ic_std > 0 else 0
    
    mean_returns = mean_return_by_q[period_str]
    top_return = mean_returns.loc[quantiles]
    bottom_return = mean_returns.loc[1]
    spread = top_return - bottom_return
    
    # 生成分位数表格行
    quantile_rows = ""
    for q in range(1, quantiles + 1):
        ret = mean_returns.loc[q]
        spread_vs_q1 = ret - bottom_return
        quantile_rows += f"<tr><td>Q{q}</td><td>{ret:.4f}</td><td>{spread_vs_q1:.4f}</td></tr>\n"
    
    title_suffix = " (平滑)" if file_suffix == "_smooth" else ""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Alphalens Report{title_suffix} - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; 
                      padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics {{ display: flex; flex-wrap: wrap; justify-content: space-around; margin: 20px 0; }}
        .metric-box {{ text-align: center; padding: 20px; background: #f8f9fa; 
                      border-radius: 8px; min-width: 150px; margin: 10px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .good {{ color: #4CAF50; }}
        .warning {{ color: #FF9800; }}
        .bad {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .summary {{ background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .explanation {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1> Alphalens 因子分析报告{title_suffix}</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2> 关键指标</h2>
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value {'good' if ic_mean > 0.05 else 'warning' if ic_mean > 0.03 else 'bad'}">{ic_mean:.4f}</div>
                <div class="metric-label">Rank IC (Spearman)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {'good' if ir > 0.5 else 'warning' if ir > 0.3 else 'bad'}">{ir:.4f}</div>
                <div class="metric-label">IR (IC/Std)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {'good' if spread > 0.03 else 'warning' if spread > 0.02 else 'bad'}">{spread:.4f}</div>
                <div class="metric-label">Long-Short Spread</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{turnover.mean():.4f}</div>
                <div class="metric-label">Factor Stability</div>
            </div>
        </div>
        
        <div class="summary">
            <h3> 结论</h3>
            <ul>
                <li><strong>Rank IC = {ic_mean:.4f}</strong>: {'因子预测能力优秀' if ic_mean > 0.05 else '因子预测能力良好' if ic_mean > 0.03 else '因子预测能力一般'}</li>
                <li><strong>IR = {ir:.4f}</strong>: {'因子稳定性优秀' if ir > 0.7 else '因子稳定性良好' if ir > 0.5 else '因子稳定性一般'}</li>
                <li><strong>Long-Short = {spread:.4f}</strong>: {'分层效果优秀' if spread > 0.03 else '分层效果良好' if spread > 0.02 else '分层效果一般'}</li>
            </ul>
        </div>
        
        <h2> Rank IC 分析</h2>
        <img src="ic_analysis{file_suffix}.png" alt="IC Analysis">
        <div class="explanation">
            <strong>说明：</strong>Rank IC 是 Spearman 相关系数，衡量预测分数排名与未来收益排名的相关性。IC > 0.05 为优秀，IR > 0.5 为稳定。
        </div>
        
        <h2> 收益分析</h2>
        <img src="returns_analysis{file_suffix}.png" alt="Returns Analysis">
        
        <h3>各分位数收益详情</h3>
        <table>
            <tr><th>Quantile</th><th>Mean Return</th><th>vs Q1 Spread</th></tr>
            {quantile_rows}
        </table>
        
        <h2> 换手率/稳定性分析</h2>
        <img src="turnover_analysis{file_suffix}.png" alt="Turnover Analysis">
        <div class="explanation">
            <strong>说明：</strong>Factor Rank Autocorrelation（因子秩自相关）衡量相邻两期因子排名的相关性。
            <ul>
                <li>值越高（接近1）：因子越稳定，换手率越低</li>
                <li>值越低（接近0）：因子变化大，换手率越高</li>
                <li>本因子稳定性: {turnover.mean():.4f} ({'高' if turnover.mean() > 0.8 else '中' if turnover.mean() > 0.5 else '低'})</li>
            </ul>
        </div>
        
        <p style="color: #999; font-size: 12px; margin-top: 30px;">
            Generated by Alphalens | Data: {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
</body>
</html>
    """
    
    with open(f"{report_dir}/alphalens_report{file_suffix}.html", 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description='Alphalens 因子有效性分析')
    parser.add_argument('--exp-id', type=str, required=True, help='实验ID')
    parser.add_argument('--periods', type=int, nargs='+', default=[20], help='远期收益周期')
    parser.add_argument('--quantiles', type=int, default=10, help='分位数分组数')
    parser.add_argument('--use-smooth', action='store_true', help='使用半衰期平滑后的预测，输出带_smooth后缀便于对比')
    
    args = parser.parse_args()
    
    run_alphalens_analysis(
        exp_id=args.exp_id,
        periods=tuple(args.periods),
        quantiles=args.quantiles,
        use_smooth=args.use_smooth
    )


if __name__ == '__main__':
    main()
