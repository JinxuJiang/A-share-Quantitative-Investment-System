"""
多因子回测脚本 - Backtrader Eval 1.1
====================================

使用示例:
---------
# 基本用法（使用 predictions.parquet）
python backtrader.eval_1.1.py --exp-id exp_001

# 指定实验ID
python backtrader.eval_1.1.py -e test_001_fined_v1

# 使用平滑后的预测（smoothed_predictions.parquet）
python backtrader.eval_1.1.py --exp-id ensemble_5d_20d_60d_v1 --use-smooth

命令行参数:
-----------
--exp-id, -e    实验ID，对应 03模型训练层/experiments/{exp_id} (默认: exp_001)
--use-smooth    使用平滑后的预测文件 smoothed_predictions.parquet

回测参数配置:
-------------
请在下方 STRATEGY_PARAMS 字典中修改回测参数
"""

import pandas as pd
import numpy as np
import gc
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from pathlib import Path
import base64
from io import BytesIO
from functools import lru_cache


# ==========================================
# 0. 回测参数配置（在此处修改参数）
# ==========================================
STRATEGY_PARAMS = {
    'stocks_per_batch': 20,           # 每次选股数量
    'start_date': datetime(2023, 10, 1),  # 回测开始日期
    'end_date': None,                 # 自动取预测与行情共同覆盖的最新日期
    'initial_cash': 50000,           # 初始资金
    'commission': 0.002               # 手续费率 (0.2%)
}

# 全局变量：记录每日净值
cash_value_history = {}

# ==========================================
# 1. 配置参数
# ==========================================
import os
# 获取项目根目录（当前文件的上上级目录）
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_PATHS = {
    'open': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'open.parquet',
    'close': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'close.parquet',
    'high': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'high.parquet',
    'low': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'low.parquet',
    'volume': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'volume.parquet',
}

# ST状态数据路径
ST_STATUS_PATH = PROJECT_ROOT / '01数据' / 'data' / 'tushare_data' / 'st_status.parquet'
TRADE_SCHEDULE_PATH = (
    PROJECT_ROOT / '01数据' / 'data' / 'tushare_data' / 'raw' / 'metadata'
    / 'trade_schedule.parquet'
)
BENCHMARK_PATH = (
    PROJECT_ROOT / '01数据' / 'data' / 'tushare_data' / 'benchmark'
    / '000852.SH.parquet'
)
BENCHMARK_NAME = '中证1000'
BENCHMARK_PLOT_NAME = 'CSI 1000'


@lru_cache(maxsize=1)
def load_st_status():
    """加载ST状态数据（带缓存）"""
    if not ST_STATUS_PATH.exists():
        print(f"  警告: 找不到ST状态文件: {ST_STATUS_PATH}")
        return None
    return pd.read_parquet(ST_STATUS_PATH)


def load_month_end_dates(start_date, end_date, available_dates):
    """从正式交易安排中取得区间内每月最后一个交易日。"""
    if not TRADE_SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            f"缺少完整交易安排: {TRADE_SCHEDULE_PATH}\n"
                "请先运行: python 01数据/tushare_data_main.py --weekly"
        )
    calendar = pd.read_parquet(TRADE_SCHEDULE_PATH)
    required = {'cal_date', 'is_open'}
    if not required.issubset(calendar.columns):
        raise ValueError(f"交易安排缺少字段: {sorted(required - set(calendar.columns))}")

    calendar = calendar.copy()
    calendar['date'] = pd.to_datetime(
        calendar['cal_date'].astype(str), format='%Y%m%d', errors='coerce'
    )
    open_days = calendar.loc[calendar['is_open'].astype(int).eq(1), 'date'].dropna()
    month_ends = open_days.groupby(open_days.dt.to_period('M')).max().sort_values()

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    available = pd.DatetimeIndex(available_dates).normalize().unique()
    scheduled = pd.DatetimeIndex(month_ends[(month_ends >= start) & (month_ends <= end)])
    missing = scheduled.difference(available)
    if len(missing):
        print(
            "  警告: 以下月末交易日没有预测/行情，跳过: "
            + ", ".join(d.strftime('%Y-%m-%d') for d in missing)
        )
    return sorted(scheduled.intersection(available).tolist())


def load_week_end_dates(start_date, end_date, available_dates):
    """从正式交易安排中取得每周最后一个实际交易日。"""
    if not TRADE_SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            f"缺少完整交易安排: {TRADE_SCHEDULE_PATH}\n"
                "请先运行: python 01数据/tushare_data_main.py --weekly"
        )
    calendar = pd.read_parquet(TRADE_SCHEDULE_PATH)
    required = {'cal_date', 'is_open'}
    if not required.issubset(calendar.columns):
        raise ValueError(f"交易安排缺少字段: {sorted(required - set(calendar.columns))}")

    calendar = calendar.copy()
    calendar['date'] = pd.to_datetime(
        calendar['cal_date'].astype(str), format='%Y%m%d', errors='coerce'
    )
    open_days = calendar.loc[calendar['is_open'].astype(int).eq(1), 'date'].dropna()
    week_ends = open_days.groupby(open_days.dt.to_period('W-FRI')).max().sort_values()

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    available = pd.DatetimeIndex(available_dates).normalize().unique()
    scheduled = pd.DatetimeIndex(week_ends[(week_ends >= start) & (week_ends <= end)])
    missing = scheduled.difference(available)
    if len(missing):
        print(
            "  警告: 以下周末交易日没有预测/行情，跳过: "
            + ", ".join(d.strftime('%Y-%m-%d') for d in missing)
        )
    return sorted(scheduled.intersection(available).tolist())


def load_benchmark(start_date, end_date):
    """加载中证1000收盘价并计算用于展示的净值与指标。"""
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"缺少中证1000基准行情: {BENCHMARK_PATH}\n"
                "请先运行: python 01数据/tushare_data_main.py --weekly"
        )
    benchmark = pd.read_parquet(BENCHMARK_PATH)
    if not {'trade_date', 'close'}.issubset(benchmark.columns):
        raise ValueError("中证1000基准行情缺少 trade_date/close 字段")
    benchmark = benchmark[['trade_date', 'close']].copy()
    benchmark.index = pd.to_datetime(
        benchmark.pop('trade_date').astype(str), format='%Y%m%d', errors='coerce'
    )
    benchmark['close'] = pd.to_numeric(benchmark['close'], errors='coerce')
    close = benchmark['close'].dropna().sort_index()
    close = close[~close.index.duplicated(keep='last')]
    close = close.loc[(close.index >= pd.Timestamp(start_date)) &
                      (close.index <= pd.Timestamp(end_date))]
    if close.empty:
        raise ValueError("回测区间内没有中证1000基准行情")

    nav = close / close.iloc[0]
    returns = close.pct_change().dropna()
    years = max((close.index[-1] - close.index[0]).days / 365.25, 1 / 252)
    total_return = nav.iloc[-1] - 1
    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_vol = returns.std(ddof=1) * np.sqrt(252) if len(returns) > 1 else np.nan
    sharpe = ((returns.mean() * 252 - 0.02) / ann_vol
              if pd.notna(ann_vol) and ann_vol > 0 else np.nan)
    drawdown = nav / nav.cummax() - 1
    metrics = {
        'total_return': total_return * 100,
        'ann_return': ann_return * 100,
        'ann_vol': ann_vol * 100 if pd.notna(ann_vol) else np.nan,
        'sharpe': sharpe,
        'max_dd': abs(drawdown.min()) * 100,
    }
    return nav, metrics

def parse_args():
    parser = argparse.ArgumentParser(description='多因子回测脚本')
    parser.add_argument('--exp-id', '-e', type=str, default='exp_001',
                        help='实验ID，如 lgbm20_profit20_full_v2')
    parser.add_argument('--use-smooth', action='store_true',
                        help='使用平滑后的预测 (smoothed_predictions.parquet)')
    return parser.parse_args()

def get_paths(exp_id, use_smooth=False):
    exp_dir = PROJECT_ROOT / '03模型训练层' / 'experiments' / exp_id
    paths = DEFAULT_PATHS.copy()
    
    # 根据参数选择文件
    if use_smooth:
        paths['pred'] = str(exp_dir / 'smoothed_predictions.parquet')
        paths['pred_col'] = 'pred_score_smooth'
    else:
        paths['pred'] = str(exp_dir / 'predictions.parquet')
        paths['pred_col'] = 'pred_score'
    
    return paths, exp_dir

# ==========================================
# 2. 数据处理模块
# ==========================================
def wide_to_long(df_wide, value_name, time_col='time'):
    if time_col not in df_wide.columns:
        if df_wide.index.name == time_col:
            df_wide = df_wide.reset_index()
        else:
            df_wide.index.name = time_col
            df_wide = df_wide.reset_index()

    df_wide = df_wide.set_index(time_col)
    df_long = df_wide.stack().reset_index()
    df_long.columns = [time_col, 'stock_code', value_name]
    
    df_long = df_long[df_long['stock_code'].str.match(r'^\d{6}\.(SZ|SH|BJ)$', na=False)]
    df_long[value_name] = df_long[value_name].astype('float32')
    df_long = df_long.dropna(subset=[value_name])
    return df_long

def load_and_merge_data(paths):
    print("--- 步骤1: 加载预测数据 ---")
    pred_col = paths['pred_col']
    
    # 检查文件是否存在
    if not os.path.exists(paths['pred']):
        raise FileNotFoundError(f"找不到预测文件: {paths['pred']}")
    
    pred_total = pd.read_parquet(
        paths['pred'],
        columns=['date', 'stock_code', pred_col],
    )
    pred_total = pred_total.rename(columns={pred_col: 'pred_score'})
    pred_total = pred_total.rename(columns={'date': 'time', 'pred_score': 'prediction'})
    pred_total['time'] = pd.to_datetime(pred_total['time'])
    pred_total['prediction'] = pred_total['prediction'].astype('float32')
    prediction_end = pred_total['time'].max()

    main_df = pred_total
    for col in ['open', 'close', 'high', 'low', 'volume']:
        print(f"正在处理 {col} 数据...")
        temp_wide = pd.read_parquet(paths[col])
        temp_long = wide_to_long(temp_wide, col)
        
        main_df = pd.merge(main_df, temp_long, on=['time', 'stock_code'], how='left')
        del temp_wide, temp_long
        gc.collect()

    main_df['openinterest'] = 0
    main_df['datetime'] = pd.to_datetime(main_df['time'])
    main_df = main_df.set_index('datetime')
    main_df['stock_code'] = main_df['stock_code'].astype('category')
    
    main_df = main_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    common_end = main_df.index.max()
    if pd.isna(common_end):
        raise ValueError('预测与行情没有可共同使用的日期')
    main_df.attrs['prediction_end'] = prediction_end
    main_df.attrs['common_end'] = common_end
    
    print(f"完成数据合并，形状: {main_df.shape}")
    print(f"数据时间范围: {main_df.index.min()} ~ {main_df.index.max()}")
    if common_end < prediction_end:
        print(
            f"警告: 预测最新日期为 {prediction_end:%Y-%m-%d}，但行情仅共同覆盖到 "
            f"{common_end:%Y-%m-%d}；回测将使用共同截止日"
        )
    return main_df

# ==========================================
# 3. 信号生成模块
# ==========================================
def generate_signals(df, top_n, start_date, end_date,
                     rebalance_frequency='monthly_last'):
    """
    生成调仓信号（信号日收盘生成，下一交易日开盘成交）
    
    返回:
        buy_dict: 买入信号字典 {date: [stock_list]}
        sell_dict: 卖出信号字典 {date: [stock_list]}
        all_held_stocks: 所有持有过的股票列表
        rebalance_records: 详细的调仓记录列表（用于导出CSV）
    """
    mask = (df.index >= pd.to_datetime(start_date) - pd.Timedelta(days=45)) & \
           (df.index <= pd.to_datetime(end_date) + pd.Timedelta(days=5))
    df_period = df.loc[mask].copy()
    
    buy_dict = {}
    sell_dict = {}
    current_position = set()
    all_held_stocks = set()
    rebalance_records = []  # 新增：记录详细信号
    
    trading_days = sorted(df_period.index.unique())
    if not trading_days:
        return {}, {}, []
    
    # 使用官方交易安排识别周期末，不能把尚未结束周期的最新数据日误判为调仓日。
    if rebalance_frequency == 'weekly_last':
        rebalance_dates = load_week_end_dates(
            start_date, end_date, available_dates=trading_days
        )
        frequency_text = '每周最后一个交易日'
    elif rebalance_frequency == 'monthly_last':
        rebalance_dates = load_month_end_dates(
            start_date, end_date, available_dates=trading_days
        )
        frequency_text = '每月最后一个交易日'
    else:
        raise ValueError(f"不支持的调仓频率: {rebalance_frequency}")
    
    print(f"\n--- 步骤2: 生成调仓信号（{frequency_text}，已加入ST过滤） ---")
    print(f"回测范围内总交易日: {len(trading_days)}, 计划调仓次数: {len(rebalance_dates)}")
    if rebalance_dates:
        print(f"首个调仓日: {rebalance_dates[0].strftime('%Y-%m-%d')}")
        print(f"末个调仓日: {rebalance_dates[-1].strftime('%Y-%m-%d')}")
    
    for date in rebalance_dates:
        if date > pd.to_datetime(end_date):
            break
            
        date_str = date.strftime('%Y-%m-%d')
        try:
            current_slice = df_period.loc[date]
        except KeyError:
            continue
            
        if isinstance(current_slice, pd.Series):
            current_slice = current_slice.to_frame().T
            
        if current_slice.empty:
            continue
        
        # 1. 只保留主板股票（60/00开头）
        current_slice = current_slice[
            current_slice['stock_code'].str.match(r'^(60|00)\d{4}\.(SH|SZ)$', na=False)
        ]
        
        if current_slice.empty:
            continue
        
        # 2. 过滤ST/*ST股票
        st_status = load_st_status()
        if st_status is not None:
            date_key = pd.Timestamp(date.date())
            if date_key in st_status.index:
                normal_stocks = st_status.loc[date_key][st_status.loc[date_key] == 0].index.tolist()
                before_count = len(current_slice)
                current_slice = current_slice[current_slice['stock_code'].isin(normal_stocks)]
                st_filtered = before_count - len(current_slice)
                if st_filtered > 0:
                    print(f"  {date_str}: 过滤 {st_filtered} 只ST/*ST股票")
            else:
                print(f"  {date_str}: 警告-ST数据中找不到该日期，跳过ST过滤")
        
        if current_slice.empty:
            continue
        
        # 3. 只用截至信号日的历史，过滤有效数据不足20天的股票。
        hist_data = df_period[df_period.index <= date]
        stock_data_counts = hist_data.groupby('stock_code', observed=True).size()
        valid_stocks_20d = stock_data_counts[stock_data_counts >= 20].index.tolist()
        before_count = len(current_slice)
        current_slice = current_slice[current_slice['stock_code'].isin(valid_stocks_20d)]
        data_filtered = before_count - len(current_slice)
        if data_filtered > 0:
            print(f"  {date_str}: 过滤 {data_filtered} 只数据不足20天股票")
        
        if current_slice.empty:
            continue
        
        # 4. 严格按T日可知信息选Top N。T+1开盘价在信号生成时未知，
        # 不再用它预先过滤或用后续股票替补。
        selected = current_slice.sort_values(by='prediction', ascending=False).head(top_n)
        buy_list = selected['stock_code'].tolist()
        buy_list_set = set(buy_list)
        
        sell_list = sorted(list(current_position - buy_list_set))
        
        # 新增：记录详细的买入信号
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            rebalance_records.append({
                'date': date_str,
                'action': 'BUY',
                'stock_code': row['stock_code'],
                'pred_score': row['prediction'],
                'rank': rank,
                'close_t': row['close'],
                'notes': ''
            })
        
        # 新增：记录详细的卖出信号
        for stock in sell_list:
            rebalance_records.append({
                'date': date_str,
                'action': 'SELL',
                'stock_code': stock,
                'pred_score': None,
                'rank': None,
                'close_t': None,
                'notes': '调出持仓'
            })
        
        buy_dict[date_str] = buy_list
        sell_dict[date_str] = sell_list
        current_position = buy_list_set
        all_held_stocks.update(current_position)
        
    return buy_dict, sell_dict, sorted(list(all_held_stocks)), rebalance_records

# ==========================================
# 4. Backtrader 策略类
# ==========================================
class MyMultiFactorStrategy(bt.Strategy):
    params = (
        ('buy_date', None), 
        ('sell_date', None), 
        ('trades', None),
        ('stop_loss', 0.2),      # 止损比例：0.2（20%）
    )

    def __init__(self):
        self.trade_count = 0
        self.prenext_count = 0

    def prenext(self):
        self.prenext_count += 1
        self.next()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def next(self):
        curr_dt = self.datetime.date(0).strftime('%Y-%m-%d')
        
        # 记录每日净值（使用全局变量）
        global cash_value_history
        cash_value_history[curr_dt] = self.broker.getvalue()
        
        # === 步骤1：每日检查止损（优先执行）===
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:                      # 有持仓
                cost_price = pos.price             # 平均成本价
                current_price = data.close[0]      # 当前收盘价
                
                # 跌超8%就止损卖出
                if current_price < cost_price * (1 - self.p.stop_loss):
                    self.order_target_percent(data=data, target=0)
                    print(f"止损(20%): {data._name} 成本{cost_price:.2f} 现价{current_price:.2f} (跌{((current_price/cost_price-1)*100):.1f}%)")
        
        # === 步骤2：调仓日逻辑 ===
        is_buy_day = curr_dt in self.p.buy_date
        is_sell_day = curr_dt in self.p.sell_date
        
        if not is_buy_day and not is_sell_day:
            return
        
        self.trade_count += 1
        action = "初始建仓" if self.trade_count == 1 else f"调仓 #{self.trade_count}"
        print(f"\n--- {action}: {curr_dt} ---")
        
        # 1. 先执行卖出
        if is_sell_day and self.p.sell_date[curr_dt]:
            s_list = self.p.sell_date[curr_dt]
            for s_code in s_list:
                try:
                    if s_code in self.getdatanames():
                        data = self.getdatabyname(s_code)
                        pos = self.getposition(data)
                        if pos.size > 0:
                            self.order_target_percent(data=data, target=0)
                except Exception as e:
                    continue

        # 2. 再执行买入
        if is_buy_day and self.p.buy_date[curr_dt]:
            b_list = self.p.buy_date[curr_dt]
            if len(b_list) > 0:
                valid_stocks = [s for s in b_list if s in self.getdatanames()]
                
                if valid_stocks:
                    target_per = 0.90 / len(valid_stocks)
                    
                    for b_code in valid_stocks:
                        try:
                            data = self.getdatabyname(b_code)
                            self.order_target_percent(data=data, target=target_per)
                        except Exception as e:
                            continue

    def notify_order(self, order):
        if order.status in [order.Completed]:
            dt = self.datetime.date(0).strftime('%Y-%m-%d')
            is_buy = order.isbuy()
            action = 'BUY' if is_buy else 'SELL'
            price = order.executed.price
            size = order.executed.size
            
            self.p.trades.append({
                'date': dt,
                'stock_code': order.data._name,
                'action': action,
                'price': round(price, 2),
                'shares': abs(size)
            })

# ==========================================
# 5. 回测引擎模块
# ==========================================
def run_backtest(full_df, buy_date, sell_date, stock_list, strategy_params, trades_list):
    global cash_value_history
    cash_value_history = {}  # 清空历史
    
    cerebro = bt.Cerebro(runonce=False)
    
    print("\n--- 步骤3: 加载 Backtrader 数据源 ---")
    total_stocks = len(stock_list)
    valid_count = 0
    
    for i, stock_code in enumerate(stock_list):
        data_slice = full_df[full_df['stock_code'] == stock_code][['open', 'high', 'low', 'close', 'volume', 'openinterest']]
        
        if data_slice.empty:
            continue
        
        data_slice = data_slice[(data_slice.index >= strategy_params['start_date']) & 
                                (data_slice.index <= strategy_params['end_date'])]
        
        if len(data_slice) < 20:
            continue
            
        data = bt.feeds.PandasData(
            dataname=data_slice,
            fromdate=strategy_params['start_date'],
            todate=strategy_params['end_date']
        )
        cerebro.adddata(data, name=stock_code)
        valid_count += 1
        
        if (i+1) % 100 == 0 or (i+1) == total_stocks:
            print(f"加载进度: {i+1}/{total_stocks} (成功: {valid_count})")

    print(f"\n成功加载 {valid_count} 只股票")
    
    if valid_count == 0:
        print("错误：没有成功加载任何股票数据！")
        return None, None, None

    cerebro.addstrategy(MyMultiFactorStrategy, buy_date=buy_date, sell_date=sell_date, trades=trades_list)
    
    cerebro.broker.setcash(strategy_params['initial_cash'])
    cerebro.broker.setcommission(commission=strategy_params['commission'])
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='Sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='Drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='Returns', tann=252)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer')

    print("\n--- 步骤4: 启动回测引擎 ---")
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    
    results = cerebro.run(runonce=False)
    
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    print(f"\n" + "="*50)
    print(f"最终净值: {final_value:.2f}")
    
    total_return = (final_value / strategy_params['initial_cash'] - 1) * 100
    print(f"总收益率: {total_return:.2f}%")
    
    metrics = {}
    try:
        # 通过 _name 获取分析器
        ret_analysis = strat.analyzers.getbyname('Returns').get_analysis()
        sharpe_analysis = strat.analyzers.getbyname('Sharpe').get_analysis()
        dd_analysis = strat.analyzers.getbyname('Drawdown').get_analysis()
        trade_analysis = strat.analyzers.getbyname('TradeAnalyzer').get_analysis()

        metrics['ann_return'] = ret_analysis.get('rnorm100', 0) if ret_analysis else 0
        metrics['sharpe'] = sharpe_analysis.get('sharperatio', 0) if sharpe_analysis else 0
        
        # 安全获取最大回撤
        max_dd = 0
        if dd_analysis:
            max_dd = dd_analysis.get('max', {}).get('drawdown', 0) if isinstance(dd_analysis, dict) else 0
        metrics['max_dd'] = max_dd
        
        # 安全获取交易统计
        total_trades = 0
        win_rate = 0
        avg_win = 0      # 平均盈利
        avg_loss = 0     # 平均亏损
        profit_factor = 0  # 盈亏比
        
        if trade_analysis and isinstance(trade_analysis, dict):
            total = trade_analysis.get('total', {}).get('total', 0)
            won = trade_analysis.get('won', {}).get('total', 0)
            total_trades = total
            win_rate = (won / total * 100) if total > 0 else 0
            
            # 获取平均盈利和平均亏损
            try:
                won_analysis = trade_analysis.get('won', {})
                lost_analysis = trade_analysis.get('lost', {})
                
                if won_analysis and 'pnl' in won_analysis:
                    avg_win = won_analysis['pnl'].get('average', 0)
                if lost_analysis and 'pnl' in lost_analysis:
                    avg_loss = abs(lost_analysis['pnl'].get('average', 0))  # 转为正数
                    
                # 计算盈亏比
                if avg_loss > 0:
                    profit_factor = avg_win / avg_loss
                elif avg_win > 0:
                    profit_factor = float('inf')  # 只盈利无亏损
            except:
                pass
        
        metrics['total_trades'] = total_trades
        metrics['win_rate'] = win_rate
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        metrics['profit_factor'] = profit_factor

        print(f"年化收益率: {metrics['ann_return']:.2f}%")
        print(f"夏普比率: {metrics['sharpe']:.2f}" if metrics['sharpe'] else "夏普比率: N/A")
        print(f"最大回撤: {metrics['max_dd']:.2f}%")
        print(f"胜率: {win_rate:.1f}% | 盈亏比: {profit_factor:.2f} (平均盈利{avg_win:.0f}/平均亏损{avg_loss:.0f})")
    except Exception as e:
        print(f"获取分析指标时出错: {e}")
        
    print("="*50)
    
    # 转换净值历史为 Series
    equity_df = pd.Series(cash_value_history)
    equity_df.index = pd.to_datetime(equity_df.index)
    equity_df = equity_df.sort_index()
    
    return cerebro, metrics, equity_df

# ==========================================
# 6. HTML报告生成
# ==========================================
def plot_equity_comparison(equity_df, benchmark_nav, exp_id):
    """绘制统一从1开始的策略与中证1000净值曲线。"""
    strategy_nav = equity_df / equity_df.iloc[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    strategy_nav.plot(ax=ax, label='Strategy', linewidth=1.8)
    benchmark_nav.plot(ax=ax, label=BENCHMARK_PLOT_NAME, linewidth=1.5, alpha=0.85)
    ax.set_title(f'Strategy vs {BENCHMARK_PLOT_NAME} - {exp_id}')
    ax.set_ylabel('Normalized NAV')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    return fig


def generate_html_report(exp_id, metrics, equity_df, benchmark_nav,
                         benchmark_metrics, output_dir,
                         rebalance_text='每月最后一个交易日收盘生成信号，下一交易日开盘成交'):
    """生成HTML报告"""
    
    # 将图片转为base64
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64
    
    # 绘制策略与基准净值曲线
    fig = plot_equity_comparison(equity_df, benchmark_nav, exp_id)
    
    img_base64 = fig_to_base64(fig)
    plt.close()
    
    total_return = (equity_df.iloc[-1] / STRATEGY_PARAMS['initial_cash'] - 1) * 100
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Backtest Report - {exp_id}</title>
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
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .summary {{ background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 回测报告 - {exp_id}</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 关键指标</h2>
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value {'good' if total_return > 0 else 'bad'}">{total_return:.2f}%</div>
                <div class="metric-label">总收益率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {'good' if metrics.get('ann_return', 0) > 0 else 'bad'}">{metrics.get('ann_return', 0):.2f}%</div>
                <div class="metric-label">年化收益率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {'good' if metrics.get('sharpe', 0) > 1 else 'warning' if metrics.get('sharpe', 0) > 0.5 else 'bad'}">{metrics.get('sharpe', 0):.2f}</div>
                <div class="metric-label">夏普比率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {'bad' if metrics.get('max_dd', 0) > 20 else 'warning' if metrics.get('max_dd', 0) > 10 else 'good'}">{metrics.get('max_dd', 0):.2f}%</div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{int(metrics.get('total_trades', 0))}</div>
                <div class="metric-label">总交易次数</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.get('win_rate', 0):.1f}%</div>
                <div class="metric-label">胜率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
                <div class="metric-label">盈亏比</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.get('avg_win', 0):.0f}</div>
                <div class="metric-label">平均盈利</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.get('avg_loss', 0):.0f}</div>
                <div class="metric-label">平均亏损</div>
            </div>
        </div>

        <h2>📉 {BENCHMARK_NAME}基准</h2>
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value">{benchmark_metrics.get('total_return', np.nan):.2f}%</div>
                <div class="metric-label">累计收益率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{benchmark_metrics.get('ann_return', np.nan):.2f}%</div>
                <div class="metric-label">年化收益率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{benchmark_metrics.get('ann_vol', np.nan):.2f}%</div>
                <div class="metric-label">年化波动率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{benchmark_metrics.get('sharpe', np.nan):.2f}</div>
                <div class="metric-label">夏普比率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{benchmark_metrics.get('max_dd', np.nan):.2f}%</div>
                <div class="metric-label">最大回撤</div>
            </div>
        </div>
        
        <div class="summary">
            <h3>📋 回测设置</h3>
            <ul>
                <li><strong>模型:</strong> {exp_id}</li>
                <li><strong>回测区间:</strong> {equity_df.index[0].strftime('%Y-%m-%d')} ~ {equity_df.index[-1].strftime('%Y-%m-%d')}</li>
                <li><strong>调仓周期:</strong> {rebalance_text}</li>
                <li><strong>主基准:</strong> {BENCHMARK_NAME}（000852.SH）</li>
                <li><strong>持仓数量:</strong> {STRATEGY_PARAMS['stocks_per_batch']}只</li>
                <li><strong>初始资金:</strong> {STRATEGY_PARAMS['initial_cash']:,}</li>
            </ul>
        </div>
        
        <h2>📈 净值曲线</h2>
        <img src="data:image/png;base64,{img_base64}" alt="Equity Curve">
        
        <p style="color: #999; font-size: 12px; margin-top: 30px;">
            Generated by Backtrader | Data: {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
</body>
</html>'''
    
    html_path = output_dir / 'backtest_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已保存: {html_path}")

# ==========================================
# 7. 执行入口
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    exp_id = args.exp_id
    rebalance_frequency = os.environ.get(
        'BACKTEST_REBALANCE_FREQUENCY', 'monthly_last'
    )
    report_variant = os.environ.get('BACKTEST_REPORT_VARIANT', '').strip()
    rebalance_text = (
        '每周最后一个交易日收盘生成信号，下一交易日开盘成交'
        if rebalance_frequency == 'weekly_last'
        else '每月最后一个交易日收盘生成信号，下一交易日开盘成交'
    )
    
    print("="*60)
    print(f"多因子回测 - {exp_id}")
    print("="*60)
    
    # 获取路径
    PATHS, exp_dir = get_paths(exp_id, args.use_smooth)
    

    
    # 创建输出目录
    reports_root = PROJECT_ROOT / '04回测层' / 'reports'
    output_dir = (
        reports_root / report_variant / exp_id
        if report_variant else reports_root / exp_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    master_df = load_and_merge_data(PATHS)
    # 每次运行自动跟随预测与行情的共同最新日期，不再维护硬编码截止日。
    STRATEGY_PARAMS['end_date'] = pd.Timestamp(
        master_df.attrs['common_end']
    ).to_pydatetime()
    
    # 诊断信息
    print(f"\n[诊断] 主表日期范围: {master_df.index.min()} ~ {master_df.index.max()}")
    print(f"[诊断] 自动回测截止日: {STRATEGY_PARAMS['end_date']:%Y-%m-%d}")

    # 生成信号（每月最后一个交易日收盘产生，下一交易日开盘成交）
    buy_date, sell_date, stock_list, rebalance_records = generate_signals(
        master_df,
        STRATEGY_PARAMS['stocks_per_batch'],
        STRATEGY_PARAMS['start_date'],
        STRATEGY_PARAMS['end_date'],
        rebalance_frequency=rebalance_frequency,
    )
    
    # 新增：保存调仓信号到CSV
    if rebalance_records:
        signals_df = pd.DataFrame(rebalance_records)
        signals_path = output_dir / 'rebalance_signals.csv'
        signals_df.to_csv(signals_path, index=False, encoding='utf-8-sig')
        print(f"\n调仓信号已导出: {signals_path}")
        print(f"  总记录数: {len(signals_df)} 条")
        print(f"  买入记录: {len(signals_df[signals_df['action']=='BUY'])} 条")
        print(f"  卖出记录: {len(signals_df[signals_df['action']=='SELL'])} 条")
    
    print(f"\n涉及股票总数: {len(stock_list)}")
    
    if not buy_date:
        print("错误：没有生成任何调仓信号！")
    else:
        # 交易记录列表
        trades_list = []
        
        # 运行回测
        cerebro, metrics, equity_df = run_backtest(master_df, buy_date, sell_date, stock_list, 
                                                    STRATEGY_PARAMS, trades_list)
        
        if cerebro and metrics and equity_df is not None:
            benchmark_nav, benchmark_metrics = load_benchmark(
                equity_df.index.min(), equity_df.index.max()
            )
            print(
                f"{BENCHMARK_NAME}: 累计收益 {benchmark_metrics['total_return']:.2f}% | "
                f"Sharpe {benchmark_metrics['sharpe']:.2f} | "
                f"最大回撤 {benchmark_metrics['max_dd']:.2f}%"
            )
            # 1. 保存交易记录
            if trades_list:
                trades_df = pd.DataFrame(trades_list)
                trades_path = output_dir / 'trades.csv'
                trades_df.to_csv(trades_path, index=False)
                print(f"交易记录已保存: {trades_path} ({len(trades_df)} 笔)")
            
            # 2. 保存净值曲线图
            fig = plot_equity_comparison(equity_df, benchmark_nav, exp_id)
            
            png_path = output_dir / 'equity_curve.png'
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"净值曲线已保存: {png_path}")
            
            # 打印最终收益（和HTML一致）
            final_return = (equity_df.iloc[-1] / STRATEGY_PARAMS['initial_cash'] - 1) * 100
            print(f"\n最终收益: {equity_df.iloc[-1]:.2f} (收益率: {final_return:.2f}%)")
            
            # 3. 生成HTML报告
            generate_html_report(
                exp_id, metrics, equity_df, benchmark_nav,
                benchmark_metrics, output_dir,
                rebalance_text=rebalance_text,
            )
            
        print(f"\n所有输出文件保存在: {output_dir}")
