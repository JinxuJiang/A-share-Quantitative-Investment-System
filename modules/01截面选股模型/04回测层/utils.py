"""
回测层公共工具函数
数据加载、路径处理、格式转换
"""

import os
import pandas as pd
from pathlib import Path


# 项目根目录（当前文件的父目录）
PROJECT_ROOT = Path(__file__).parent.parent


def get_predictions_path(exp_id: str, use_smooth: bool = False) -> str:
    """获取predictions.parquet路径，use_smooth=True时读取平滑后的预测"""
    filename = "smoothed_predictions.parquet" if use_smooth else "predictions.parquet"
    return str(PROJECT_ROOT / "03模型训练层" / "experiments" / exp_id / filename)


def get_close_price_path() -> str:
    """获取收盘价数据路径"""
    return str(PROJECT_ROOT / "02因子库" / "processed_data" / "market_data" / "close.parquet")


def load_predictions(exp_id: str, use_smooth: bool = False) -> pd.DataFrame:
    """
    加载模型预测结果
    
    Args:
        exp_id: 实验ID，如 'exp_20260303_155448'
        use_smooth: 是否使用平滑后的预测
    
    Returns:
        DataFrame: 包含 date, stock_code, pred_score, actual_return, fold_id (长格式)
    """
    path = get_predictions_path(exp_id, use_smooth)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到预测文件: {path}")
    
    df = pd.read_parquet(path)
    
    # 检查是否是MultiIndex格式（date, stock_code作为索引）
    if isinstance(df.index, pd.MultiIndex):
        # 将索引重置为列
        df = df.reset_index()
    
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    filename = "smoothed_predictions.parquet" if use_smooth else "predictions.parquet"
    print(f"[数据加载] {filename}: {len(df)} 行, {df['date'].nunique()} 个交易日")
    return df


def load_close_prices() -> pd.DataFrame:
    """
    加载收盘价数据（宽表格式）
    
    Returns:
        DataFrame: index=date, columns=stock_code
    """
    path = get_close_price_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到收盘价文件: {path}")
    
    df = pd.read_parquet(path)
    
    # 设置日期索引（parquet中time是普通列）
    if 'time' in df.columns:
        df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
    
    print(f"[数据加载] close.parquet: {df.shape[0]} 个交易日, {df.shape[1]} 只股票")
    return df


def prepare_alphalens_data(predictions: pd.DataFrame, prices: pd.DataFrame, pred_col: str = 'pred_score'):
    """
    准备Alphalens需要的数据格式
    
    Args:
        predictions: 预测结果DataFrame
        prices: 收盘价宽表
        pred_col: 预测列名，默认'pred_score'，平滑后为'pred_score_smooth'
    
    Returns:
        factor: Series, MultiIndex (date, asset)
        prices_aligned: DataFrame, 与factor对齐后的价格
    """
    # 1. factor: MultiIndex (date, asset) -> pred_score
    factor = predictions.set_index(['date', 'stock_code'])[pred_col]
    
    # 2. 对齐价格数据：取factor日期范围内的价格
    start_date = factor.index.get_level_values('date').min()
    end_date = factor.index.get_level_values('date').max()
    
    prices_aligned = prices.loc[start_date:end_date].copy()
    
    # 3. 确保价格的columns包含所有股票
    stocks_in_factor = factor.index.get_level_values('stock_code').unique()
    stocks_in_prices = set(prices_aligned.columns)
    
    missing_stocks = set(stocks_in_factor) - stocks_in_prices
    if missing_stocks:
        print(f"[警告] 有 {len(missing_stocks)} 只股票在价格数据中不存在")
    
    print(f"[数据对齐] factor: {len(factor)} 条记录, prices: {prices_aligned.shape}")
    return factor, prices_aligned


def ensure_report_dir(exp_id: str) -> str:
    """确保报告目录存在，返回目录路径"""
    report_dir = PROJECT_ROOT / "04回测层" / "reports" / exp_id
    report_dir.mkdir(parents=True, exist_ok=True)
    return str(report_dir)
