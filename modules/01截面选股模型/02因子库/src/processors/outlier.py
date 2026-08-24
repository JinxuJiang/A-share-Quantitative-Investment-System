# -*- coding: utf-8 -*-
"""
去极值模块 - MAD缩尾法

功能:
    使用MAD (Median Absolute Deviation) 方法识别并处理极端值
    超过3倍MAD范围的值用边界值替换（缩尾而非删除）

使用示例:
    from processors.outlier import mad_winsorize
    
    # 单个截面
    clean_factor = mad_winsorize(raw_factor, n_mad=3)
    
    # 宽表批量处理
    clean_df = mad_winsorize_wide(factor_df, n_mad=3)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def mad_winsorize(
    factor: pd.Series,
    n_mad: float = 3.0,
    constant: float = 1.4826
) -> pd.Series:
    """
    MAD缩尾法去极值 (单截面)
    
    参数:
    -----
    factor : pd.Series
        原始因子值，index=股票代码
    n_mad : float, default=3.0
        MAD倍数，默认3倍
    constant : float, default=1.4826
        调整常数，使MAD与标准差在正态分布下一致
        
    返回:
    ------
    pd.Series : 去极值后的因子，index保持不变
    
    公式:
    -----
    median = factor.median()
    mad = (factor - median).abs().median()
    lower = median - n_mad * constant * mad
    upper = median + n_mad * constant * mad
    
    超过边界的值用边界值替换
    """
    if factor.empty:
        return factor
    
    # 复制数据，避免修改原始数据
    result = factor.copy()
    
    # 只使用非缺失值计算MAD
    valid_data = result.dropna()
    
    if len(valid_data) == 0:
        return result
    
    # 计算中位数
    median = valid_data.median()
    
    # 计算MAD
    mad = (valid_data - median).abs().median()
    
    # 如果MAD为0（所有值相同），无需处理
    if mad < 1e-10:
        return result
    
    # 计算边界
    lower_bound = median - n_mad * constant * mad
    upper_bound = median + n_mad * constant * mad
    
    # 缩尾处理：超过边界的用边界值替换
    result = result.clip(lower=lower_bound, upper=upper_bound)
    
    return result


def mad_winsorize_wide(
    factor_df: pd.DataFrame,
    n_mad: float = 3.0,
    constant: float = 1.4826
) -> pd.DataFrame:
    """
    MAD缩尾法去极值 (宽表格式，批量处理)
    
    参数:
    -----
    factor_df : pd.DataFrame
        因子宽表，index=日期, columns=股票代码
    n_mad : float, default=3.0
        MAD倍数
    constant : float, default=1.4826
        调整常数
        
    返回:
    ------
    pd.DataFrame : 去极值后的因子宽表
    
    说明:
    -----
    每日截面独立计算MAD并缩尾，不跨天使用数据
    """
    if factor_df.empty:
        return factor_df
    
    # 逐日处理
    result = factor_df.copy()
    
    for date in result.index:
        day_factor = result.loc[date]
        result.loc[date] = mad_winsorize(day_factor, n_mad=n_mad, constant=constant)
    
    return result


def get_outlier_bounds(
    factor: pd.Series,
    n_mad: float = 3.0,
    constant: float = 1.4826
) -> tuple:
    """
    获取MAD上下边界值（用于调试）
    
    参数:
    -----
    factor : pd.Series
        因子值
    n_mad : float, default=3.0
        MAD倍数
    constant : float, default=1.4826
        调整常数
        
    返回:
    ------
    tuple : (lower_bound, upper_bound, median, mad)
    
    使用示例:
    --------
    lower, upper, median, mad = get_outlier_bounds(factor)
    print(f"中位数:{median:.2f}, MAD:{mad:.2f}, 边界:[{lower:.2f}, {upper:.2f}]")
    """
    valid_data = factor.dropna()
    
    if len(valid_data) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    median = valid_data.median()
    mad = (valid_data - median).abs().median()
    
    if mad < 1e-10:
        return (median, median, median, 0.0)
    
    lower_bound = median - n_mad * constant * mad
    upper_bound = median + n_mad * constant * mad
    
    return (lower_bound, upper_bound, median, mad)


if __name__ == "__main__":
    # 测试
    print("测试 outlier 模块...")
    
    # 构造测试数据（含极端值）
    test_data = pd.Series(
        [10, 12, 11, 13, 10.5, 11.5, 100, -50, 10.2, np.nan],  # 100和-50是极端值
        index=[f"00000{i}.SZ" for i in range(1, 11)]
    )
    
    print(f"原始数据:\n{test_data}")
    print(f"\n原始均值: {test_data.mean():.2f}, 标准差: {test_data.std():.2f}")
    
    # 获取边界
    lower, upper, median, mad = get_outlier_bounds(test_data, n_mad=3)
    print(f"\nMAD边界: [{lower:.2f}, {upper:.2f}]")
    print(f"中位数: {median:.2f}, MAD: {mad:.2f}")
    
    # 去极值
    clean_data = mad_winsorize(test_data, n_mad=3)
    print(f"\n清洗后数据:\n{clean_data}")
    print(f"\n清洗后均值: {clean_data.mean():.2f}, 标准差: {clean_data.std():.2f}")
    
    print("\n✓ outlier 模块测试完成")
