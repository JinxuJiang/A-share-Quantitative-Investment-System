# -*- coding: utf-8 -*-
"""
标准化模块 - Z-Score标准化

功能:
    使用Z-Score方法标准化因子，使其分布为N(0,1)
    适用于消除不同因子量纲的影响

使用示例:
    from processors.standardizer import zscore_standardize
    
    # 单个截面
    clean_factor = zscore_standardize(factor)
    
    # 宽表批量处理
    clean_df = zscore_standardize_wide(factor_df)
"""

import numpy as np
import pandas as pd
from typing import Union


def zscore_standardize(factor: pd.Series) -> pd.Series:
    """
    Z-Score标准化 (单截面)
    
    参数:
    -----
    factor : pd.Series
        待标准化的因子值，index=股票代码
        
    返回:
    ------
    pd.Series : 标准化后的因子，index保持不变
    
    公式:
    -----
    z = (x - μ) / σ
    
    其中:
    - μ = factor.mean()  (当日截面均值)
    - σ = factor.std()   (当日截面标准差)
    
    特殊情况:
    ---------
    如果 σ < 1e-10（标准差接近0，所有值几乎相同），
    则所有值设为0（表示该因子当天无区分度）
    """
    if factor.empty:
        return factor
    
    # 复制数据
    result = factor.copy()
    
    # 只使用非缺失值计算统计量
    valid_data = result.dropna()
    
    if len(valid_data) == 0:
        return result
    
    # 计算均值和标准差
    mean = valid_data.mean()
    std = valid_data.std()
    
    # 如果标准差接近0，所有值设为0
    if std < 1e-10:
        result.loc[result.notna()] = 0.0
        return result
    
    # Z-Score标准化
    result = (result - mean) / std
    
    return result


def zscore_standardize_wide(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-Score标准化 (宽表格式，批量处理)
    
    参数:
    -----
    factor_df : pd.DataFrame
        因子宽表，index=日期, columns=股票代码
        
    返回:
    ------
    pd.DataFrame : 标准化后的因子宽表
    
    说明:
    -----
    每日截面独立计算均值和标准差，不跨天使用数据
    """
    if factor_df.empty:
        return factor_df
    
    # 逐日处理
    result = factor_df.copy()
    
    for date in result.index:
        day_factor = result.loc[date]
        result.loc[date] = zscore_standardize(day_factor)
    
    return result


def get_standardize_stats(factor: pd.Series) -> dict:
    """
    获取标准化统计量（用于调试）
    
    参数:
    -----
    factor : pd.Series
        因子值
        
    返回:
    ------
    dict : {'mean':均值, 'std':标准差, 'count':非缺失数量}
    
    使用示例:
    --------
    stats = get_standardize_stats(factor)
    print(f"均值:{stats['mean']:.4f}, 标准差:{stats['std']:.4f}")
    """
    valid_data = factor.dropna()
    
    if len(valid_data) == 0:
        return {'mean': np.nan, 'std': np.nan, 'count': 0}
    
    return {
        'mean': valid_data.mean(),
        'std': valid_data.std(),
        'count': len(valid_data)
    }


if __name__ == "__main__":
    # 测试
    print("测试 standardizer 模块...")
    
    # 构造测试数据
    np.random.seed(42)
    test_data = pd.Series(
        np.random.normal(100, 20, 100),  # 均值100，标准差20
        index=[f"{i:06d}.SZ" for i in range(1, 101)]
    )
    # 添加一些缺失值
    test_data.iloc[90:95] = np.nan
    
    print(f"原始数据统计:")
    stats_before = get_standardize_stats(test_data)
    print(f"  均值: {stats_before['mean']:.4f}")
    print(f"  标准差: {stats_before['std']:.4f}")
    print(f"  非缺失数量: {stats_before['count']}")
    
    # 标准化
    clean_data = zscore_standardize(test_data)
    
    print(f"\n标准化后统计:")
    stats_after = get_standardize_stats(clean_data)
    print(f"  均值: {stats_after['mean']:.4f} (应为≈0)")
    print(f"  标准差: {stats_after['std']:.4f} (应为≈1)")
    print(f"  非缺失数量: {stats_after['count']} (应保持不变)")
    
    # 测试标准差为0的情况
    print("\n测试标准差为0的情况:")
    const_data = pd.Series([5.0] * 10, index=[f"00000{i}.SZ" for i in range(1, 11)])
    const_clean = zscore_standardize(const_data)
    print(f"  输入: 全是5.0")
    print(f"  输出: {const_clean.unique()} (应全为0)")
    
    print("\n✓ standardizer 模块测试完成")
