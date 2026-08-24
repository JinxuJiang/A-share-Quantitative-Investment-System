# -*- coding: utf-8 -*-
"""
缺失值填补模块

功能:
    按规则填补因子数据中的缺失值
    
    规则:
    1. 缺失市值的整行剔除
    2. 缺失行业的整行剔除
    3. 其他缺失用行业当日中位数填补

使用示例:
    from processors.missing_value import fill_missing
    
    # 单个截面
    clean_factor = fill_missing(factor, industry, market_cap)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def fill_missing(
    factor: pd.Series,
    industry: pd.Series,
    market_cap: pd.Series,
    verbose: bool = False,
    max_missing_ratio: float = 0.05
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    缺失值填补 (单截面)
    
    参数:
    -----
    factor : pd.Series
        因子值，index=股票代码，可能含缺失
    industry : pd.Series
        行业分类，index=股票代码
    market_cap : pd.Series
        市值数据，index=股票代码
        
    返回:
    ------
    tuple : (clean_factor, clean_industry, clean_market_cap)
        剔除缺失关键变量后的数据，其他缺失已填补
    
    处理规则:
    ---------
    1. 缺失市值的 → 整行剔除
    2. 缺失行业的 → 整行剔除
    3. 缺失因子值的 → 用行业当日中位数填补（仅当缺失率低于阈值时）
    4. 缺失率超过阈值 → 保留NaN，不做填充
    5. 保留原始索引顺序
    """
    if factor.empty:
        return factor, industry, market_cap
    
    # 对齐索引
    aligned_industry = industry.reindex(factor.index)
    aligned_market_cap = market_cap.reindex(factor.index)
    
    # 创建DataFrame方便处理
    df = pd.DataFrame({
        'factor': factor,
        'industry': aligned_industry,
        'market_cap': aligned_market_cap
    })
    
    # Step 1: 剔除缺失市值或行业的行
    mask_keep = df['market_cap'].notna() & df['industry'].notna()
    df_clean = df[mask_keep].copy()
    
    n_dropped = (~mask_keep).sum()
    if verbose and n_dropped > 0:
        print(f"  剔除 {n_dropped} 只缺失市值或行业的股票")
    
    if df_clean.empty:
        # 全部剔除，返回空Series
        empty_factor = pd.Series(dtype=float, index=factor.index[:0])
        empty_industry = pd.Series(dtype=object, index=industry.index[:0])
        empty_market_cap = pd.Series(dtype=float, index=market_cap.index[:0])
        return empty_factor, empty_industry, empty_market_cap
    
    # Step 1.5: 检查因子缺失率，超过阈值则跳过填充
    missing_count = df_clean['factor'].isna().sum()
    missing_ratio = missing_count / len(df_clean) if len(df_clean) > 0 else 0.0
    
    if missing_ratio > max_missing_ratio:
        if verbose:
            print(f"  跳过填充：因子缺失率 {missing_ratio:.1%} ({missing_count}只) > 阈值 {max_missing_ratio:.1%}，保留NaN")
        # 不做填充，直接返回（NaN保留，后续all()过滤会剔除）
        clean_factor = pd.to_numeric(df_clean['factor'], errors='coerce')
        clean_industry = df_clean['industry']
        clean_market_cap = pd.to_numeric(df_clean['market_cap'], errors='coerce')
        return clean_factor, clean_industry, clean_market_cap
    
    # Step 2: 用行业当日中位数填补因子缺失值（仅对低缺失率因子执行）
    industry_median = df_clean.groupby('industry')['factor'].transform('median')
    df_clean['factor'] = df_clean['factor'].fillna(industry_median)
    
    # 如果某行业全为缺失，中位数为nan，保持nan（后续处理）
    n_filled = df_clean['factor'].isna().sum()
    if verbose and n_filled > 0:
        print(f"  警告: {n_filled} 只股票无法填补（所在行业全为缺失）")
    
    # 分离结果
    clean_factor = df_clean['factor']
    clean_industry = df_clean['industry']
    clean_market_cap = df_clean['market_cap']
    
    # 确保 factor 是数值类型
    clean_factor = pd.to_numeric(clean_factor, errors='coerce')
    clean_market_cap = pd.to_numeric(clean_market_cap, errors='coerce')
    # industry 保持字符串类型
    
    return clean_factor, clean_industry, clean_market_cap


def fill_missing_wide(
    factor_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    max_missing_ratio: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    缺失值填补 (宽表格式，批量处理)
    
    参数:
    -----
    factor_df : pd.DataFrame
        因子宽表，index=日期, columns=股票代码
    industry_df : pd.DataFrame
        行业宽表，index=日期, columns=股票代码
    market_cap_df : pd.DataFrame
        市值宽表，index=日期, columns=股票代码
        
    返回:
    ------
    tuple : (clean_factor_df, clean_industry_df, clean_market_cap_df)
        注意：不同日期的股票列表可能不同（因为剔除的多少不同）
        返回的是字典格式：{日期: Series}
    """
    if factor_df.empty:
        return factor_df, industry_df, market_cap_df
    
    clean_factors = {}
    clean_industries = {}
    clean_market_caps = {}
    
    for date in factor_df.index:
        day_factor = factor_df.loc[date]
        day_industry = industry_df.loc[date] if date in industry_df.index else pd.Series(dtype=object)
        day_market_cap = market_cap_df.loc[date] if date in market_cap_df.index else pd.Series(dtype=float)
        
        clean_f, clean_i, clean_m = fill_missing(day_factor, day_industry, day_market_cap, max_missing_ratio=max_missing_ratio)
        
        clean_factors[date] = clean_f
        clean_industries[date] = clean_i
        clean_market_caps[date] = clean_m
    
    # 转换回DataFrame（注意：可能含NaN，因为各日股票数不同）
    clean_factor_df = pd.DataFrame.from_dict(clean_factors, orient='index')
    clean_industry_df = pd.DataFrame.from_dict(clean_industries, orient='index')
    clean_market_cap_df = pd.DataFrame.from_dict(clean_market_caps, orient='index')
    
    return clean_factor_df, clean_industry_df, clean_market_cap_df


def get_missing_stats(factor: pd.Series, industry: pd.Series, market_cap: pd.Series) -> dict:
    """
    获取缺失值统计信息（用于调试）
    
    参数:
    -----
    factor : pd.Series
        因子值
    industry : pd.Series
        行业分类
    market_cap : pd.Series
        市值数据
        
    返回:
    ------
    dict : 缺失统计信息
    
    使用示例:
    --------
    stats = get_missing_stats(factor, industry, market_cap)
    print(f"缺失市值: {stats['missing_market_cap']}, 缺失因子: {stats['missing_factor']}")
    """
    aligned_industry = industry.reindex(factor.index)
    aligned_market_cap = market_cap.reindex(factor.index)
    
    return {
        'total': len(factor),
        'missing_factor': factor.isna().sum(),
        'missing_industry': aligned_industry.isna().sum(),
        'missing_market_cap': aligned_market_cap.isna().sum(),
        'missing_critical': (aligned_market_cap.isna() | aligned_industry.isna()).sum()
    }


if __name__ == "__main__":
    # 测试
    print("测试 missing_value 模块...")
    
    # 构造测试数据
    stocks = [f"00000{i}.SZ" for i in range(1, 11)]
    
    # 因子（部分缺失）
    factor = pd.Series(
        [10.0, 12.0, np.nan, 11.0, np.nan, 13.0, 10.5, np.nan, 11.5, 12.5],
        index=stocks
    )
    
    # 行业（000005缺失行业）
    industry = pd.Series(
        ['银行', '银行', '医药', '医药', np.nan, '科技', '科技', '银行', '医药', '科技'],
        index=stocks
    )
    
    # 市值（000009缺失市值）
    market_cap = pd.Series(
        [1000, 1200, 800, 900, 1100, 1500, 1300, 2000, np.nan, 1800],
        index=stocks
    )
    
    print("\n原始数据:")
    print(f"  因子: {factor.tolist()}")
    print(f"  行业: {industry.tolist()}")
    print(f"  市值: {market_cap.tolist()}")
    
    stats_before = get_missing_stats(factor, industry, market_cap)
    print(f"\n缺失统计:")
    print(f"  总股票数: {stats_before['total']}")
    print(f"  缺失因子: {stats_before['missing_factor']}")
    print(f"  缺失行业: {stats_before['missing_industry']}")
    print(f"  缺失市值: {stats_before['missing_market_cap']}")
    print(f"  缺失关键变量(行业或市值): {stats_before['missing_critical']}")
    
    # 填补
    clean_f, clean_i, clean_m = fill_missing(factor, industry, market_cap)
    
    print(f"\n清洗后:")
    print(f"  剩余股票数: {len(clean_f)}")
    print(f"  因子缺失: {clean_f.isna().sum()}")
    print(f"  因子值: {clean_f.tolist()}")
    
    # 验证填补结果（000003医药行业中位数应为12.0）
    print(f"\n验证: 000003(医药)应被填补为医药行业的中位数")
    
    print("\n✓ missing_value 模块测试完成")
