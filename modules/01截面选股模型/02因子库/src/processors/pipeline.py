# -*- coding: utf-8 -*-
"""
Pipeline 流程串联模块

功能:
    将因子清洗的4个步骤串联起来，提供一键清洗功能
    
    流程: 去极值 → 填补缺失 → 中性化 → 标准化

使用示例:
    from processors.pipeline import clean_factor
    
    # 一键清洗
    factor_clean = clean_factor(
        factor_raw,      # 原始因子
        industry,        # 行业分类
        market_cap       # 市值数据
    )
"""

import pandas as pd
import numpy as np
from typing import Optional, List

# 导入各模块函数
from .outlier import mad_winsorize
from .missing_value import fill_missing, get_missing_stats
from .neutralizer import neutralize
from .standardizer import zscore_standardize


def clean_factor(
    factor: pd.Series,
    industry: pd.Series,
    market_cap: pd.Series,
    steps: Optional[List[str]] = None,
    verbose: bool = False,
    max_missing_ratio: float = 0.05
) -> pd.Series:
    """
    一键清洗因子 (单截面)
    
    参数:
    -----
    factor : pd.Series
        原始因子值，index=股票代码
    industry : pd.Series
        行业分类，index=股票代码
    market_cap : pd.Series
        市值数据，index=股票代码
    steps : List[str], optional
        指定执行的步骤，默认 ['outlier', 'missing', 'neutralize', 'standardize']
        可选项: 'outlier'(去极值), 'missing'(填补), 'neutralize'(中性化), 'standardize'(标准化)
    verbose : bool, default=False
        是否打印详细日志
        
    返回:
    ------
    pd.Series : 清洗后的因子值，index保持不变
    
    流程:
    -----
    1. MAD去极值 (3倍MAD)
    2. 填补缺失值 (缺市值/行业剔除，低缺失率用行业中位数填补，高缺失率保留NaN)
    3. OLS中性化 (剥离行业+对数市值Beta)
    4. Z-Score标准化 (输出N(0,1))
    """
    if steps is None:
        steps = ['outlier', 'missing', 'neutralize', 'standardize']
    
    if verbose:
        print(f"开始因子清洗，初始样本数: {len(factor)}")
        missing_stats = get_missing_stats(factor, industry, market_cap)
        print(f"  缺失统计: 因子{missing_stats['missing_factor']}, "
              f"行业{missing_stats['missing_industry']}, "
              f"市值{missing_stats['missing_market_cap']}")
    
    result = factor.copy()
    
    # Step 1: 去极值
    if 'outlier' in steps:
        if verbose:
            print("Step 1: MAD去极值...")
        result = mad_winsorize(result, n_mad=3.0)
        if verbose:
            print(f"  完成，样本数: {len(result)}")
    
    # Step 2: 填补缺失
    if 'missing' in steps:
        if verbose:
            print("Step 2: 填补缺失值...")
        result, industry_clean, market_cap_clean = fill_missing(result, industry, market_cap, verbose=verbose, max_missing_ratio=max_missing_ratio)
        if verbose:
            print(f"  完成，剩余样本数: {len(result)}")
    else:
        industry_clean = industry
        market_cap_clean = market_cap
    
    # Step 3: 中性化
    if 'neutralize' in steps:
        if verbose:
            print("Step 3: OLS中性化...")
        result = neutralize(result, industry_clean, market_cap_clean)
        if verbose:
            print(f"  完成，样本数: {len(result)}")
    
    # Step 4: 标准化
    if 'standardize' in steps:
        if verbose:
            print("Step 4: Z-Score标准化...")
        result = zscore_standardize(result)
        if verbose:
            mean = result.mean()
            std = result.std()
            print(f"  完成，均值={mean:.6f}, 标准差={std:.4f}")
    
    if verbose:
        print(f"清洗完成，最终样本数: {len(result)}, 缺失值: {result.isna().sum()}")
    
    # 确保返回数值类型
    result = pd.to_numeric(result, errors='coerce')
    return result


def clean_factor_wide(
    factor_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    steps: Optional[List[str]] = None,
    verbose: bool = True,
    max_missing_ratio: float = 0.05
) -> pd.DataFrame:
    """
    一键清洗因子 (宽表格式，批量处理)
    
    参数:
    -----
    factor_df : pd.DataFrame
        原始因子宽表，index=日期, columns=股票代码
    industry_df : pd.DataFrame
        行业宽表，index=日期, columns=股票代码
    market_cap_df : pd.DataFrame
        市值宽表，index=日期, columns=股票代码
    steps : List[str], optional
        指定执行的步骤，默认全部
    verbose : bool, default=True
        是否打印进度
        
    返回:
    ------
    pd.DataFrame : 清洗后的因子宽表
        
    说明:
    -----
    每日截面独立处理，不跨天使用数据
    """
    if steps is None:
        steps = ['outlier', 'missing', 'neutralize', 'standardize']
    
    if factor_df.empty:
        return factor_df
    
    result = pd.DataFrame(index=factor_df.index, columns=factor_df.columns, dtype=float)
    
    total_dates = len(factor_df.index)
    
    for i, date in enumerate(factor_df.index, 1):
        if verbose and (i % 200 == 0 or i == 1 or i == total_dates):
            print(f"处理第 {i}/{total_dates} 天: {date}")
        
        # 获取当天数据
        day_factor = factor_df.loc[date]
        
        # 获取当天的行业和市值数据
        if date in industry_df.index:
            day_industry = industry_df.loc[date]
        else:
            if verbose:
                print(f"  警告: {date} 无行业数据，跳过")
            continue
            
        if date in market_cap_df.index:
            day_market_cap = market_cap_df.loc[date]
        else:
            if verbose:
                print(f"  警告: {date} 无市值数据，跳过")
            continue
        
        # 清洗当天数据
        try:
            day_clean = clean_factor(
                day_factor,
                day_industry,
                day_market_cap,
                steps=steps,
                verbose=False,
                max_missing_ratio=max_missing_ratio
            )
            # 确保是数值类型
            result.loc[date] = pd.to_numeric(day_clean, errors='coerce')
        except Exception as e:
            if verbose:
                print(f"  错误: {date} 处理失败: {e}")
            result.loc[date] = np.nan
    
    if verbose:
        print(f"\n宽表清洗完成，共 {total_dates} 天")
        print(f"最终数据形状: {result.shape}")
        print(f"总缺失值比例: {result.isna().sum().sum() / (result.shape[0] * result.shape[1]) * 100:.2f}%")
    
    return result


def clean_factor_summary(
    factor_raw: pd.Series,
    factor_clean: pd.Series
) -> dict:
    """
    生成清洗前后对比统计（用于报告）
    
    参数:
    -----
    factor_raw : pd.Series
        清洗前的因子
    factor_clean : pd.Series
        清洗后的因子
        
    返回:
    ------
    dict : 对比统计信息
    
    使用示例:
    --------
    summary = clean_factor_summary(raw, clean)
    print(f"均值变化: {summary['mean_change']:.4f}")
    """
    return {
        'raw_mean': factor_raw.mean(),
        'raw_std': factor_raw.std(),
        'raw_min': factor_raw.min(),
        'raw_max': factor_raw.max(),
        'clean_mean': factor_clean.mean(),
        'clean_std': factor_clean.std(),
        'clean_min': factor_clean.min(),
        'clean_max': factor_clean.max(),
        'mean_change': factor_clean.mean() - factor_raw.mean(),
        'std_change': factor_clean.std() - factor_raw.std(),
        'range_shrink': (factor_raw.max() - factor_raw.min()) / (factor_clean.max() - factor_clean.min() + 1e-10)
    }


if __name__ == "__main__":
    # 测试
    import numpy as np
    
    print("=" * 60)
    print("测试 pipeline 模块")
    print("=" * 60)
    
    # 构造测试数据
    np.random.seed(42)
    n = 100
    stocks = [f"{i:06d}.SZ" for i in range(1, n+1)]
    
    # 行业
    industries = ['银行'] * 20 + ['医药'] * 20 + ['科技'] * 20 + ['消费'] * 20 + ['能源'] * 20
    industry = pd.Series(industries, index=stocks)
    
    # 市值（对数正态分布）
    market_cap = pd.Series(np.random.lognormal(20, 0.5, n), index=stocks)
    
    # 原始因子（含极端值和缺失）
    factor_raw = pd.Series(
        np.where(np.array(industries) == '银行', 2.0, 0.0) +  # 银行业因子高
        np.random.normal(0, 1, n),
        index=stocks
    )
    # 添加极端值
    factor_raw.iloc[0] = 100  # 极端大值
    factor_raw.iloc[1] = -50  # 极端小值
    # 添加缺失值
    factor_raw.iloc[95:100] = np.nan
    
    print("\n原始因子统计:")
    print(f"  样本数: {len(factor_raw)}")
    print(f"  均值: {factor_raw.mean():.4f}")
    print(f"  标准差: {factor_raw.std():.4f}")
    print(f"  最小值: {factor_raw.min():.4f}")
    print(f"  最大值: {factor_raw.max():.4f}")
    print(f"  缺失值: {factor_raw.isna().sum()}")
    
    # 一键清洗
    print("\n执行一键清洗...")
    factor_clean = clean_factor(factor_raw, industry, market_cap, verbose=True)
    
    print("\n清洗后因子统计:")
    print(f"  样本数: {len(factor_clean)}")
    print(f"  均值: {factor_clean.mean():.6f} (应≈0)")
    print(f"  标准差: {factor_clean.std():.4f} (应≈1)")
    print(f"  最小值: {factor_clean.min():.4f}")
    print(f"  最大值: {factor_clean.max():.4f}")
    print(f"  缺失值: {factor_clean.isna().sum()}")
    
    # 对比统计
    summary = clean_factor_summary(factor_raw, factor_clean)
    print("\n对比摘要:")
    print(f"  均值变化: {summary['mean_change']:.4f}")
    print(f"  标准差变化: {summary['std_change']:.4f}")
    print(f"  值域收缩比: {summary['range_shrink']:.2f}x")
    
    print("\n✓ pipeline 模块测试完成")
