# -*- coding: utf-8 -*-
"""
中性化模块 - OLS残差法

功能:
    使用OLS线性回归剥离行业Beta和市值Beta，提取残差Alpha

使用示例:
    from processors.neutralizer import neutralize
    
    # 单个截面
    alpha = neutralize(factor, industry, market_cap)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional


def neutralize(
    factor: pd.Series,
    industry: pd.Series,
    market_cap: pd.Series,
    min_industry_stocks: int = 2
) -> pd.Series:
    """
    OLS残差法中性化 (单截面)
    
    参数:
    -----
    factor : pd.Series
        待中性化的因子值，index=股票代码
    industry : pd.Series
        行业分类，index=股票代码
    market_cap : pd.Series
        市值数据（总市值），index=股票代码，用于计算对数市值
    min_industry_stocks : int, default=2
        行业最少股票数，少于该数的行业合并为"其他"
        
    返回:
    ------
    pd.Series : 中性化后的残差(Alpha)，index保持不变
    
    模型:
    -----
    因子 = β₀ + β₁×行业dummy₁ + ... + βₙ×行业dummyₙ + βₘ×对数市值 + ε
    
    输出: ε (残差)
    
    说明:
    -----
    1. 市值取对数处理（减少右偏影响）
    2. 行业使用dummy变量（独热编码）
    3. 少于min_industry_stocks的行业合并为"其他"类别
    4. 如果某行业全为同一值，跳过该行业dummy（避免完全共线）
    """
    if factor.empty:
        return factor
    
    # 复制数据
    y = factor.copy()
    
    # 对齐索引
    aligned_industry = industry.reindex(y.index)
    aligned_market_cap = market_cap.reindex(y.index)
    
    # 检查缺失值（应在之前步骤已处理，但再次检查）
    valid_mask = y.notna() & aligned_industry.notna() & aligned_market_cap.notna()
    
    if valid_mask.sum() < 3:
        # 有效样本太少，无法回归，返回原值或0
        print("  警告: 有效样本少于3，跳过中性化")
        return y
    
    y_valid = y[valid_mask]
    industry_valid = aligned_industry[valid_mask]
    market_cap_valid = aligned_market_cap[valid_mask]
    
    # 确保 y_valid 是数值类型
    if y_valid.dtype == object:
        try:
            y_valid = pd.to_numeric(y_valid, errors='coerce')
        except:
            print("  警告: 因子无法转为数值类型，跳过中性化")
            return y
    
    # 计算对数市值
    # 确保 market_cap_valid 是数值类型
    market_cap_valid = pd.to_numeric(market_cap_valid, errors='coerce')
    log_market_cap = np.log(market_cap_valid.replace(0, np.nan))
    log_market_cap = log_market_cap.fillna(log_market_cap.median())
    
    # 确保是 float 类型
    log_market_cap = log_market_cap.astype(float)
    
    # 处理小行业（合并为"其他"）
    industry_counts = industry_valid.value_counts()
    small_industries = industry_counts[industry_counts < min_industry_stocks].index
    
    if len(small_industries) > 0:
        industry_valid = industry_valid.copy()
        industry_valid[industry_valid.isin(small_industries)] = "其他"
    
    # 创建行业dummy变量
    # 确保 industry_valid 是字符串类型
    industry_valid = industry_valid.astype(str)
    
    industry_dummies = pd.get_dummies(industry_valid, prefix='ind', drop_first=False)
    
    # 确保 dummy 变量是数值类型
    industry_dummies = industry_dummies.astype(float)
    
    # 检查并移除常量行业列（避免完全共线）
    constant_cols = []
    for col in industry_dummies.columns:
        if industry_dummies[col].std() < 1e-10:  # 该列全为0或全为1
            constant_cols.append(col)
    
    if constant_cols:
        industry_dummies = industry_dummies.drop(columns=constant_cols)
    
    # 如果行业dummy为空（只有一种行业），跳过中性化
    if industry_dummies.shape[1] == 0:
        print("  警告: 只有一种行业，无法中性化，返回原值")
        return y
    
    # 构建自变量矩阵
    X = pd.concat([industry_dummies, log_market_cap.rename('log_market_cap')], axis=1)
    
    # 确保 X 所有列都是数值类型
    for col in X.columns:
        if X[col].dtype == object:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                print(f"  警告: 列 {col} 无法转为数值，移除该列")
                X = X.drop(columns=[col])
    
    # 确保全部是 float 类型
    X = X.astype(float)
    
    # 添加常数项
    X = sm.add_constant(X)
    
    # 检查共线性（如果行业数和样本数接近，可能会共线）
    if X.shape[1] >= X.shape[0]:
        print(f"  警告: 变量数({X.shape[1]}) >= 样本数({X.shape[0]})，移除部分行业dummy")
        # 保留最大的几个行业
        keep_cols = ['const', 'log_market_cap'] + list(industry_dummies.columns[:X.shape[0]-3])
        X = X[keep_cols]
    
    # OLS回归
    try:
        model = sm.OLS(y_valid, X, missing='drop')
        results = model.fit()
        
        # 提取残差
        residuals = results.resid
        
        # 将残差映射回原始索引
        y_neutralized = pd.Series(index=y.index, dtype=float)
        y_neutralized[residuals.index] = residuals
        
        # 不填充NaN：原始因子为NaN的股票保持NaN，后续all()过滤会剔除
        return y_neutralized
        
    except Exception as e:
        print(f"  警告: OLS回归失败: {e}，返回原值")
        return y


def neutralize_wide(
    factor_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    min_industry_stocks: int = 2
) -> pd.DataFrame:
    """
    OLS残差法中性化 (宽表格式，批量处理)
    
    参数:
    -----
    factor_df : pd.DataFrame
        因子宽表，index=日期, columns=股票代码
    industry_df : pd.DataFrame
        行业宽表，index=日期, columns=股票代码
    market_cap_df : pd.DataFrame
        市值宽表，index=日期, columns=股票代码
    min_industry_stocks : int, default=2
        行业最少股票数
        
    返回:
    ------
    pd.DataFrame : 中性化后的因子宽表
    
    说明:
    -----
    每日截面独立回归，不跨天使用数据
    """
    if factor_df.empty:
        return factor_df
    
    result = factor_df.copy()
    
    for date in result.index:
        day_factor = result.loc[date]
        
        # 获取当天的行业和市值数据
        if date in industry_df.index:
            day_industry = industry_df.loc[date]
        else:
            continue  # 跳过没有行业数据的日期
            
        if date in market_cap_df.index:
            day_market_cap = market_cap_df.loc[date]
        else:
            continue  # 跳过没有市值数据的日期
        
        # 中性化
        day_neutralized = neutralize(day_factor, day_industry, day_market_cap, min_industry_stocks)
        result.loc[date] = day_neutralized
    
    return result


def get_neutralize_info(
    factor: pd.Series,
    industry: pd.Series,
    market_cap: pd.Series
) -> dict:
    """
    获取中性化前的统计信息（用于调试）
    
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
    dict : 统计信息
    """
    aligned_industry = industry.reindex(factor.index)
    aligned_market_cap = market_cap.reindex(factor.index)
    
    valid_mask = factor.notna() & aligned_industry.notna() & aligned_market_cap.notna()
    
    industry_counts = aligned_industry[valid_mask].value_counts()
    
    return {
        'total_stocks': len(factor),
        'valid_stocks': valid_mask.sum(),
        'n_industries': len(industry_counts),
        'industry_distribution': industry_counts.to_dict(),
        'mean_market_cap': aligned_market_cap[valid_mask].mean(),
        'median_market_cap': aligned_market_cap[valid_mask].median()
    }


if __name__ == "__main__":
    # 测试
    print("测试 neutralizer 模块...")
    
    # 构造测试数据
    np.random.seed(42)
    n = 100
    stocks = [f"{i:06d}.SZ" for i in range(1, n+1)]
    
    # 行业（5个行业，每个20只）
    industries = ['银行'] * 20 + ['医药'] * 20 + ['科技'] * 20 + ['消费'] * 20 + ['能源'] * 20
    industry = pd.Series(industries, index=stocks)
    
    # 市值（不同行业有不同市值水平）
    market_cap = pd.Series(
        np.concatenate([
            np.random.lognormal(20, 0.5, 20),   # 银行：大市值
            np.random.lognormal(19, 0.6, 20),   # 医药
            np.random.lognormal(18, 0.7, 20),   # 科技：小市值
            np.random.lognormal(19.5, 0.5, 20), # 消费
            np.random.lognormal(20.5, 0.4, 20)  # 能源：大市值
        ]),
        index=stocks
    )
    
    # 构造一个与行业和市值相关的因子
    # 银行行业因子高，科技行业因子低，且与市值正相关
    factor = pd.Series(
        np.where(industries == '银行', 2.0,
                np.where(industries == '科技', -1.0, 0.0)) + 
        np.log(market_cap) * 0.1 + 
        np.random.normal(0, 0.5, n),
        index=stocks
    )
    
    print("\n中性化前统计:")
    stats_before = get_neutralize_info(factor, industry, market_cap)
    print(f"  总股票数: {stats_before['total_stocks']}")
    print(f"  有效股票数: {stats_before['valid_stocks']}")
    print(f"  行业数: {stats_before['n_industries']}")
    print(f"  行业分布: {stats_before['industry_distribution']}")
    
    print(f"\n原始因子与行业的相关性:")
    for ind in ['银行', '医药', '科技', '消费', '能源']:
        mask = industry == ind
        print(f"  {ind}: 均值={factor[mask].mean():.3f}, 标准差={factor[mask].std():.3f}")
    
    # 中性化
    print("\n执行中性化...")
    factor_neutral = neutralize(factor, industry, market_cap)
    
    print("\n中性化后统计:")
    print(f"  残差均值: {factor_neutral.mean():.6f} (应≈0)")
    print(f"  残差标准差: {factor_neutral.std():.3f}")
    
    print(f"\n中性化后因子与行业的相关性（应接近0）:")
    for ind in ['银行', '医药', '科技', '消费', '能源']:
        mask = industry == ind
        print(f"  {ind}: 均值={factor_neutral[mask].mean():.6f}")
    
    print("\n✓ neutralizer 模块测试完成")
