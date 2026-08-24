# -*- coding: utf-8 -*-
"""
Processors 因子清洗模块

统一的因子清洗流程：
    去极值 → 填补缺失 → 中性化 → 标准化

使用示例:
    # 分步使用
    from processors import mad_winsorize, fill_missing, neutralize, zscore_standardize
    
    factor_clean = mad_winsorize(factor_raw)
    factor_clean, industry, market_cap = fill_missing(factor_clean, industry, market_cap)
    factor_clean = neutralize(factor_clean, industry, market_cap)
    factor_clean = zscore_standardize(factor_clean)
    
    # 或一键清洗
    from processors.pipeline import clean_factor
    factor_clean = clean_factor(factor_raw, industry, market_cap)
"""

# 导入主要函数，方便调用
from .outlier import mad_winsorize, mad_winsorize_wide, get_outlier_bounds
from .missing_value import fill_missing, fill_missing_wide, get_missing_stats
from .standardizer import zscore_standardize, zscore_standardize_wide, get_standardize_stats

# neutralizer 和 pipeline 在单独文件中，避免循环导入
# 使用时: from processors.neutralizer import neutralize
#         from processors.pipeline import clean_factor

__version__ = "1.0.0"

__all__ = [
    # Outlier
    'mad_winsorize',
    'mad_winsorize_wide',
    'get_outlier_bounds',
    
    # Missing Value
    'fill_missing',
    'fill_missing_wide',
    'get_missing_stats',
    
    # Standardizer
    'zscore_standardize',
    'zscore_standardize_wide',
    'get_standardize_stats',
]
