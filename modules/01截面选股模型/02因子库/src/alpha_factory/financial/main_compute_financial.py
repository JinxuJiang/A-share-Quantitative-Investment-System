# -*- coding: utf-8 -*-
"""
财务因子统一计算入口脚本

计算并保存所有财务因子（估值+盈利），可选择是否进行清洗。

用法：
    python main_compute_financial.py
    
可选参数：
    --family   : 指定因子家族，如 valuation, profitability
    --factors  : 指定要计算的因子，如 pe roe
    --list     : 列出所有可用因子
    --skip-clean : 跳过清洗步骤
    例：
    python main_compute_financial.py --factor current_asset_ratio
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# 添加项目路径
current_file = Path(__file__).resolve()
current_dir = current_file.parent
factor_lib_root = current_dir.parent.parent.parent

sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(current_dir.parent.parent))
sys.path.insert(0, str(factor_lib_root))

from valuation import ValuationFactors
from profitability import ProfitabilityFactors
from growth import GrowthFactors
from quality import QualityFactors
from safety import SafetyFactors
from investment import InvestmentFactors
from efficiency import EfficiencyFactors
from processors.pipeline import clean_factor_wide


# 所有可用的财务因子注册表
FINANCIAL_FACTORS = {
    # 估值家族
    'pe': {'family': 'valuation', 'method': 'factor_pe', 'desc': '市盈率'},
    'pb': {'family': 'valuation', 'method': 'factor_pb', 'desc': '市净率'},
    'ps': {'family': 'valuation', 'method': 'factor_ps', 'desc': '市销率'},
    'ey': {'family': 'valuation', 'method': 'factor_ey', 'desc': '盈利收益率'},
    
    # 盈利家族 (4个原有 + 1个新增)
    'roe': {'family': 'profitability', 'method': 'factor_roe', 'desc': '净资产收益率'},
    'roa': {'family': 'profitability', 'method': 'factor_roa', 'desc': '总资产收益率'},
    'roe_growth': {'family': 'profitability', 'method': 'factor_roe_growth', 'desc': 'ROE同比增长'},
    'opm': {'family': 'profitability', 'method': 'factor_opm', 'desc': '营业利润率'},

    # 成长家族
    'profit_growth': {'family': 'growth', 'method': 'factor_profit_growth', 'desc': '净利润增长率'},
    'revenue_growth': {'family': 'growth', 'method': 'factor_revenue_growth', 'desc': '营收增长率'},
    'oper_profit_growth': {'family': 'growth', 'method': 'factor_oper_profit_growth', 'desc': '营业利润增长率'},
    
    # 质量家族 (3个原有 + 3个新增)
    'financial_leverage': {'family': 'quality', 'method': 'factor_financial_leverage', 'desc': '财务杠杆'},
    'profit_quality': {'family': 'quality', 'method': 'factor_profit_quality', 'desc': '利润质量'},
    'current_asset_ratio': {'family': 'quality', 'method': 'factor_current_asset_ratio', 'desc': '流动资产占比'},
    'accrual': {'family': 'quality', 'method': 'factor_accrual', 'desc': '应计利润比'},
    'cashflow_to_profit': {'family': 'quality', 'method': 'factor_cashflow_to_profit', 'desc': '现金流利润比'},
    'ocf_to_revenue': {'family': 'quality', 'method': 'factor_ocf_to_revenue', 'desc': '收现率'},
    
    # 安全家族 (新建3个)
    'debt_to_equity': {'family': 'safety', 'method': 'factor_debt_to_equity', 'desc': '产权比率'},
    'current_ratio': {'family': 'safety', 'method': 'factor_current_ratio', 'desc': '流动比率'},
    'cash_ratio': {'family': 'safety', 'method': 'factor_cash_ratio', 'desc': '现金比率'},
    
    # 投资家族 (新建2个)
    'asset_growth': {'family': 'investment', 'method': 'factor_asset_growth', 'desc': '总资产增长率'},
    'capex_to_assets': {'family': 'investment', 'method': 'factor_capex_to_assets', 'desc': '资本支出强度'},
    
    # 效率家族 (新建2个)
    'asset_turnover': {'family': 'efficiency', 'method': 'factor_asset_turnover', 'desc': '资产周转率'},
    'working_capital_ratio': {'family': 'efficiency', 'method': 'factor_working_capital_ratio', 'desc': '营运资本占比'},
}


def load_factor(file_path: Path, index_col: str = 'time') -> pd.DataFrame:
    """
    通用因子数据加载器
    """
    df = pq.read_table(file_path).to_pandas()
    
    if index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    return df


def clean_financial_factor(factor_name: str, factor_df: pd.DataFrame, processed_data_path: Path):
    """
    清洗财务因子
    """
    print(f"\n【清洗】{factor_name}...")
    
    # 加载行业数据
    industry_file = processed_data_path / "financial_data" / "industry.parquet"
    if not industry_file.exists():
        print(f"  警告: 行业数据不存在，跳过清洗")
        return factor_df
    
    industry_df = load_factor(industry_file)
    
    # 加载市值数据
    close_file = processed_data_path / "market_data" / "close.parquet"
    cap_stk_file = processed_data_path / "financial_data" / "cap_stk.parquet"
    
    if not close_file.exists() or not cap_stk_file.exists():
        print(f"  警告: 市值数据不完整，跳过清洗")
        return factor_df
    
    close_df = load_factor(close_file)
    cap_stk_df = load_factor(cap_stk_file)
    
    # 计算市值
    common_cols = close_df.columns.intersection(cap_stk_df.columns)
    common_index = close_df.index.intersection(cap_stk_df.index)
    market_cap_df = (close_df.loc[common_index, common_cols] * 
                     cap_stk_df.loc[common_index, common_cols])
    
    # 对齐数据
    common_stocks = factor_df.columns.intersection(industry_df.columns).intersection(market_cap_df.columns)
    common_dates = factor_df.index.intersection(industry_df.index).intersection(market_cap_df.index)
    
    if len(common_stocks) == 0 or len(common_dates) == 0:
        print(f"  警告: 无共同数据，跳过清洗")
        return factor_df
    
    factor_df = factor_df.loc[common_dates, common_stocks]
    industry_df = industry_df.loc[common_dates, common_stocks]
    market_cap_df = market_cap_df.loc[common_dates, common_stocks]
    
    print(f"  共同股票数: {len(common_stocks)}, 共同交易日: {len(common_dates)}")
    
    # 清洗
    try:
        factor_clean = clean_factor_wide(
            factor_df,
            industry_df,
            market_cap_df,
            steps=['outlier', 'missing', 'neutralize', 'standardize'],
            verbose=False
        )
        print(f"  清洗完成")
        return factor_clean
    except Exception as e:
        print(f"  清洗失败: {e}，返回原始因子")
        return factor_df


def compute_single_factor(factor_name: str, factor_info: dict,
                          valuation: ValuationFactors, profitability: ProfitabilityFactors,
                          growth: GrowthFactors, quality: QualityFactors,
                          safety: SafetyFactors, investment: InvestmentFactors,
                          efficiency: EfficiencyFactors,
                          skip_clean: bool = False) -> Path:
    """
    计算单个财务因子
    
    参数:
    -----
    factor_name : str
        因子名称
    factor_info : dict
        因子信息
    valuation : ValuationFactors
        估值因子计算器
    profitability : ProfitabilityFactors
        盈利因子计算器
    safety : SafetyFactors
        安全因子计算器
    investment : InvestmentFactors
        投资因子计算器
    efficiency : EfficiencyFactors
        效率因子计算器
    skip_clean : bool
        是否跳过清洗
        
    返回:
    ------
    Path : 输出文件路径
    """
    print(f"\n{'='*60}")
    print(f"计算因子: {factor_name} ({factor_info['desc']})")
    print(f"{'='*60}")
    
    # 根据家族选择计算器
    if factor_info['family'] == 'valuation':
        calculator = valuation
    elif factor_info['family'] == 'profitability':
        calculator = profitability
    elif factor_info['family'] == 'growth':
        calculator = growth
    elif factor_info['family'] == 'quality':
        calculator = quality
    elif factor_info['family'] == 'safety':
        calculator = safety
    elif factor_info['family'] == 'investment':
        calculator = investment
    elif factor_info['family'] == 'efficiency':
        calculator = efficiency
    else:
        raise ValueError(f"未知家族: {factor_info['family']}")
    
    # 调用计算方法
    method = getattr(calculator, factor_info['method'])
    factor_df = method(save=False)  # 先不保存，等清洗后再存
    
    if skip_clean:
        factor_clean = factor_df
        print(f"\n[跳过清洗] 直接保存原始因子")
    else:
        processed_data_path = calculator.processed_data_path
        factor_clean = clean_financial_factor(factor_name, factor_df, processed_data_path)
    
    # 保存
    output_file = calculator.output_path / f"{factor_name}.parquet"
    
    dates_arr = factor_clean.index
    arrays = [pa.array(dates_arr, type=pa.timestamp('ns'))]
    names = ['time']
    
    for col in factor_clean.columns:
        col_data = factor_clean[col]
        col_list = [
            None if (pd.isna(v) or np.isinf(v) if pd.notna(v) else True) else float(v)
            for v in col_data
        ]
        arrays.append(pa.array(col_list, type=pa.float64()))
        names.append(col)
    
    table = pa.table(arrays, names=names)
    pq.write_table(table, output_file)
    
    print(f"\n已保存: {output_file}")
    print(f"  维度: {table.num_rows} 行 × {table.num_columns} 列")
    
    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='财务因子统一计算工具（估值+盈利）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 计算所有财务因子（含清洗）
  python main_compute_financial.py
  
  # 只计算估值家族
  python main_compute_financial.py --family valuation
  
  # 只计算盈利家族
  python main_compute_financial.py --family profitability
  
  # 指定特定因子
  python main_compute_financial.py --factors pe pb roe
  
  # 只计算原始因子，不清洗
  python main_compute_financial.py --skip-clean
  
  # 列出所有可用因子
  python main_compute_financial.py --list
        """
    )
    
    parser.add_argument(
        '--family',
        choices=['valuation', 'profitability', 'growth', 'quality', 'safety', 'investment', 'efficiency', 'all'],
        default='all',
        help='指定因子家族，默认 all'
    )
    
    parser.add_argument(
        '--factors',
        nargs='+',
        default=None,
        help='指定要计算的因子列表'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用因子'
    )
    
    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='跳过清洗步骤'
    )
    
    args = parser.parse_args()
    
    # 列出可用因子
    if args.list:
        print("=" * 60)
        print("可用财务因子列表")
        print("=" * 60)
        
        print("\n【估值家族 (valuation)】")
        for name, info in FINANCIAL_FACTORS.items():
            if info['family'] == 'valuation':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【盈利家族 (profitability)】")
        for name, info in FINANCIAL_FACTORS.items():
            if info['family'] == 'profitability':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【成长家族 (growth)】")
        for name, info in FINANCIAL_FACTORS.items():
            if info['family'] == 'growth':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【质量家族 (quality)】")
        for name, info in FINANCIAL_FACTORS.items():
            if info['family'] == 'quality':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【安全家族 (safety)】")
        for name, info in FINANCIAL_FACTORS.items():
            if info['family'] == 'safety':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【投资家族 (investment)】")
        for name, info in FINANCIAL_FACTORS.items():
            if info['family'] == 'investment':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【效率家族 (efficiency)】")
        for name, info in FINANCIAL_FACTORS.items():
            if info['family'] == 'efficiency':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【使用示例】")
        print("  python main_compute_financial.py --family valuation")
        print("  python main_compute_financial.py --factors pe pb roe")
        return 0
    
    # 确定要计算的因子列表
    if args.factors:
        factors_to_compute = {}
        for f in args.factors:
            if f in FINANCIAL_FACTORS:
                factors_to_compute[f] = FINANCIAL_FACTORS[f]
            else:
                print(f"警告: 未知因子 '{f}'，已跳过")
        
        if not factors_to_compute:
            print("错误: 没有有效的因子可计算")
            return 1
    else:
        if args.family == 'all':
            factors_to_compute = FINANCIAL_FACTORS
        else:
            factors_to_compute = {
                name: info for name, info in FINANCIAL_FACTORS.items()
                if info['family'] == args.family
            }
    
    # 主流程
    print("=" * 60)
    print("财务因子统一计算工具")
    print("=" * 60)
    print(f"\n本次将计算 {len(factors_to_compute)} 个因子:")
    for name, info in factors_to_compute.items():
        print(f"  - {name} ({info['desc']})")
    
    # 初始化计算器
    families_needed = set(info['family'] for info in factors_to_compute.values())
    
    valuation = None
    profitability = None
    growth = None
    quality = None
    safety = None
    investment = None
    efficiency = None
    
    if 'valuation' in families_needed:
        print("\n【初始化】估值因子计算器...")
        valuation = ValuationFactors()
    
    if 'profitability' in families_needed:
        print("\n【初始化】盈利因子计算器...")
        profitability = ProfitabilityFactors()
    
    if 'growth' in families_needed:
        print("\n【初始化】成长因子计算器...")
        growth = GrowthFactors()
    
    if 'quality' in families_needed:
        print("\n【初始化】质量因子计算器...")
        quality = QualityFactors()
    
    if 'safety' in families_needed:
        print("\n【初始化】安全因子计算器...")
        safety = SafetyFactors()
    
    if 'investment' in families_needed:
        print("\n【初始化】投资因子计算器...")
        investment = InvestmentFactors()
    
    if 'efficiency' in families_needed:
        print("\n【初始化】效率因子计算器...")
        efficiency = EfficiencyFactors()
    
    # 计算所有因子
    output_files = []
    
    for factor_name, factor_info in factors_to_compute.items():
        try:
            output_file = compute_single_factor(
                factor_name, factor_info,
                valuation, profitability, growth, quality,
                safety, investment, efficiency,
                skip_clean=args.skip_clean
            )
            output_files.append(output_file)
        except Exception as e:
            print(f"\n计算因子 {factor_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 60)
    print(f"计算完成！共生成 {len(output_files)} 个因子文件")
    print("=" * 60)
    
    for f in output_files:
        print(f"  - {f.name}")
    
    print(f"\n输出目录: {output_files[0].parent if output_files else 'N/A'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
