# -*- coding: utf-8 -*-
"""
技术因子统一计算入口脚本

计算并保存所有技术因子（动量+波动率），可选择是否进行清洗。

用法：
    python main_compute_technical.py
    
可选参数：
    --family   : 指定因子家族，如 momentum, volatility
    --factors  : 指定要计算的因子，如 ret20 std20
    --list     : 列出所有可用因子
    --skip-clean : 跳过清洗步骤
"""

import argparse
import sys
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

from momentum import MomentumFactors
from volatility import VolatilityFactors
from liquidity import LiquidityFactors
from price_volume import PriceVolumeFactors
from processors.pipeline import clean_factor_wide


# 所有可用的技术因子注册表
TECHNICAL_FACTORS = {
    # 动量家族
    'ret1': {'family': 'momentum', 'method': 'factor_ret1', 'desc': '1日收益率(短期反转)'},
    'ret5': {'family': 'momentum', 'method': 'factor_ret5', 'desc': '5日收益率(周度反转)'},
    'ret20': {'family': 'momentum', 'method': 'factor_ret20', 'desc': '20日收益率'},
    'ret60': {'family': 'momentum', 'method': 'factor_ret60', 'desc': '60日收益率'},
    'ret120': {'family': 'momentum', 'method': 'factor_ret120', 'desc': '120日收益率'},
    'ret20_60': {'family': 'momentum', 'method': 'factor_ret20_60', 'desc': '动量差(ret20-ret60)'},
    
    # 波动率家族
    'std20': {'family': 'volatility', 'method': 'factor_std20', 'desc': '20日波动率'},
    'std60': {'family': 'volatility', 'method': 'factor_std60', 'desc': '60日波动率'},
    'atr20': {'family': 'volatility', 'method': 'factor_atr20', 'desc': '20日平均真实波幅'},
    'volatility_regime': {'family': 'volatility', 'method': 'factor_volatility_regime', 'desc': '波动率状态(std20/std60)'},
    
    # 流动性家族
    'amihud': {'family': 'liquidity', 'method': 'factor_amihud', 'desc': 'Amihud非流动性(|ret|/amount)'},
    'pv_corr20': {'family': 'liquidity', 'method': 'factor_pv_corr20', 'desc': '20日量价相关性'},
    'vol_trend': {'family': 'liquidity', 'method': 'factor_vol_trend', 'desc': '成交量趋势(5日/20日)'},
    'amount_ratio': {'family': 'liquidity', 'method': 'factor_amount_ratio', 'desc': '成交额比率(当日/20日均)'},
    
    # 价格-成交量家族
    'close_position': {'family': 'price_volume', 'method': 'factor_close_position', 'desc': '收盘价位置(日内)'},
    'intraday_return_ma5': {'family': 'price_volume', 'method': 'factor_intraday_return_ma5', 'desc': '日内收益率5日均'},
    'intraday_return_ma20': {'family': 'price_volume', 'method': 'factor_intraday_return_ma20', 'desc': '日内收益率20日均'},
    'close_position_ma5': {'family': 'price_volume', 'method': 'factor_close_position_ma5', 'desc': '收盘价位置5日均'},
    'close_position_ma20': {'family': 'price_volume', 'method': 'factor_close_position_ma20', 'desc': '收盘价位置20日均'},
    'skew20': {'family': 'price_volume', 'method': 'factor_skew20', 'desc': '20日收益偏度'},
    'kurt20': {'family': 'price_volume', 'method': 'factor_kurt20', 'desc': '20日收益峰度'},
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


def clean_factor(factor_name: str, factor_df: pd.DataFrame, processed_data_path: Path):
    """
    清洗技术因子
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
                          momentum: MomentumFactors, volatility: VolatilityFactors,
                          liquidity: LiquidityFactors, price_volume: PriceVolumeFactors,
                          skip_clean: bool = False) -> Path:
    """
    计算单个因子
    
    参数:
    -----
    factor_name : str
        因子名称
    factor_info : dict
        因子信息（家族、方法名、描述）
    momentum : MomentumFactors
        动量因子计算器实例
    volatility : VolatilityFactors
        波动率因子计算器实例
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
    if factor_info['family'] == 'momentum':
        calculator = momentum
        dates, stocks, _ = momentum._to_numpy('close')
        output_path = momentum.output_path
    elif factor_info['family'] == 'volatility':
        calculator = volatility
        dates, stocks, _ = volatility._get_numpy_matrix('close')
        output_path = volatility.output_path
    elif factor_info['family'] == 'liquidity':
        calculator = liquidity
        dates, stocks, _ = liquidity._to_numpy('close')
        output_path = liquidity.output_path
    else:  # price_volume
        calculator = price_volume
        dates, stocks, _ = price_volume._to_numpy('close')
        output_path = price_volume.output_path
    
    # 调用计算方法
    method = getattr(calculator, factor_info['method'])
    factor_matrix = method(save=False)
    
    # 构建 DataFrame
    factor_df = pd.DataFrame(factor_matrix, index=dates, columns=stocks)
    
    if skip_clean:
        # 不清洗，直接保存
        factor_clean = factor_df
        print(f"\n[跳过清洗] 直接保存原始因子")
    else:
        # 清洗
        processed_data_path = output_path.parent.parent
        factor_clean = clean_factor(factor_name, factor_df, processed_data_path)
    
    # 保存
    output_file = output_path / f"{factor_name}.parquet"
    
    dates_arr = factor_clean.index
    arrays = [pa.array(dates_arr, type=pa.timestamp('ns'))]
    names = ['time']
    
    for col in factor_clean.columns:
        col_data = factor_clean[col]
        # 处理 NaN 和 Inf
        col_list = [
            None if (pd.isna(v) or np.isinf(v) if pd.notna(v) else True) else float(v)
            for v in col_data
        ]
        arrays.append(pa.array(col_list, type=pa.float64()))
        names.append(col)
    
    table = pa.table(arrays, names=names)
    pq.write_table(table, output_file)
    
    print(f"\n✓ 已保存: {output_file}")
    print(f"  维度: {table.num_rows} 行 × {table.num_columns} 列")
    
    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='技术因子统一计算工具（动量+波动率）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 计算所有技术因子（含清洗）
  python main_compute_technical.py
  
  # 只计算动量家族
  python main_compute_technical.py --family momentum
  
  # 只计算波动率家族
  python main_compute_technical.py --family volatility
  
  # 指定特定因子
  python main_compute_technical.py --factors ret20 std20 atr20
  
  # 只计算原始因子，不清洗
  python main_compute_technical.py --skip-clean
  
  # 列出所有可用因子
  python main_compute_technical.py --list
        """
    )
    
    parser.add_argument(
        '--family',
        choices=['momentum', 'volatility', 'liquidity', 'price_volume', 'all'],
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
        help='跳过清洗步骤，只计算原始因子'
    )
    
    args = parser.parse_args()
    
    # 列出可用因子
    if args.list:
        print("=" * 60)
        print("可用技术因子列表")
        print("=" * 60)
        
        print("\n【动量家族 (momentum)】")
        for name, info in TECHNICAL_FACTORS.items():
            if info['family'] == 'momentum':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【波动率家族 (volatility)】")
        for name, info in TECHNICAL_FACTORS.items():
            if info['family'] == 'volatility':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【流动性家族 (liquidity)】")
        for name, info in TECHNICAL_FACTORS.items():
            if info['family'] == 'liquidity':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【价格-成交量家族 (price_volume)】")
        for name, info in TECHNICAL_FACTORS.items():
            if info['family'] == 'price_volume':
                print(f"  - {name:15s}: {info['desc']}")
        
        print("\n【使用示例】")
        print("  python main_compute_technical.py --family momentum")
        print("  python main_compute_technical.py --family liquidity")
        print("  python main_compute_technical.py --factors ret1 amihud close_position")
        return 0
    
    # 确定要计算的因子列表
    if args.factors:
        # 用户指定了具体因子
        factors_to_compute = {}
        for f in args.factors:
            if f in TECHNICAL_FACTORS:
                factors_to_compute[f] = TECHNICAL_FACTORS[f]
            else:
                print(f"警告: 未知因子 '{f}'，已跳过")
        
        if not factors_to_compute:
            print("错误: 没有有效的因子可计算")
            return 1
    else:
        # 按家族筛选
        if args.family == 'all':
            factors_to_compute = TECHNICAL_FACTORS
        else:
            factors_to_compute = {
                name: info for name, info in TECHNICAL_FACTORS.items()
                if info['family'] == args.family
            }
    
    # 主流程
    print("=" * 60)
    print("技术因子统一计算工具")
    print("=" * 60)
    print(f"\n本次将计算 {len(factors_to_compute)} 个因子:")
    for name, info in factors_to_compute.items():
        print(f"  - {name} ({info['desc']})")
    
    # 初始化计算器（延迟加载数据）
    momentum = None
    volatility = None
    liquidity = None
    price_volume = None
    
    # 检查需要哪些家族的计算器
    families_needed = set(info['family'] for info in factors_to_compute.values())
    
    if 'momentum' in families_needed:
        print("\n【初始化】动量因子计算器...")
        momentum = MomentumFactors()
    
    if 'volatility' in families_needed:
        print("\n【初始化】波动率因子计算器...")
        volatility = VolatilityFactors()
    
    if 'liquidity' in families_needed:
        print("\n【初始化】流动性因子计算器...")
        liquidity = LiquidityFactors()
    
    if 'price_volume' in families_needed:
        print("\n【初始化】价格-成交量因子计算器...")
        price_volume = PriceVolumeFactors()
    
    # 计算所有因子
    output_files = []
    
    for factor_name, factor_info in factors_to_compute.items():
        try:
            output_file = compute_single_factor(
                factor_name, factor_info,
                momentum, volatility, liquidity, price_volume,
                skip_clean=args.skip_clean
            )
            output_files.append(output_file)
        except Exception as e:
            print(f"\n✗ 计算因子 {factor_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 60)
    print(f"计算完成！共生成 {len(output_files)} 个因子文件")
    print("=" * 60)
    
    for f in output_files:
        print(f"  ✓ {f.name}")
    
    print(f"\n输出目录: {output_files[0].parent if output_files else 'N/A'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
