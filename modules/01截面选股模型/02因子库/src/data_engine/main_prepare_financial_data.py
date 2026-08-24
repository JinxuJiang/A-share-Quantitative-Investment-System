# -*- coding: utf-8 -*-
"""
财务数据准备入口脚本

将原始财务数据转换为 PIT 对齐后的日度宽表。

用法：
    python main_prepare_financial_data.py
    
可选参数：
    --overwrite : 覆盖已存在的文件
    --industry-only : 只处理行业数据
    --financial-only : 只处理三张财务报表数据
    --fields : 指定要处理的财务字段
"""

import argparse
import sys
from pathlib import Path

from industry_loader import IndustryLoader
from financial_data_loader import FinancialDataLoader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='准备财务数据：PIT对齐和TTM计算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 处理所有财务数据
  python main_prepare_financial_data.py
  
  # 只处理行业数据
  python main_prepare_financial_data.py --industry-only

  # 只处理财务数据
  python main_prepare_financial_data.py --financial-only
  
  # 强制覆盖已存在文件
  python main_prepare_financial_data.py --overwrite
        """
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的文件'
    )
    
    parser.add_argument(
        '--industry-only',
        action='store_true',
        help='只处理行业数据'
    )

    parser.add_argument(
        '--financial-only',
        action='store_true',
        help='只处理财务数据，不重建行业文件'
    )
    
    parser.add_argument(
        '--fields',
        nargs='+',
        default=None,
        help='指定要处理的字段，如: --fields cap_stk tot_assets'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("财务数据准备工具")
    print("=" * 60)
    
    if args.industry_only and args.financial_only:
        parser.error('--industry-only 和 --financial-only 不能同时使用')

    industry_file = None
    if not args.financial_only:
        # 1. 处理行业数据
        print("\n【1/2】处理行业数据...")
        print("-" * 60)
        industry_loader = IndustryLoader()
        industry_file = industry_loader.prepare_industry_data(
            overwrite=args.overwrite
        )
    
    if args.industry_only:
        print("\n只处理行业数据，跳过财务数据...")
        print("=" * 60)
        return 0
    
    # 2. 处理财务数据
    print("\n【2/2】处理财务数据（PIT对齐 + TTM计算）...")
    print("-" * 60)
    
    try:
        financial_loader = FinancialDataLoader()
        output_files = financial_loader.prepare_all_fields(
            fields=args.fields,
            overwrite=args.overwrite
        )
        
        print("\n" + "=" * 60)
        print("输出文件列表:")
        if industry_file is not None:
            print(f"  - {industry_file} (行业数据)")
        for f in output_files:
            print(f"  - {f}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n提示: 请先运行 main_prepare_market_data.py 准备行情数据")
        return 1
    except Exception as e:
        print(f"\n处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
