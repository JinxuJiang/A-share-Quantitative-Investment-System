# -*- coding: utf-8 -*-
"""
市场数据准备入口脚本

将原始行情数据转换为宽表格式，供后续因子计算使用。

用法：
    python main_prepare_market_data.py
    
可选参数：
    --fields : 指定要处理的字段，默认处理所有字段
    --start-date : 开始日期 (YYYY-MM-DD)
    --end-date : 结束日期 (YYYY-MM-DD)
    --overwrite : 覆盖已存在的文件
"""

import argparse
import sys
from pathlib import Path

from market_data_loader import MarketDataLoader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='准备市场数据：将原始行情数据转换为宽表格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 准备所有字段
  python main_prepare_market_data.py
  
  # 只准备 close 和 volume
  python main_prepare_market_data.py --fields close volume
  
  # 指定日期范围
  python main_prepare_market_data.py --start-date 2020-01-01 --end-date 2023-12-31
  
  # 强制覆盖已存在的文件
  python main_prepare_market_data.py --overwrite
        """
    )
    
    parser.add_argument(
        '--fields',
        nargs='+',
        choices=MarketDataLoader.VALID_FIELDS,
        default=None,
        help=f'要处理的字段列表，可选: {MarketDataLoader.VALID_FIELDS}，默认处理所有'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='开始日期，格式 YYYY-MM-DD'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='结束日期，格式 YYYY-MM-DD'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的文件'
    )
    
    args = parser.parse_args()
    
    # 初始化加载器
    print("=" * 60)
    print("市场数据准备工具")
    print("=" * 60)
    
    loader = MarketDataLoader()
    print(f"原始数据路径: {loader.raw_data_path}")
    print(f"输出路径: {loader.output_path}")
    
    # 执行数据准备
    print("\n开始处理...")
    print("-" * 60)
    
    output_files = loader.prepare_all_fields(
        fields=args.fields,
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite
    )
    
    print("\n" + "=" * 60)
    print("输出文件列表:")
    for f in output_files:
        print(f"  - {f}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
