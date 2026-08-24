#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子库全量更新脚本
一键运行所有因子计算流程
"""

import subprocess
import sys
import time
from datetime import datetime

# 基于本脚本位置的路径
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 4个执行步骤（按顺序）
STEPS = [
    ("行情宽表", "src/data_engine/main_prepare_market_data.py", ["--overwrite"]),
    ("财务PIT对齐", "src/data_engine/main_prepare_financial_data.py", ["--overwrite"]),
    ("技术因子", "src/alpha_factory/technical/main_compute_technical.py", []),
    ("财务因子", "src/alpha_factory/financial/main_compute_financial.py", []),
]


def run_step(name, script_path, args_list):
    """运行单个步骤"""
    full_path = os.path.join(BASE_DIR, script_path)
    cmd = [sys.executable, full_path] + args_list
    
    print(f"\n{'='*60}")
    print(f"步骤: {name}")
    print(f"脚本: {script_path}")
    if args_list:
        print(f"参数: {' '.join(args_list)}")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"❌ {name} 失败！")
        return False, elapsed
    
    print(f"✅ {name} 完成，耗时 {elapsed:.1f} 秒")
    return True, elapsed


def main():
    print("="*60)
    print("因子库全量更新")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    total_start = time.time()
    
    for name, script, args_list in STEPS:
        success, elapsed = run_step(name, script, args_list)
        if not success:
            print(f"\n❌ 更新中断！请检查错误。")
            return 1
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("✅ 所有步骤完成！")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
