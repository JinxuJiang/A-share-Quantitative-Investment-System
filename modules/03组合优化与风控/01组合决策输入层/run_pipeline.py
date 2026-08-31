"""组合决策输入层的统一运行入口。

一次正式运行必须完成以下两个阶段，而且顺序不能颠倒：
    1. 从截面模型、市场模型同步正式输出及全量行情；
    2. 基于本地数据构建并发布全部日度决策历史表。

在 qf 环境中，日常只需要执行：
    python 01组合决策输入层/run_pipeline.py

不加任何参数时，程序会自动完成上述两个阶段。以下参数只用于维护和排错，
不能代替一次完整的正式运行：
    --sync-only  只检查或更新本地源数据，不产生新快照；
    --skip-sync  基于已同步的数据调试或重建快照，不检查上游是否更新。

本文件只负责编排，不在这里实现数据读取、Alpha 转换或协方差估计。
具体业务逻辑分别保留在 adapters 和 estimation 中，便于单独测试和定位错误。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# 所有相对路径均以“01组合决策输入层”为基准，避免从不同工作目录运行时找错文件。
LAYER_ROOT = Path(__file__).resolve().parent
SYNC_SCRIPT = LAYER_ROOT / "adapters" / "sync_source_data.py"
BUILD_SCRIPT = LAYER_ROOT / "estimation" / "build_decision_snapshot.py"


def parse_args() -> argparse.Namespace:
    """解析统一入口参数，并保证两种单阶段模式不能同时启用。"""

    parser = argparse.ArgumentParser(
        description="运行组合决策输入层：先同步源数据，再构建全部日度历史表",
        epilog=(
            "正式运行（两个阶段都会执行）：\n"
            "  python 01组合决策输入层/run_pipeline.py\n\n"
            "维护模式：\n"
            "  python 01组合决策输入层/run_pipeline.py --sync-only\n"
            "  python 01组合决策输入层/run_pipeline.py --skip-sync --start-date 2024-01-01"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --sync-only 与 --skip-sync 语义冲突，因此使用互斥参数组提前阻止错误组合。
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sync-only", action="store_true", help="维护模式：只同步源数据，不构建快照")
    mode.add_argument("--skip-sync", action="store_true", help="调试模式：复用本地数据，直接构建快照")

    # 以下参数会原样传给对应的子阶段；总入口不解释配置内容和日期业务规则。
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-date", help="可选历史起始signal_date")
    parser.add_argument("--end-date", help="可选历史结束signal_date")
    parser.add_argument("--release-id", default=None, help="可选的正式发布版本号")
    return parser.parse_args()


def run(command: list[str]) -> None:
    """在输入层根目录运行子阶段，并在子阶段失败时立即终止总流程。

    使用 ``sys.executable`` 调用子脚本，确保它们与本入口使用同一个 Python
    解释器；例如从 qf 环境启动本入口时，两个子阶段也一定运行在 qf 环境。
    """

    # 先打印完整命令，方便从终端日志判断失败发生在哪个阶段。
    print(f"\n> {' '.join(command)}", flush=True)

    # check=True 会保留子脚本的非零退出码，不会在同步失败后继续发布快照。
    subprocess.run(command, cwd=LAYER_ROOT, check=True)


def main() -> None:
    """正式模式依次完成两个阶段；显式维护参数才会跳过其中一个阶段。"""

    args = parse_args()

    # 默认和 --sync-only 模式都需要同步；只有 --skip-sync 明确复用本地数据。
    if not args.skip_sync:
        run([sys.executable, str(SYNC_SCRIPT), "--config", args.config])

    # 只同步时到此正常结束，不触发任何估计或正式发布。
    if args.sync_only:
        return

    # 构造第二阶段命令。默认构建全部共同交易日。
    build_command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--config",
        args.config,
    ]

    if args.start_date:
        build_command.extend(["--start-date", args.start_date])
    if args.end_date:
        build_command.extend(["--end-date", args.end_date])

    # 未指定 release_id 时不传该参数，让构建脚本按照日期生成默认版本号。
    if args.release_id:
        build_command.extend(["--release-id", args.release_id])
    run(build_command)


if __name__ == "__main__":
    # 仅在直接运行本文件时启动流程；被测试或其他模块导入时不会自动执行。
    main()
