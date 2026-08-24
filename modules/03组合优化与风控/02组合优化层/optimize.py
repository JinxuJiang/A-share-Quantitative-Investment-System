"""02组合优化层的日常入口：逐周生成完整历史权重表。"""

from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    run_path(str(Path(__file__).with_name("optimization.py")), run_name="__main__")
