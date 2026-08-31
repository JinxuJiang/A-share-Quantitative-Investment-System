#!/usr/bin/env python
"""Run the complete A-share investment workflow with one command.

This file only orchestrates the existing project entry points. Business logic
remains in the three independent projects.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PORTFOLIO_ROOT = Path(__file__).resolve().parent
SYSTEM_ROOT = PORTFOLIO_ROOT.parent
CROSS_SECTION_ROOT = SYSTEM_ROOT / "01截面选股模型"
MARKET_ROOT = SYSTEM_ROOT / "02市场仓位模型"

CROSS_EXPERIMENTS = {
    5: "lgbm5_tushare_profit20_strictpit_v2",
    20: "lgbm20_tushare_profit20_strictpit_v2",
    60: "lgbm60_tushare_profit20_strictpit_v2",
}
FUSION_EXPERIMENT = "ensemble_5d_20d_60d_profit20_strictpit_v2"
CROSS_TRAINING_START_DATE = "2020-01-01"
PIPELINE_LOG_ROOT = PORTFOLIO_ROOT / "pipeline_logs"
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class Step:
    name: str
    project_root: Path
    arguments: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从数据更新到调仓报告，运行完整 A 股投资系统",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不执行，用于检查流程",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="不询问，继续本周第一个未完成步骤",
    )
    mode.add_argument(
        "--restart",
        action="store_true",
        help="不询问，归档本周旧记录并从第1步重新运行",
    )
    return parser.parse_args()


def validate_layout() -> None:
    required = {
        "截面多因子模型": CROSS_SECTION_ROOT,
        "市场预测模型": MARKET_ROOT,
        "组合风控层": PORTFOLIO_ROOT,
    }
    missing = [name for name, path in required.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"缺少项目目录: {', '.join(missing)}")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"当前 Python 为 {sys.version.split()[0]}；请先 conda activate qf_clean"
        )


def run_step(step: Step, number: int, total: int, dry_run: bool) -> None:
    command = [sys.executable, *step.arguments]
    print("\n" + "=" * 78, flush=True)
    print(f"[{number}/{total}] {step.name}", flush=True)
    print(f"目录: {step.project_root}", flush=True)
    print(f"> {subprocess.list2cmdline(command)}", flush=True)
    print("=" * 78, flush=True)
    if not dry_run:
        subprocess.run(command, cwd=step.project_root, check=True)


def latest_cross_section_date() -> str:
    close_path = (
        CROSS_SECTION_ROOT
        / "02因子库"
        / "processed_data"
        / "market_data"
        / "close.parquet"
    )
    if not close_path.is_file():
        raise FileNotFoundError(f"找不到行情宽表: {close_path}")
    dates = pd.read_parquet(close_path, columns=["time"])["time"]
    dates = pd.to_datetime(dates, errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"行情宽表没有有效日期: {close_path}")
    return dates.max().strftime("%Y-%m-%d")


def local_now() -> datetime:
    return datetime.now(SHANGHAI_TZ)


def iso_timestamp() -> str:
    return local_now().isoformat(timespec="seconds")


def current_week_key() -> str:
    calendar = local_now().isocalendar()
    return f"{calendar.year}-W{calendar.week:02d}"


def checkpoint_path(week_key: str) -> Path:
    return PIPELINE_LOG_ROOT / week_key / "current.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "full_pipeline_checkpoint_v1":
        raise ValueError(f"无法识别的流水线检查点: {path}")
    return state


def new_checkpoint(week_key: str, step_names: list[str]) -> dict:
    now = local_now()
    run_tag = now.strftime("%Y%m%d_%H%M%S")
    timestamp = now.isoformat(timespec="seconds")
    return {
        "schema_version": "full_pipeline_checkpoint_v1",
        "week_key": week_key,
        "run_tag": run_tag,
        "status": "pending",
        "created_at": timestamp,
        "updated_at": timestamp,
        "end_date": None,
        "last_error": None,
        "steps": [
            {
                "number": number,
                "name": name,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "command": None,
                "error": None,
            }
            for number, name in enumerate(step_names, start=1)
        ],
    }


def archive_checkpoint(path: Path, state: dict) -> Path:
    history_dir = path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    run_tag = str(state.get("run_tag", "unknown"))
    status = str(state.get("status", "unknown"))
    archive_path = history_dir / f"{run_tag}_{status}.json"
    counter = 1
    while archive_path.exists():
        archive_path = history_dir / f"{run_tag}_{status}_{counter}.json"
        counter += 1
    shutil.copy2(path, archive_path)
    return archive_path


def completed_step_count(state: dict) -> int:
    return sum(step.get("status") == "completed" for step in state["steps"])


def first_incomplete_name(state: dict) -> str | None:
    for step in state["steps"]:
        if step.get("status") != "completed":
            return str(step["name"])
    return None


def reconcile_steps(state: dict, step_names: list[str]) -> bool:
    old_names = [str(step["name"]) for step in state["steps"]]
    if old_names == step_names:
        return False
    old_by_name = {str(step["name"]): step for step in state["steps"]}
    state["steps"] = []
    for number, name in enumerate(step_names, start=1):
        previous = old_by_name.get(name, {})
        state["steps"].append(
            {
                "number": number,
                "name": name,
                "status": previous.get("status", "pending"),
                "started_at": previous.get("started_at"),
                "completed_at": previous.get("completed_at"),
                "command": previous.get("command"),
                "error": previous.get("error"),
            }
        )
    state["updated_at"] = iso_timestamp()
    return True


def choose_checkpoint(
    args: argparse.Namespace,
    path: Path,
    existing: dict | None,
    week_key: str,
    step_names: list[str],
) -> dict | None:
    if existing is None:
        state = new_checkpoint(week_key, step_names)
        write_json_atomic(path, state)
        print(f"本周没有旧记录，从第1步开始。\n检查点: {path}", flush=True)
        return state

    definition_changed = [step["name"] for step in existing["steps"]] != step_names
    completed = completed_step_count(existing)
    next_name = first_incomplete_name(existing)
    print("\n发现本周流水线记录", flush=True)
    print(f"检查点: {path}", flush=True)
    print(f"状态: {existing.get('status')} | 已完成: {completed}/{len(existing['steps'])}", flush=True)
    if next_name:
        print(f"下一步: {next_name}", flush=True)
    if definition_changed:
        print("注意：步骤结构与记录创建时不同，建议从第1步重新运行。", flush=True)

    if args.restart:
        choice = "restart"
    elif args.resume:
        choice = "resume"
    elif existing.get("status") == "completed":
        response = input("本周流程已经完成。[1] 退出（默认）  [2] 从第1步重跑：").strip()
        choice = "restart" if response == "2" else "exit"
    else:
        response = input(
            "请选择：[1] 从未完成步骤继续（默认）  [2] 从第1步重跑  [3] 退出："
        ).strip()
        choice = {"2": "restart", "3": "exit"}.get(response, "resume")

    if choice == "exit":
        print("未运行任何步骤。", flush=True)
        return None
    if choice == "restart":
        archive = archive_checkpoint(path, existing)
        state = new_checkpoint(week_key, step_names)
        write_json_atomic(path, state)
        print(f"旧记录已归档: {archive}", flush=True)
        print("将从第1步重新运行。", flush=True)
        return state

    if definition_changed:
        reconcile_steps(existing, step_names)
        print("按步骤名称合并了新旧结构；已完成的同名步骤继续保留。", flush=True)
    existing["status"] = "pending"
    existing["last_error"] = None
    existing["updated_at"] = iso_timestamp()
    write_json_atomic(path, existing)
    return existing


def build_steps(run_tag: str, end_date: str) -> list[Step]:
    py = lambda *parts: tuple(str(part) for part in parts)
    steps: list[Step] = [
        Step(
            "截面模型：每周增量更新数据",
            CROSS_SECTION_ROOT,
            py("01数据/tushare_data_main.py", "--weekly"),
        ),
        Step(
            "截面模型：重建行情、财务宽表和全部因子",
            CROSS_SECTION_ROOT,
            py("02因子库/update_all.py"),
        ),
    ]

    for horizon, experiment in CROSS_EXPERIMENTS.items():
        steps.append(
            Step(
                f"截面模型：增量训练 {horizon} 日模型",
                CROSS_SECTION_ROOT,
                py(
                    "03模型训练层/main_train_v2.py",
                    "--config",
                    f"configs/production/horizon{horizon}_profit20_tuned_config.yaml",
                    "--exp-id",
                    experiment,
                    "--start-date",
                    CROSS_TRAINING_START_DATE,
                    "--end-date",
                    end_date,
                    "--freeze",
                    "-y",
                ),
            )
        )

    steps.extend(
        [
            Step(
                "截面模型：融合 5/20/60 日预测",
                CROSS_SECTION_ROOT,
                py(
                    "03模型训练层/fuse_predictions.py",
                    "--exps",
                    CROSS_EXPERIMENTS[5],
                    CROSS_EXPERIMENTS[20],
                    CROSS_EXPERIMENTS[60],
                    "--base-idx",
                    "1",
                    "--output-exp",
                    FUSION_EXPERIMENT,
                ),
            ),
            Step(
                "截面模型：Alphalens 验收",
                CROSS_SECTION_ROOT,
                py(
                    "04回测层/alphalens_analysis.py",
                    "--exp-id",
                    FUSION_EXPERIMENT,
                    "--use-smooth",
                ),
            ),
            Step(
                "截面模型：月度 Backtrader 回测",
                CROSS_SECTION_ROOT,
                py(
                    "04回测层/backtrader.eval.py",
                    "--exp-id",
                    FUSION_EXPERIMENT,
                    "--use-smooth",
                ),
            ),
            Step(
                "截面模型：周度 Backtrader 回测",
                CROSS_SECTION_ROOT,
                py(
                    "04回测层/backtrader.weekly.eval.py",
                    "--exp-id",
                    FUSION_EXPERIMENT,
                    "--use-smooth",
                ),
            ),
            Step(
                "截面模型：发布最新 Alpha",
                CROSS_SECTION_ROOT,
                py(
                    "05输出层/publish_alpha.py",
                    "--exp-id",
                    FUSION_EXPERIMENT,
                    "--release-id",
                    f"alpha_tushare_{run_tag}",
                ),
            ),
            Step(
                "市场模型：增量更新数据",
                MARKET_ROOT,
                py("01数据/download_data.py"),
            ),
            Step(
                "市场模型：构建特征",
                MARKET_ROOT,
                py("02特征层/build_features.py"),
            ),
            Step(
                "市场模型：验证特征",
                MARKET_ROOT,
                py("02特征层/validate_features.py"),
            ),
            Step(
                "市场模型：训练 Ridge 和 CNN-GRU",
                MARKET_ROOT,
                py(
                    "03模型训练层/run_models.py",
                    "--model",
                    "all",
                    "--rebuild-data",
                ),
            ),
        ]
    )

    market_run = f"production_904500_{run_tag}"
    steps.extend(
        [
            Step(
                "市场模型：90/45/0 经济意义回测",
                MARKET_ROOT,
                py(
                    "04经济意义与回测层/run_economic_value.py",
                    "--strategy-config",
                    "04经济意义与回测层/strategy_configs/bull90_neutral45_bear0.yaml",
                    "--run-id",
                    market_run,
                    "--model",
                    "all",
                    "--overwrite",
                ),
            ),
            Step(
                "市场模型：验证回测",
                MARKET_ROOT,
                py(
                    "04经济意义与回测层/validate_backtest.py",
                    "--run-id",
                    market_run,
                ),
            ),
            Step(
                "市场模型：发布 CNN-GRU 信号",
                MARKET_ROOT,
                py(
                    "05输出层/publish_market_signal.py",
                    "--model",
                    "cnn_gru",
                    "--run-id",
                    market_run,
                    "--release-id",
                    f"market_cnn_gru_{run_tag}",
                ),
            ),
            Step(
                "市场模型：发布 Ridge 信号并设为 current",
                MARKET_ROOT,
                py(
                    "05输出层/publish_market_signal.py",
                    "--model",
                    "ridge",
                    "--run-id",
                    market_run,
                    "--release-id",
                    f"market_ridge_{run_tag}",
                    "--set-current",
                ),
            ),
            Step(
                "组合系统：同步上游并构建日度决策输入",
                PORTFOLIO_ROOT,
                py(
                    "01组合决策输入层/run_pipeline.py",
                    "--release-id",
                    f"decision_daily_{run_tag}",
                ),
            ),
        ]
    )

    for frequency in ("weekly", "monthly"):
        steps.append(
            Step(
                f"组合系统：生成{frequency}优化权重",
                PORTFOLIO_ROOT,
                py(
                    "02组合优化层/optimize.py",
                    "--frequency",
                    frequency,
                    "--release-id",
                    f"portfolio_{frequency}_{run_tag}",
                ),
            )
        )
    for frequency in ("weekly", "monthly"):
        steps.append(
            Step(
                f"组合系统：生成{frequency}风险记录",
                PORTFOLIO_ROOT,
                py(
                    "03组合风控层/assess_risk.py",
                    "--frequency",
                    frequency,
                    "--release-id",
                    f"risk_{frequency}_{run_tag}",
                ),
            )
        )
    for frequency in ("weekly", "monthly"):
        steps.append(
            Step(
                f"组合系统：运行{frequency}账户回测",
                PORTFOLIO_ROOT,
                py(
                    "04组合回测层/backtrader.eval.py",
                    "--frequency",
                    frequency,
                    "--report-id",
                    f"portfolio_{frequency}_{run_tag}",
                    "--overwrite",
                ),
            )
        )
    for frequency in ("weekly", "monthly"):
        steps.append(
            Step(
                f"组合系统：生成{frequency}调仓报告",
                PORTFOLIO_ROOT,
                py(
                    "05调仓输出层/generate_rebalance_report.py",
                    "--frequency",
                    frequency,
                    "--report-id",
                    f"rebalance_{frequency}_{run_tag}",
                    "--overwrite",
                ),
            )
        )
    return steps


def mark_step_started(state: dict, state_path: Path, step: Step, index: int) -> None:
    entry = state["steps"][index]
    entry["status"] = "running"
    entry["started_at"] = iso_timestamp()
    entry["completed_at"] = None
    entry["command"] = subprocess.list2cmdline([sys.executable, *step.arguments])
    entry["error"] = None
    state["status"] = "running"
    state["last_error"] = None
    state["updated_at"] = iso_timestamp()
    write_json_atomic(state_path, state)


def mark_step_completed(state: dict, state_path: Path, index: int) -> None:
    entry = state["steps"][index]
    entry["status"] = "completed"
    entry["completed_at"] = iso_timestamp()
    state["status"] = "running"
    state["updated_at"] = iso_timestamp()
    write_json_atomic(state_path, state)


def mark_step_failed(
    state: dict,
    state_path: Path,
    index: int,
    status: str,
    error: str,
) -> None:
    entry = state["steps"][index]
    entry["status"] = status
    entry["error"] = error
    state["status"] = status
    state["last_error"] = {
        "step_number": index + 1,
        "step_name": entry["name"],
        "error": error,
        "recorded_at": iso_timestamp(),
    }
    state["updated_at"] = iso_timestamp()
    write_json_atomic(state_path, state)


def run_dry_run() -> int:
    run_tag = local_now().strftime("%Y%m%d_%H%M%S")
    end_date = latest_cross_section_date()
    steps = build_steps(run_tag, end_date)
    for number, step in enumerate(steps, start=1):
        run_step(step, number, len(steps), True)
    print("\n干跑完成：没有执行命令，也没有修改检查点。", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    validate_layout()
    if args.dry_run:
        return run_dry_run()

    week_key = current_week_key()
    state_path = checkpoint_path(week_key)
    existing = load_checkpoint(state_path)
    reference_tag = (
        str(existing["run_tag"])
        if existing is not None
        else local_now().strftime("%Y%m%d_%H%M%S")
    )
    reference_date = (
        str(existing.get("end_date") or "LATEST_DATE_PENDING")
        if existing is not None
        else "LATEST_DATE_PENDING"
    )
    reference_steps = build_steps(reference_tag, reference_date)
    step_names = [step.name for step in reference_steps]
    state = choose_checkpoint(args, state_path, existing, week_key, step_names)
    if state is None:
        return 0

    run_tag = str(state["run_tag"])
    total = len(state["steps"])
    for index in range(total):
        if state["steps"][index].get("status") == "completed":
            print(
                f"[检查点跳过 {index + 1}/{total}] {state['steps'][index]['name']}",
                flush=True,
            )
            continue

        if index >= 2 and not state.get("end_date"):
            state["end_date"] = latest_cross_section_date()
            state["updated_at"] = iso_timestamp()
            write_json_atomic(state_path, state)
        end_date = str(state.get("end_date") or "LATEST_DATE_PENDING")
        steps = build_steps(run_tag, end_date)
        step = steps[index]
        mark_step_started(state, state_path, step, index)
        try:
            run_step(step, index + 1, total, False)
        except KeyboardInterrupt:
            mark_step_failed(state, state_path, index, "interrupted", "KeyboardInterrupt")
            print(
                f"\n已记录中断位置。下次可从第 {index + 1} 步继续。\n检查点: {state_path}",
                flush=True,
            )
            return 130
        except Exception as exc:
            mark_step_failed(state, state_path, index, "failed", repr(exc))
            print(
                f"\n第 {index + 1} 步失败，已保存检查点。\n检查点: {state_path}",
                flush=True,
            )
            raise
        mark_step_completed(state, state_path, index)

        if index == 1 and not state.get("end_date"):
            state["end_date"] = latest_cross_section_date()
            state["updated_at"] = iso_timestamp()
            write_json_atomic(state_path, state)

    state["status"] = "completed"
    state["completed_at"] = iso_timestamp()
    state["updated_at"] = iso_timestamp()
    write_json_atomic(state_path, state)
    print("\n" + "=" * 78)
    print("完整流程运行成功")
    print(f"run tag: {run_tag}")
    print(f"截面数据截止日: {state['end_date']}")
    print(f"检查点: {state_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
