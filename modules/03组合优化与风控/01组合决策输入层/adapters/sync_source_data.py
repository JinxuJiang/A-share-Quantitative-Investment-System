from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

LAYER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAYER_ROOT.parent
if str(LAYER_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER_ROOT))

from adapters.common import read_json, read_yaml, resolve_from


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步组合决策输入层所需的全市场上游数据")
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def copy_atomic_with_hash(source: Path, destination: Path) -> Dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as reader, temp_path.open("wb") as writer:
            while chunk := reader.read(8 * 1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return {
        "source": str(source),
        "destination": str(destination),
        "size": size,
        "sha256": digest.hexdigest(),
        "source_mtime": datetime.fromtimestamp(source.stat().st_mtime).astimezone().isoformat(),
    }


def resolve_release(export_root: Path, expected_current_schema: str, data_name: str) -> Dict[str, Path]:
    current_path = export_root / "current.json"
    current = read_json(current_path)
    if current.get("schema_version") != expected_current_schema:
        raise ValueError(f"current schema不符: {current_path}")
    manifest_path = export_root / str(current["manifest"])
    manifest = read_json(manifest_path)
    data_path = manifest_path.parent / data_name
    return {"current": current_path, "manifest": manifest_path, "data": data_path}


def main() -> None:
    args = parse_args()
    config_path = resolve_from(LAYER_ROOT, args.config)
    config = read_yaml(config_path)
    cross_root = resolve_from(PROJECT_ROOT, config["paths"]["cross_section_project"])
    market_root = resolve_from(PROJECT_ROOT, config["paths"]["market_project"])
    local_root = resolve_from(LAYER_ROOT, config["paths"]["local_data_root"])

    alpha_release = resolve_release(
        cross_root / "05输出层" / "exports",
        "stock_alpha_current_v1",
        "stock_alpha.parquet",
    )
    market_release = resolve_release(
        market_root / "05输出层" / "exports",
        "market_signal_current_v1",
        "market_signal.parquet",
    )

    copy_plan = []
    for key, source in alpha_release.items():
        name = "current.json" if key == "current" else "manifest.json" if key == "manifest" else "stock_alpha.parquet"
        copy_plan.append((source, local_root / "source_alpha" / name))
    for key, source in market_release.items():
        name = "current.json" if key == "current" else "manifest.json" if key == "manifest" else "market_signal.parquet"
        copy_plan.append((source, local_root / "source_market_signal" / name))

    market_source = cross_root / "02因子库" / "processed_data" / "market_data"
    for field in config["sync"]["market_fields"]:
        copy_plan.append((market_source / f"{field}.parquet", local_root / "market_data" / f"{field}.parquet"))

    tushare_root = cross_root / "01数据" / "data" / "tushare_data"
    for relative in config["sync"]["status_files"]:
        source = tushare_root / relative
        copy_plan.append((source, local_root / "status" / Path(relative).name))
    for relative in config["sync"]["metadata_files"]:
        source = tushare_root / relative
        copy_plan.append((source, local_root / "metadata" / Path(relative).name))

    records = []
    for index, (source, destination) in enumerate(copy_plan, start=1):
        print(f"[{index}/{len(copy_plan)}] {source.name} -> {destination}")
        records.append(copy_atomic_with_hash(source, destination))

    manifest = {
        "schema_version": "decision_source_data_v1",
        "synced_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cross_section_project": str(cross_root),
        "market_project": str(market_root),
        "alpha_release_id": read_json(alpha_release["manifest"])["release_id"],
        "market_release_id": read_json(market_release["manifest"])["release_id"],
        "file_count": len(records),
        "files": records,
    }
    manifest_path = local_root / "source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = manifest_path.with_name(".source_manifest.json.tmp")
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    print(f"同步完成: {manifest_path}")


if __name__ == "__main__":
    main()
