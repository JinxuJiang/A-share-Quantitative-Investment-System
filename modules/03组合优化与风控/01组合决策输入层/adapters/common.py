from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON根节点必须是对象: {path}")
    return value


def read_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML根节点必须是对象: {path}")
    return value


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()

