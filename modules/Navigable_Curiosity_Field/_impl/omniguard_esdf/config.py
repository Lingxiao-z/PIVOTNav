from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def dump_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
