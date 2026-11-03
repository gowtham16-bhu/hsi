"""Config loading with validation and a stable hash recorded with every run (constitution P6)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class PluginSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plugin: str
    options: dict[str, Any] = {}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def config_hash(cfg: BaseModel | dict[str, Any]) -> str:
    data = cfg.model_dump(mode="json") if isinstance(cfg, BaseModel) else cfg
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
