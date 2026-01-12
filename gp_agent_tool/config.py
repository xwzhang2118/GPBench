"""
Minimal config loader for gwas-llm-judge.

The config file should be located at:
    /home/common/hwluo/project/gwas-llm-judge/config/config.json

and contain at least the sections:
    - "llm"
    - "codegen_llm"
    - "multimodal_llm"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from logging_utils import get_logger


logger = get_logger(__name__)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _config_path() -> Path:
    # 固定为项目下的 config/config.json，避免依赖旧项目路径
    return Path(__file__).resolve().parent / "config" / "config.json"


def _load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = _config_path()
    try:
        with path.open("r", encoding="utf-8") as f:
            _CONFIG_CACHE = json.load(f)
    except FileNotFoundError:
        logger.warning("Config file not found: %s", path)
        _CONFIG_CACHE = {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load config %s: %s", path, exc)
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_llm_config() -> Dict[str, Any]:
    cfg = _load_config()
    return dict(cfg.get("llm", {}))


def get_codegen_llm_config() -> Dict[str, Any]:
    cfg = _load_config()
    return dict(cfg.get("codegen_llm", {}))


def get_multimodal_llm_config() -> Dict[str, Any]:
    cfg = _load_config()
    return dict(cfg.get("multimodal_llm", {}))



