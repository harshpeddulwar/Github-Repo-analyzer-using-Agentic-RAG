"""
Single place that loads config.yaml.

* Resolves the file relative to the project root, so it works no matter which
  directory you launch from (streamlit, python test.py, pytest, ...).
* Merges user values over DEFAULTS, so a missing key never crashes the app.
* Nothing is read at import time -- callers ask for the config explicitly and
  can pass a dict in (handy for tests).
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULTS: Dict[str, Any] = {
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0, "max_retries": 4},
    "embedding": {"model": "jinaai/jina-embeddings-v2-base-code", "batch_size": 8},
    "vector_db": {
        "collection_name": "rag_repo_collection",
        "vector_size": 768,
        "url": "http://localhost:6333",
        "path": None,  # set to e.g. "qdrant_storage" for embedded (no server) mode
    },
    "retrieval": {
        "candidate_limit": 20,
        "final_result_count": 4,
        "max_chunks_per_file": 2,
        "min_rerank_score": -2.0,
        "max_context_tokens": 3000,
    },
    "orchestrator": {
        "enable_planner": True,
        "enable_verifier": True,   # LLM verifier. The deterministic citation check always runs.
        "max_subquestions": 3,
        "max_revisions": 1,        # HARD cap on revise->verify rounds. Never loops beyond this.
        "history_turns": 4,
        "route_budgets": {"lookup": 3, "explain_flow": 5, "architecture": 6},
    },
    "logging": {
        "level": "INFO",
        "log_file": "logs/pipeline.log",
        "max_bytes": 5242880,
        "backup_count": 3,
        "enable_console": True,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    user_cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    merged = _merge(DEFAULTS, user_cfg)

    # Automatically set up logger with loaded config
    try:
        from src.logger import setup_logging
        log_cfg = merged.get("logging", {})
        setup_logging(
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("log_file"),
            max_bytes=log_cfg.get("max_bytes", 5242880),
            backup_count=log_cfg.get("backup_count", 3),
            enable_console=log_cfg.get("enable_console", True),
        )
    except Exception:
        pass

    return merged
