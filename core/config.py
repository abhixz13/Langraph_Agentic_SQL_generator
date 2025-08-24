"""
core/config.py — Central configuration & policy knobs for the SQL Assistant.

This config module loads env vars, validates them, and exposes 
- Loads environment variables (.env) and validates types.
- Exposes a typed Settings object.
- Provides small helper dicts for seeding graph state and tools.
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}

def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_list(name: str, default: List[str], sep: str = ",") -> List[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [t.strip() for t in val.split(sep) if t.strip()]

class Settings(BaseModel):
    # Provider / LLM
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_org_id: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_ORG_ID"))
    openai_base_url: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))

    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = Field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.1))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    # Database
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    db_dialect: Optional[Literal["postgres", "mysql", "sqlite"]] = Field(
        default_factory=lambda: os.getenv("DB_DIALECT")
    )
    db_readonly: bool = Field(default_factory=lambda: _get_bool("DB_READONLY", True))
    db_pool_size: int = Field(default_factory=lambda: _get_int("DB_POOL_SIZE", 5))
    db_max_overflow: int = Field(default_factory=lambda: _get_int("DB_MAX_OVERFLOW", 5))
    db_connect_timeout_sec: int = Field(default_factory=lambda: _get_int("DB_CONNECT_TIMEOUT_SEC", 10))

    # Schema / semantic
    semantic_schema_path: Optional[str] = Field(default_factory=lambda: os.getenv("SEMANTIC_SCHEMA_PATH"))
    schema_cache_path: str = Field(default_factory=lambda: os.getenv("SCHEMA_CACHE_PATH", "./data/schema_cache.json"))

    # Policy / safety (graph-level)
    gen_k: int = Field(default_factory=lambda: _get_int("GEN_K", 3))
    max_loops: int = Field(default_factory=lambda: _get_int("MAX_LOOPS", 2))
    row_cap: int = Field(default_factory=lambda: _get_int("ROW_CAP", 5000))
    execute_timeout_ms: int = Field(default_factory=lambda: _get_int("EXECUTE_TIMEOUT_MS", 20000))
    require_validate_before_execute: bool = Field(
        default_factory=lambda: _get_bool("REQUIRE_VALIDATE_BEFORE_EXECUTE", True)
    )
    blocked_keywords: List[str] = Field(
        default_factory=lambda: _get_list(
            "BLOCKED_KEYWORDS", ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"]
        )
    )

    # Intent interpretation policy
    intent_min_confidence: float = Field(default_factory=lambda: _get_float("INTENT_MIN_CONFIDENCE", 0.3))
    intent_default_confidence: float = Field(default_factory=lambda: _get_float("INTENT_DEFAULT_CONFIDENCE", 0.5))
    intent_fallback_confidence: float = Field(default_factory=lambda: _get_float("INTENT_FALLBACK_CONFIDENCE", 0.1))
    intent_default_action: str = Field(default_factory=lambda: os.getenv("INTENT_DEFAULT_ACTION", "SELECT"))
    intent_valid_actions: List[str] = Field(
        default_factory=lambda: _get_list(
            "INTENT_VALID_ACTIONS", ["SELECT", "COUNT", "AGGREGATE", "SEARCH", "COMPARE"]
        )
    )
    intent_default_complexity: str = Field(default_factory=lambda: os.getenv("INTENT_DEFAULT_COMPLEXITY", "simple"))
    intent_valid_complexities: List[str] = Field(
        default_factory=lambda: _get_list(
            "INTENT_VALID_COMPLEXITIES", ["simple", "moderate", "complex"]
        )
    )
    
    # Schema prefiltering limits
    intent_max_tables: int = Field(default_factory=lambda: _get_int("INTENT_MAX_TABLES", 8))
    intent_max_columns_per_table: int = Field(default_factory=lambda: _get_int("INTENT_MAX_COLUMNS_PER_TABLE", 10))

    # Checkpointer / memory
    checkpointer_backend: Literal["file", "sqlite", "redis"] = Field(
        default_factory=lambda: os.getenv("CHECKPOINTER_BACKEND", "file")
    )
    checkpointer_path: str = Field(default_factory=lambda: os.getenv("CHECKPOINTER_PATH", "./data/checkpoints"))
    redis_url: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_URL"))

    few_shot_examples_dir: str = Field(default_factory=lambda: os.getenv("FEW_SHOT_EXAMPLES_DIR", "./data/examples"))
    golden_queries_path: str = Field(default_factory=lambda: os.getenv("GOLDEN_QUERIES_PATH", "./data/examples/golden.jsonl"))

    # Observability
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    env: str = Field(default_factory=lambda: os.getenv("ENV", "development"))

    # UI
    gradio_server_name: str = Field(default_factory=lambda: os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
    gradio_server_port: int = Field(default_factory=lambda: _get_int("GRADIO_SERVER_PORT", 7860))

    class Config:
        extra = "ignore"

def load_settings() -> Settings:
    """
    Load and validate settings from environment variables.
    Raises ValidationError if required fields are missing/invalid.
    """
    s = Settings()
    # Minimal hard checks:
    if not s.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if not s.database_url:
        raise ValueError("DATABASE_URL is required")
    return s

def policy_from_settings(s: Settings) -> dict:
    """
    Convert Settings to a small policy dict used to seed the graph state.
    """
    return {
        "gen_k": s.gen_k,
        "max_loops": s.max_loops,
        "row_cap": s.row_cap,
        "execute_timeout_ms": s.execute_timeout_ms,
        "require_validate_before_execute": s.require_validate_before_execute,
        "blocked_keywords": s.blocked_keywords,
        "dialect": s.db_dialect,  # may be None; downstream can auto-detect from DATABASE_URL
        
        # Intent interpretation policy
        "intent": {
            "min_confidence": s.intent_min_confidence,
            "default_confidence": s.intent_default_confidence,
            "fallback_confidence": s.intent_fallback_confidence,
            "default_action": s.intent_default_action,
            "valid_actions": s.intent_valid_actions,
            "default_complexity": s.intent_default_complexity,
            "valid_complexities": s.intent_valid_complexities,
            "max_tables": s.intent_max_tables,
            "max_columns_per_table": s.intent_max_columns_per_table,
        }
    }

def db_config_from_settings(s: Settings) -> dict:
    """
    DB/runtime knobs for tool layer (safe execute, pooling, read-only flags).
    """
    return {
        "database_url": s.database_url,
        "readonly": s.db_readonly,
        "pool_size": s.db_pool_size,
        "max_overflow": s.db_max_overflow,
        "connect_timeout_sec": s.db_connect_timeout_sec,
        "dialect": s.db_dialect,
    }

def llm_config_from_settings(s: Settings) -> dict:
    """
    LLM/provider knobs for agent prompts/tool-calling.
    """
    return {
        "model": s.llm_model,
        "temperature": s.llm_temperature,
        "embedding_model": s.embedding_model,
        "openai": {
            "api_key": s.openai_api_key,
            "org_id": s.openai_org_id,
            "base_url": s.openai_base_url,
        },
    }

def checkpointer_config_from_settings(s: Settings) -> dict:
    """
    Checkpointer backend selection.
    """
    return {
        "backend": s.checkpointer_backend,
        "path": s.checkpointer_path,
        "redis_url": s.redis_url,
    }
