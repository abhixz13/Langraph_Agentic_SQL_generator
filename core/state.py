"""
core/state.py
--------------
Typed, merge-safe application state for the SQL Assistant (LangGraph).

This module is the machine-enforced mirror of docs/AgentCards.md:
- Defines the canonical state keys and their types
- Encodes merge rules (REPLACE / APPEND / INCREMENT / MERGE_DICT)
- Provides an initializer that seeds policy knobs (gen_k, max_loops, dialect, etc.)
- Provides a patch applier that nodes/subgraphs can use to safely update state
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, TYPE_CHECKING

from pydantic import BaseModel, Field, RootModel, ValidationError

if TYPE_CHECKING:  # Optional import to avoid circular dependency at runtime
    from .config import Settings


# ----------------------------
# Merge modes & specification
# ----------------------------

class MergeMode(str, Enum):
    REPLACE = "REPLACE"      # overwrite with new value
    APPEND = "APPEND"        # list-append (extend)
    INCREMENT = "INCREMENT"  # add numeric delta (typically +1)
    MERGE_DICT = "MERGE_DICT"  # shallow dict.update()


# Merge spec per key; unknown keys are rejected by apply_patch()
MERGE_SPEC: Dict[str, MergeMode] = {
    # Inputs / control
    "user_query": MergeMode.REPLACE,
    "dialect": MergeMode.REPLACE,
    "gen_k": MergeMode.REPLACE,
    "max_loops": MergeMode.REPLACE,
    "loop_count": MergeMode.INCREMENT,

    # Schema
    "db_schema": MergeMode.REPLACE,
    "schema_context": MergeMode.REPLACE,

    # Interpret / Plan
    "intent_json": MergeMode.REPLACE,
    "plan": MergeMode.REPLACE,
    "plan_ok": MergeMode.REPLACE,
    "plan_confidence": MergeMode.REPLACE,
    "ambiguity": MergeMode.REPLACE,
    "clarifying_question": MergeMode.REPLACE,
    "clarifying_answer": MergeMode.REPLACE,

    # Generate
    "few_shots": MergeMode.REPLACE,
    "sql_candidates": MergeMode.APPEND,
    "sql": MergeMode.REPLACE,
    "gen_attempts": MergeMode.INCREMENT,

    # Validate
    "validation_ok": MergeMode.REPLACE,
    "val_reasons": MergeMode.APPEND,
    "val_signals": MergeMode.REPLACE,  # could be MERGE_DICT if you want to accumulate
    "val_attempts": MergeMode.INCREMENT,

    # Execute
    "exec_preview": MergeMode.REPLACE,
    "result_sample": MergeMode.REPLACE,
    "exec_rows": MergeMode.REPLACE,
    "error": MergeMode.REPLACE,
    "status": MergeMode.REPLACE,
    "error_message": MergeMode.REPLACE,

    # Observability
    "metrics": MergeMode.MERGE_DICT,
    
    # Policy configuration
    "policy": MergeMode.MERGE_DICT,
}


# ----------------------------
# Typed state model
# ----------------------------

class Metrics(TypedDict, total=False):
    latency_ms: float
    tokens: int
    model: str
    # add per-node metrics as needed (e.g., {"validator": {...}})


class AppState(BaseModel):
    """
    Canonical state object used by the graph. Mirrors docs/AgentCards.md.

    Merge rules are enforced by apply_patch() using MERGE_SPEC.
    """
    # Inputs / control
    user_query: str = ""
    dialect: Optional[str] = None
    gen_k: int = 3
    max_loops: int = 2
    loop_count: int = 0

    # Schema
    db_schema: Optional[Dict[str, Any]] = None
    schema_context: Optional[Dict[str, Any]] = None

    # Interpret / Plan
    intent_json: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    plan_ok: Optional[bool] = None
    plan_confidence: Optional[float] = None
    ambiguity: Optional[bool] = None
    clarifying_question: Optional[str] = None
    clarifying_answer: Optional[str] = None

    # Generate
    few_shots: Optional[List[Dict[str, Any]]] = None
    sql_candidates: List[str] = Field(default_factory=list)
    sql: Optional[str] = None
    gen_attempts: int = 0

    # Validate
    validation_ok: Optional[bool] = None
    val_reasons: List[str] = Field(default_factory=list)
    val_signals: Optional[Dict[str, Any]] = None
    val_attempts: int = 0

    # Execute
    exec_preview: Optional[Dict[str, Any]] = None
    result_sample: Optional[List[Dict[str, Any]]] = None
    exec_rows: Optional[int] = None
    error: Optional[str] = None
    status: Optional[str] = None
    error_message: Optional[str] = None

    # Observability
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Policy configuration
    policy: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"  # allow extra fields for workflow flexibility


# ----------------------------
# Helpers: init, patch, utils
# ----------------------------

def initial_state(
    *,
    user_query: str = "",
    dialect: Optional[str] = None,
    gen_k: int = 3,
    max_loops: int = 2,
    policy: Optional[Dict[str, Any]] = None,
) -> AppState:
    """
    Create a fresh AppState with safe defaults.
    Typically you pass values from core.config.policy_from_settings(...).
    """
    if gen_k < 1:
        raise ValueError("gen_k must be >= 1")
    if max_loops < 0:
        raise ValueError("max_loops must be >= 0")

    return AppState(
        user_query=user_query,
        dialect=dialect,
        gen_k=gen_k,
        max_loops=max_loops,
        loop_count=0,
        policy=policy,
        # everything else defaults
    )


def initial_state_from_settings(
    settings: "Settings",
    *,
    user_query: str = "",
) -> AppState:
    """
    Convenience initializer if you have a Settings object from core.config.load_settings().
    """
    # Import here to avoid hard import at module import time (optional).
    try:
        from .config import policy_from_settings
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Unable to import policy_from_settings from core.config") from exc

    policy = policy_from_settings(settings)
    return initial_state(
        user_query=user_query,
        dialect=policy.get("dialect"),
        gen_k=policy.get("gen_k", 3),
        max_loops=policy.get("max_loops", 2),
        policy=policy,  # Include the full policy configuration
    )


def apply_patch(state: AppState, patch: Dict[str, Any]) -> AppState:
    """
    Apply a node's patch to state according to MERGE_SPEC, returning a NEW AppState.

    Rules:
    - Unknown keys → ValueError
    - REPLACE → overwrite
    - APPEND → extend list; patch value must be a list (we don't auto-wrap to avoid silent mistakes)
    - INCREMENT → add integer delta (e.g., +1)
    - MERGE_DICT → shallow dict.update()
    - Types are validated by Pydantic when constructing the new AppState
    """
    unknown = [k for k in patch.keys() if k not in MERGE_SPEC]
    if unknown:
        raise KeyError(f"Patch contains unknown keys not in state contract: {unknown}")

    # Start from current dict
    data = state.model_dump()

    for key, value in patch.items():
        mode = MERGE_SPEC[key]

        if mode == MergeMode.REPLACE:
            data[key] = value

        elif mode == MergeMode.APPEND:
            existing = data.get(key)
            if existing is None:
                existing = []
            if not isinstance(existing, list):
                raise TypeError(f"Key '{key}' is not a list; cannot APPEND")
            if not isinstance(value, list):
                raise TypeError(f"Patch for '{key}' must be a list when using APPEND")
            existing.extend(value)
            data[key] = existing

        elif mode == MergeMode.INCREMENT:
            existing = data.get(key, 0)
            if not isinstance(existing, int):
                raise TypeError(f"Key '{key}' is not an int; cannot INCREMENT")
            if not isinstance(value, int):
                raise TypeError(f"Patch for '{key}' must be an int when using INCREMENT")
            data[key] = existing + value

        elif mode == MergeMode.MERGE_DICT:
            existing = data.get(key) or {}
            if not isinstance(existing, dict):
                raise TypeError(f"Key '{key}' is not a dict; cannot MERGE_DICT")
            if not isinstance(value, dict):
                raise TypeError(f"Patch for '{key}' must be a dict when using MERGE_DICT")
            merged = dict(existing)
            merged.update(value)
            data[key] = merged

        else:  # pragma: no cover
            raise ValueError(f"Unhandled merge mode for key '{key}': {mode}")

    # Validate types & return new instance
    try:
        return AppState(**data)
    except ValidationError as ve:
        # Make errors easier to read for developers
        raise ve


def can_execute(state: AppState) -> bool:
    """
    Helper: graph-level guard. Execution is allowed only if Validator set validation_ok=True.
    """
    return bool(state.validation_ok)


def as_dict(state: AppState) -> Dict[str, Any]:
    """
    Serialize state to a plain dict for checkpointing or UI.
    """
    return state.model_dump()


def create_error_response(error_message: str, status: str = "ERROR", **additional_fields) -> Dict[str, Any]:
    """
    Create a standardized error response with consistent fields.
    
    Args:
        error_message: Human-readable error description
        status: Status code (ERROR, WARNING, etc.)
        **additional_fields: Additional fields to include in the response
    
    Returns:
        Dict with standardized error fields
    """
    response = {
        "status": status,
        "error_message": error_message,
        "error": error_message,  # Always set legacy error field for consistency
    }
    response.update(additional_fields)
    return response

def create_success_response(status: str = "SUCCESS", **fields) -> Dict[str, Any]:
    """
    Create a standardized success response with consistent fields.
    
    Args:
        status: Status code (SUCCESS, PLANNING, etc.)
        **fields: Additional fields to include in the response
    
    Returns:
        Dict with standardized success fields
    """
    response = {
        "status": status,
    }
    response.update(fields)
    return response
