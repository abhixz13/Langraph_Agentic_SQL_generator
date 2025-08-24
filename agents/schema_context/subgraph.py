# agents/schema_context/subgraph.py
"""
Schema Context Agent - Production Optimized for Complex SQL Generation

Purpose: Build robust, performant schema context for LLM-based SQL generation
Focus: Essential metadata with production-grade robustness and performance
"""

import os
import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from langgraph.graph import StateGraph, START, END
from core.state import AppState

# Optional dependencies with graceful fallbacks
try:
    import yaml
except ImportError:
    yaml = None

try:
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _safe_int(value: Any, default: int) -> int:
    """Safely convert to int with fallback for robust configuration handling"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _compute_schema_signature(db_schema: Dict[str, Any], database_url: str, dialect: Optional[str]) -> str:
    """
    Create stable hash for schema caching - critical for performance on large databases
    Detects schema changes to avoid unnecessary rebuilds
    """
    tables = db_schema.get("tables", {}) or {}
    parts = [database_url or "", dialect or ""]
    
    # Build signature from table structure
    for tname in sorted(tables.keys()):
        parts.append(tname)
        cols = tables[tname].get("columns", []) or []
        # Include column names and types in signature
        parts.extend([f"{c.get('name', '')}:{str(c.get('type', ''))}" for c in cols])
        # Include primary key structure
        pk = tables[tname].get("primary_key", []) or []
        parts.append("pk:" + ",".join(sorted(pk)))
    
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _get_cache_file_path() -> str:
    """Get the path to the schema context cache file"""
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "schema_context.json")


def _load_cached_schema_context() -> Optional[Dict[str, Any]]:
    """
    Load cached schema context from file if it exists and is valid
    Returns None if cache is missing, invalid, or corrupted
    """
    cache_file = _get_cache_file_path()
    
    if not os.path.exists(cache_file):
        logger.info("No cached schema context found")
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        
        # Validate cached data structure
        required_keys = ['db_schema', 'semantic_schema', 'schema_context', 'metadata']
        if not all(key in cached_data for key in required_keys):
            logger.warning("Cached schema context has invalid structure")
            return None
        
        # Check if cache is too old (optional: 24 hours)
        metadata = cached_data.get('metadata', {})
        cache_time = metadata.get('created_at', 0)
        current_time = time.time()
        
        # Cache expires after 24 hours (86400 seconds)
        if current_time - cache_time > 86400:
            logger.info("Cached schema context is expired (older than 24 hours)")
            return None
        
        logger.info(f"Loaded cached schema context (created: {time.ctime(cache_time)})")
        return cached_data
        
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load cached schema context: {e}")
        return None


def _save_schema_context_to_cache(db_schema: Dict[str, Any], semantic_schema: Dict[str, Any], 
                                 schema_context: Dict[str, Any], database_url: str, 
                                 dialect: Optional[str]) -> bool:
    """
    Save schema context to cache file for future use
    Returns True if successful, False otherwise
    """
    cache_file = _get_cache_file_path()
    
    try:
        # Create cache data with metadata
        cache_data = {
            'db_schema': db_schema,
            'semantic_schema': semantic_schema,
            'schema_context': schema_context,
            'metadata': {
                'created_at': time.time(),
                'database_url': database_url,
                'dialect': dialect,
                'schema_signature': _compute_schema_signature(db_schema, database_url, dialect),
                'version': '1.0'
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Write to cache file
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved schema context to cache: {cache_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save schema context to cache: {e}")
        return False


def _is_database_changed(cached_data: Dict[str, Any], current_database_url: str, 
                        current_dialect: Optional[str]) -> bool:
    """
    Check if the database has changed since the cache was created
    Returns True if database changed, False if same
    """
    metadata = cached_data.get('metadata', {})
    cached_url = metadata.get('database_url', '')
    cached_dialect = metadata.get('dialect', '')
    
    # Check if database URL or dialect changed
    if cached_url != current_database_url or cached_dialect != (current_dialect or ''):
        logger.info(f"Database changed: URL={cached_url}->{current_database_url}, Dialect={cached_dialect}->{current_dialect}")
        return True
    
    return False


class SchemaReflector:
    """
    Robust database schema reflection optimized for complex SQL generation
    Balances performance with essential metadata for production use
    """
    
    def __init__(self, database_url: str, dialect: Optional[str], sample_rows: int = 2, timeout_ms: int = 5000):
        self.database_url = database_url
        self.dialect = (dialect or "").lower() or None
        self.sample_rows = max(0, sample_rows)
        self.timeout_s = max(1, timeout_ms // 1000)
    
    def reflect(self) -> Dict[str, Any]:
        """
        Reflect essential schema metadata for SQL generation
        Returns robust dict: {dialect, tables: {name: {columns, pk, fk, row_count, sample}}}
        """
        # Use SQLAlchemy for complex databases, fallback to sqlite3 for SQLite
        if SQLALCHEMY_AVAILABLE and self.database_url and not self._is_sqlite_without_sa():
            return self._reflect_sqlalchemy()
        elif SQLITE_AVAILABLE and self._is_sqlite_url():
            return self._reflect_sqlite()
        
        return {"dialect": self._infer_dialect(), "tables": {}}
    
    def _is_sqlite_url(self) -> bool:
        return (self.database_url or "").startswith("sqlite://")
    
    def _is_sqlite_without_sa(self) -> bool:
        return self._is_sqlite_url() and not SQLALCHEMY_AVAILABLE
    
    def _infer_dialect(self) -> str:
        """Infer SQL dialect from URL for query generation"""
        if self.dialect:
            return self.dialect
        url = (self.database_url or "").lower()
        if url.startswith("sqlite"):
            return "sqlite"
        if url.startswith("postgres"):
            return "postgres"
        if "mysql" in url:
            return "mysql"
        return "unknown"
    
    def _reflect_sqlalchemy(self) -> Dict[str, Any]:
        """
        SQLAlchemy reflection for complex databases (PostgreSQL, MySQL, etc.)
        Optimized for large datasets with essential metadata
        """
        result = {"dialect": self._infer_dialect(), "tables": {}}
        engine = None
        
        try:
            # Create engine with optimized pooling for performance
            engine = create_engine(
                self.database_url, 
                pool_pre_ping=True,
                pool_size=3,  # Optimized for typical usage
                max_overflow=5
            )
            insp = inspect(engine)
            
            # Get all tables efficiently
            for tname in insp.get_table_names():
                # Extract essential column metadata
                cols_meta = []
                for col in insp.get_columns(tname):
                    cols_meta.append({
                        "name": col.get("name"),
                        "type": str(col.get("type")),
                        "nullable": bool(col.get("nullable", True)),
                        "default": col.get("default")  # Restored for constraint understanding
                    })
                
                # Get primary key for JOIN optimization
                pk_cols = insp.get_pk_constraint(tname).get("constrained_columns") or []
                
                # Get foreign keys for complex query generation
                fk_meta = []
                for fk in insp.get_foreign_keys(tname):
                    rtab = fk.get("referred_table")
                    rcols = fk.get("referred_columns") or []
                    for c, rc in zip(fk.get("constrained_columns") or [], rcols):
                        fk_meta.append({"column": c, "references": f"{rtab}.{rc}"})
                
                # Sampling and row count for optimization
                sample = []
                row_count = None
                if self.sample_rows > 0 and text:
                    try:
                        with engine.connect() as conn:
                            # Use parameterized query for safety
                            rs = conn.execute(
                                text("SELECT * FROM {} LIMIT :lim".format(tname)), 
                                {"lim": self.sample_rows}
                            )
                            sample = [dict(row._mapping) for row in rs]
                            
                            # Get row count for optimization hints (restored)
                            try:
                                rc = conn.execute(text(f"SELECT COUNT(1) FROM {tname}"))
                                row_count = int(list(rc)[0][0])
                            except Exception:
                                row_count = None
                    except Exception:
                        sample = []
                        row_count = None
                
                result["tables"][tname] = {
                    "columns": cols_meta,
                    "primary_key": list(pk_cols),
                    "foreign_keys": fk_meta,
                    "row_count_estimate": row_count,  # Restored for optimization
                    "sample": sample
                }
                
        except SQLAlchemyError as e:
            logger.warning(f"SQLAlchemy reflection failed: {e}")
        finally:
            if engine:
                try:
                    engine.dispose()
                except Exception:
                    pass
        
        return result
    
    def _reflect_sqlite(self) -> Dict[str, Any]:
        """
        SQLite-specific reflection using PRAGMA (robust fallback)
        Optimized for SQLite databases with essential metadata
        """
        path = (self.database_url or "").replace("sqlite:///", "")
        result = {"dialect": "sqlite", "tables": {}}
        
        if not os.path.exists(path):
            return result
        
        conn = None
        try:
            conn = sqlite3.connect(path, timeout=self.timeout_s)
            conn.row_factory = sqlite3.Row
            
            # Get all tables
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            
            for row in tables:
                tname = row["name"]
                
                # Get essential column metadata via PRAGMA
                cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
                cols_meta = [{
                    "name": c["name"],
                    "type": c["type"],
                    "nullable": (c["notnull"] == 0),
                    "default": c["dflt_value"]  # Restored for constraint understanding
                } for c in cols]
                
                # Get primary key for optimization
                pk_cols = [c["name"] for c in cols if c["pk"]]
                
                # Get foreign keys for complex queries
                fk = conn.execute(f"PRAGMA foreign_key_list('{tname}')").fetchall()
                fk_meta = [{"column": f["from"], "references": f"{f['table']}.{f['to']}"} for f in fk]
                
                # Sample data and row count
                sample = []
                row_count = None
                if self.sample_rows > 0:
                    try:
                        # Use parameterized query for safety
                        rs = conn.execute(
                            f"SELECT * FROM '{tname}' LIMIT ?", 
                            (self.sample_rows,)
                        ).fetchall()
                        sample = [dict(r) for r in rs]
                        
                        # Get row count for optimization (restored)
                        rcc = conn.execute(f"SELECT COUNT(1) as c FROM '{tname}'").fetchone()
                        row_count = int(rcc["c"]) if rcc and "c" in rcc.keys() else None
                    except Exception:
                        sample = []
                        row_count = None
                
                result["tables"][tname] = {
                    "columns": cols_meta,
                    "primary_key": pk_cols,
                    "foreign_keys": fk_meta,
                    "row_count_estimate": row_count,  # Restored for optimization
                    "sample": sample
                }
                
        except Exception as e:
            logger.warning(f"SQLite reflection failed: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        
        return result


class SemanticLoader:
    """
    Load semantic context for business understanding
    Focuses on essential business mappings with robust error handling
    """
    
    def __init__(self, semantic_dir: str):
        self.semantic_dir = semantic_dir or "data/semantic"
    
    def load(self) -> Dict[str, Any]:
        """
        Load essential semantic mappings for SQL generation
        Returns: {tables: {}, synonyms: {}, metrics: {}}
        """
        result = {"tables": {}, "synonyms": {}, "metrics": {}}
        
        if not os.path.isdir(self.semantic_dir):
            return result
        
        # Load JSON files efficiently
        for path in os.listdir(self.semantic_dir):
            if not path.endswith('.json'):
                continue
                
            try:
                with open(os.path.join(self.semantic_dir, path), "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Merge semantic data with robust error handling
                if isinstance(data, dict):
                    result["tables"].update(data.get("tables", {}))
                    result["synonyms"].update(data.get("synonyms", {}))
                    result["metrics"].update(data.get("metrics", {}))  # Restored for business logic
            except Exception as e:
                logger.warning(f"Failed to load semantic file {path}: {e}")
                continue
        
        # Load YAML files if available
        if yaml:
            for path in os.listdir(self.semantic_dir):
                if not (path.endswith('.yaml') or path.endswith('.yml')):
                    continue
                    
                try:
                    with open(os.path.join(self.semantic_dir, path), "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    
                    if isinstance(data, dict):
                        result["tables"].update(data.get("tables", {}))
                        result["synonyms"].update(data.get("synonyms", {}))
                        result["metrics"].update(data.get("metrics", {}))  # Restored for business logic
                except Exception as e:
                    logger.warning(f"Failed to load semantic file {path}: {e}")
                    continue
        
        return result


class SchemaContextBuilder:
    """
    Build robust schema context optimized for complex SQL generation
    Balances performance with essential information for JOINs, WHERE clauses, and aggregations
    """
    
    def __init__(self, max_tables: int = 20, max_cols_per_table: int = 15, include_samples: bool = True):
        self.max_tables = max_tables
        self.max_cols = max_cols_per_table
        self.include_samples = include_samples
    
    def build(self, db_schema: Dict[str, Any], semantic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build robust LLM-friendly schema context for complex queries
        Prioritizes tables by importance (relationships + row count)
        """
        tables = db_schema.get("tables", {}) or {}
        dialect = db_schema.get("dialect", "unknown")
        
        # Sort tables by importance (relationships + row count, then alphabetically)
        def sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, int, str]:
            tname, meta = item
            # Prioritize tables with foreign keys (complex queries)
            has_fk = len(meta.get("foreign_keys", [])) > 0
            # Use row count for secondary sorting (restored)
            row_count = meta.get("row_count_estimate") or 0
            return (-int(has_fk), -int(row_count), tname)  # Descending by relationship count, then row count
        
        # Select most important tables for complex queries
        sorted_tables = sorted(tables.items(), key=sort_key)
        limited_tables = sorted_tables[:self.max_tables]
        
        # Build robust context tables
        context_tables = {}
        for tname, meta in limited_tables:
            cols = (meta.get("columns", []) or [])[:self.max_cols]
            
            # Get semantic information
            sem_table = (semantic.get("tables", {}) or {}).get(tname, {}) or {}
            
            # Build essential table context
            context_tables[tname] = {
                "alias": sem_table.get("alias") or sem_table.get("display_name") or tname,
                "description": sem_table.get("description") or "",  # Restored for context
                "columns": [
                    {
                        "name": c.get("name"),
                        "type": c.get("type"),
                        "alias": (sem_table.get("columns", {}) or {}).get(c.get("name"), {}).get("alias"),
                        "meaning": (sem_table.get("columns", {}) or {}).get(c.get("name"), {}).get("meaning"),  # Restored
                        "possible_values": (sem_table.get("columns", {}) or {}).get(c.get("name"), {}).get("possible_values"),  # Added
                        "business_context": (sem_table.get("columns", {}) or {}).get(c.get("name"), {}).get("business_context"),  # Added
                    }
                    for c in cols
                ],
                "primary_key": meta.get("primary_key", []),
                "foreign_keys": meta.get("foreign_keys", []),  # Essential for JOINs
                "row_count_estimate": meta.get("row_count_estimate"),  # Restored for optimization
                "sample": meta.get("sample", []) if self.include_samples else [],
            }
        
        return {
            "dialect": dialect,
            "tables": context_tables,
            "synonyms": dict(semantic.get("synonyms", {}) or {}),
            "metrics": dict(semantic.get("metrics", {}) or {})  # Restored for business logic
        }


def schema_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main schema context node - optimized for complex SQL generation with robust caching
    
    Steps:
    1. Check for cached schema context (if available and valid)
    2. If cache miss: Reflect essential database schema (with caching)
    3. If cache miss: Load business context (aliases, synonyms, metrics)
    4. If cache miss: Build LLM-friendly context for complex queries
    5. Save to cache for future use
    6. Return robust patch for graph state
    """
    start_ms = int(time.time() * 1000)
    
    # Extract configuration with robust defaults
    database_url = getattr(state, "database_url", None) or os.getenv("DATABASE_URL", "sqlite:///data/sample.db")
    dialect = getattr(state, "dialect", None)
    semantic_dir = getattr(state, "semantic_dir", None) or "data/semantic"
    
    # Get policy with production-optimized defaults
    policy = getattr(state, "policy", None) or {}
    max_tables = _safe_int(policy.get("schema_max_tables"), 20)
    max_cols = _safe_int(policy.get("schema_max_columns"), 30)
    include_samples = bool(policy.get("schema_include_samples", True))
    sample_rows = _safe_int(policy.get("schema_sample_rows"), 2)
    timeout_ms = _safe_int(policy.get("execute_timeout_ms"), 5000)
    cache_enabled = bool(policy.get("schema_cache_enabled", True))  # Restored caching
    
    metrics = dict(getattr(state, "metrics", None) or {})
    
    try:
        # Step 1: Check for cached schema context first
        if cache_enabled:
            cached_data = _load_cached_schema_context()
            if cached_data and not _is_database_changed(cached_data, database_url, dialect):
                # Cache hit - use existing schema context
                logger.info("Using cached schema context")
                metrics["schema_cache_hit"] = True
                metrics["schema_context_ms"] = int(time.time() * 1000) - start_ms
                
                return {
                    "db_schema": cached_data["db_schema"],
                    "schema_context": cached_data["schema_context"],
                    "last_schema_signature": cached_data["metadata"]["schema_signature"],
                    "metrics": metrics,
                    "status": "PLANNING"
                }
        
        # Cache miss - regenerate schema context
        logger.info("Generating new schema context")
        metrics["schema_cache_hit"] = False
        
        # Step 2: Reflect essential database schema
        t0 = int(time.time() * 1000)
        reflector = SchemaReflector(database_url, dialect, sample_rows if include_samples else 0, timeout_ms)
        db_schema = reflector.reflect()
        metrics["schema_reflection_ms"] = int(time.time() * 1000) - t0
        metrics["schema_tables_raw"] = len((db_schema or {}).get("tables", {}) or {})
        
        # Step 3: Check in-memory cache for performance optimization (restored)
        signature = _compute_schema_signature(db_schema, database_url, dialect)
        if cache_enabled and getattr(state, "last_schema_signature", None) == signature and getattr(state, "schema_context", None):
            # In-memory cache hit - reuse existing context
            metrics["schema_memory_cache_hit"] = True
            return {
                "db_schema": getattr(state, "db_schema", None) or db_schema,
                "schema_context": getattr(state, "schema_context", None),
                "last_schema_signature": signature,
                "metrics": metrics,
                "status": "PLANNING"
            }
        
        metrics["schema_memory_cache_hit"] = False
        
        # Step 4: Load semantic context
        t1 = int(time.time() * 1000)
        semantic = SemanticLoader(semantic_dir).load()
        metrics["schema_semantic_ms"] = int(time.time() * 1000) - t1
        
        # Step 5: Build optimized context for complex queries
        t2 = int(time.time() * 1000)
        context_builder = SchemaContextBuilder(max_tables, max_cols, include_samples)
        schema_context = context_builder.build(db_schema, semantic)
        metrics["schema_context_ms"] = int(time.time() * 1000) - t2
        metrics["schema_tables_in_context"] = len((schema_context or {}).get("tables", {}) or {})
        
        # Step 6: Save to cache for future use
        if cache_enabled:
            _save_schema_context_to_cache(db_schema, semantic, schema_context, database_url, dialect)
        
        # Return robust patch
        return {
            "db_schema": db_schema,
            "schema_context": schema_context,
            "last_schema_signature": signature,  # Restored for caching
            "metrics": metrics,
            "status": "PLANNING"
        }
        
    except Exception as e:
        logger.exception(f"Schema context node failed: {e}")
        elapsed = int(time.time() * 1000) - start_ms
        metrics["schema_context_ms"] = elapsed
        
        # Return safe fallback for downstream nodes
        return {
            "db_schema": {"dialect": dialect or "unknown", "tables": {}},
            "schema_context": {"dialect": dialect or "unknown", "tables": {}, "synonyms": {}, "metrics": {}},
            "metrics": metrics,
            "status": "ERROR",
            "error_message": f"Schema context error: {e}"
        }


def build_schema_context_subgraph() -> StateGraph:
    """
    Build the schema context subgraph
    
    This subgraph handles:
    1. Database schema reflection
    2. Semantic context loading
    3. Schema context building and caching
    4. Performance optimization
    
    Returns a subgraph that routes to END for parent workflow integration.
    """
    workflow = StateGraph(AppState)
    
    # Add nodes
    workflow.add_node("schema_context", schema_context_node)
    
    # Add edges
    workflow.add_edge(START, "schema_context")
    workflow.add_edge("schema_context", END)
    
    return workflow.compile()

# Export the compiled subgraph
schema_context_subgraph = build_schema_context_subgraph()


# Quick test for development
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Robust test configuration
    test_state = {
        "database_url": os.getenv("DATABASE_URL", "sqlite:///data/sample.db"),
        "dialect": os.getenv("DB_DIALECT", "sqlite"),
        "semantic_dir": os.getenv("SEMANTIC_DIR", "data/semantic"),
        "policy": {
            "schema_max_tables": 20,      # Balanced for performance
            "schema_max_columns": 15,     # Essential columns
            "schema_include_samples": True,
            "schema_sample_rows": 2,      # Minimal sampling
            "execute_timeout_ms": 5000,   # Fast timeout
            "schema_cache_enabled": True  # Restored caching
        }
    }
    
    # Run test
    result = schema_context_node(test_state)
    
    # Print comprehensive summary
    summary = {
        "status": result.get("status"),
        "tables_raw": len((result.get("db_schema") or {}).get("tables", {}) or {}),
        "tables_in_context": len((result.get("schema_context") or {}).get("tables", {}) or {}),
        "dialect": (result.get("schema_context") or {}).get("dialect", "unknown"),
        "performance_ms": {k: v for k, v in (result.get("metrics") or {}).items() if k.endswith("_ms")},
        "cache_hit": result.get("metrics", {}).get("schema_cache_hit", False),
        "memory_cache_hit": result.get("metrics", {}).get("schema_memory_cache_hit", False)
    }
    print(json.dumps(summary, indent=2))
