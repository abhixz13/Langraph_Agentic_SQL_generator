"""
Interpret Plan Agent Subgraph

This module implements the interpret plan agent that uses LLM to understand
natural language queries and create execution plans.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
from langgraph.graph import StateGraph, START, END

from core.state import AppState, create_error_response, create_success_response
from core.llm_adapter import get_llm_adapter

logger = logging.getLogger(__name__)

def adapt_schema_for_llm(schema_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt schema to LLM-friendly format similar to reference implementation.
    
    Converts our schema format to a more structured format that's easier for LLM to parse.
    """
    tables = []
    
    for table_name, table_info in schema_info.get("tables", {}).items():
        # Convert columns to list format
        columns = []
        columns_info = table_info.get("columns", {})
        
        if isinstance(columns_info, dict):
            # Handle dictionary format
            for col_name, col_info in columns_info.items():
                if isinstance(col_info, dict):
                    col_data = {
                        "name": col_name,
                        "type": col_info.get("data_type", "TEXT"),
                        "is_pk": bool(col_info.get("is_primary_key", False)),
                        "not_null": not col_info.get("nullable", True)
                    }
                else:
                    col_data = {
                        "name": col_info.get("name", col_name),
                        "type": col_info.get("type", "TEXT"),
                        "is_pk": bool(col_info.get("primary_key", False)),
                        "not_null": not col_info.get("nullable", True)
                    }
                columns.append(col_data)
        elif isinstance(columns_info, list):
            # Handle list format from schema reflection
            for col_info in columns_info:
                col_data = {
                    "name": col_info.get("name", ""),
                    "type": col_info.get("type", "TEXT"),
                    "is_pk": bool(col_info.get("is_pk", False)),
                    "not_null": not col_info.get("nullable", True)
                }
                columns.append(col_data)
        
        # Extract primary keys
        primary_keys = [col["name"] for col in columns if col.get("is_pk", False)]
        
        # Extract foreign keys (if available)
        foreign_keys = []
        relationships = table_info.get("relationships", [])
        for rel in relationships:
            foreign_keys.append({
                "from": rel.get("column"),
                "to_table": rel.get("referenced_table"),
                "to_column": rel.get("referenced_column")
            })
        
        tables.append({
            "name": table_name,
            "columns": columns,
            "primary_key": primary_keys,
            "foreign_keys": foreign_keys
        })
    
    result = {"tables": tables}
    
    # Include semantic context if available
    if "semantic_schema" in schema_info:
        result["semantic_context"] = schema_info["semantic_schema"]
    
    # Include business context from schema_context (enhanced)
    if "synonyms" in schema_info:
        result["synonyms"] = schema_info["synonyms"]
    if "metrics" in schema_info:
        result["metrics"] = schema_info["metrics"]
    
    # Include business context for each table
    for table_name, table_info in schema_info.get("tables", {}).items():
        if table_name in result["tables"]:
            # Find the table in result
            for table in result["tables"]:
                if table["name"] == table_name:
                    # Add business context to table
                    if "alias" in table_info:
                        table["alias"] = table_info["alias"]
                    if "description" in table_info:
                        table["description"] = table_info["description"]
                    
                    # Add business context to columns
                    for col in table["columns"]:
                        col_name = col["name"]
                        # Handle both dict and list formats for columns
                        columns_info = table_info.get("columns", {})
                        if isinstance(columns_info, dict) and col_name in columns_info:
                            col_info = columns_info[col_name]
                            if "alias" in col_info:
                                col["alias"] = col_info["alias"]
                            if "meaning" in col_info:
                                col["meaning"] = col_info["meaning"]
                            if "possible_values" in col_info:
                                col["possible_values"] = col_info["possible_values"]
                            if "business_context" in col_info:
                                col["business_context"] = col_info["business_context"]
                        elif isinstance(columns_info, list):
                            # Find column in list format
                            for col_info in columns_info:
                                if col_info.get("name") == col_name:
                                    if "alias" in col_info:
                                        col["alias"] = col_info["alias"]
                                    if "meaning" in col_info:
                                        col["meaning"] = col_info["meaning"]
                                    if "possible_values" in col_info:
                                        col["possible_values"] = col_info["possible_values"]
                                    if "business_context" in col_info:
                                        col["business_context"] = col_info["business_context"]
                                    break
                    break
    
    return result

def detect_mentions_generic(user_query: str) -> List[Dict[str, Any]]:
    """
    Detect various types of mentions in user query for better intent parsing.
    
    Based on reference implementation with enhancements for our use case.
    """
    mentions: List[Dict[str, Any]] = []

    # Years (existing)
    for m in re.finditer(r"\b(19|20)\d{2}\b", user_query):
        mentions.append({"text": m.group(0), "type": "year", "span": [m.start(), m.end()]})

    # Rank with explicit N (existing)
    for m in re.finditer(r"\b(top|bottom)\s+(\d{1,3})\b", user_query, re.I):
        direction = "DESC" if m.group(1).lower() == "top" else "ASC"
        mentions.append({
            "text": m.group(0), 
            "type": "rank", 
            "n": int(m.group(2)), 
            "direction": direction,
            "span": [m.start(), m.end()]
        })

    # Enhanced Rank words WITHOUT N (enhanced)
    rank_patterns = [
        (r"\b(top|best|highest|maximum)\b", "DESC"),
        (r"\b(bottom|worst|lowest|minimum)\b", "ASC"),
        (r"\b(most)\s+(\w+)", "DESC"),  # "most bookings"
        (r"\b(fewest|least)\s+(\w+)", "ASC"),  # "fewest bookings"
    ]
    
    for pattern, direction in rank_patterns:
        for m in re.finditer(pattern, user_query, re.I):
            # avoid duplicating spans already captured as "top/bottom N"
            if not any(m.start() >= r["span"][0] and m.end() <= r["span"][1]
                    for r in mentions if r["type"] == "rank"):
                mentions.append({
                    "text": m.group(0), 
                    "type": "rank", 
                    "n": None, 
                    "direction": direction,
                    "metric_hint": m.group(2) if len(m.groups()) > 1 else None,
                    "span": [m.start(), m.end()]
                })

    # Comparators (existing)
    for m in re.finditer(r"([<>]=?|=)\s*\$?\d[\d,]*(\.\d+)?", user_query):
        mentions.append({"text": m.group(0), "type": "comparator", "span": [m.start(), m.end()]})

    # Enhanced Period hints with per-group detection (enhanced)
    per_group_patterns = [
        (r"\bfor\s+each\s+(\w+)", "for_each"),     # "for each year"
        (r"\bin\s+each\s+(\w+)", "in_each"),       # "in each year"
        (r"\bper\s+(\w+)", "per"),                 # "per year"
        (r"\bby\s+(\w+)", "by"),                   # "by year" (when used with ranking)
        (r"\bwithin\s+each\s+(\w+)", "within"),    # "within each year"
        (r"\bacross\s+(\w+)", "across"),           # "across years"
    ]
    
    for pattern, per_type in per_group_patterns:
        for m in re.finditer(pattern, user_query, re.I):
            mentions.append({
                "text": m.group(0), 
                "type": "per_group", 
                "per_type": per_type,
                "group_hint": m.group(1),
                "span": [m.start(), m.end()]
            })
    
    # Simple period mentions (fallback)
    for m in re.finditer(r"\b(year|years|quarter|quarters|month|months)\b", user_query, re.I):
        # Only add if not already captured in per_group
        if not any(m.start() >= r["span"][0] and m.end() <= r["span"][1]
                for r in mentions if r["type"] == "per_group"):
            mentions.append({"text": m.group(0), "type": "period", "span": [m.start(), m.end()]})

    # Entity hints (new)
    for m in re.finditer(r"\b(customer|customers|account|accounts|product|products|region|regions|country|countries|market|markets)\b",
                        user_query, re.I):
        mentions.append({"text": m.group(0), "type": "entity_hint", "span": [m.start(), m.end()]})

    return mentions

def build_context_generic(user_query: str, schema_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build comprehensive context for LLM including mentions and schema information.
    """
    # Detect mentions in the query
    detected_mentions = detect_mentions_generic(user_query)
    
    # Build column aliases for better matching
    column_aliases: Dict[str, List[str]] = {}
    tables = schema_manifest.get("tables", {})
    
    # Handle both list and dict formats for tables
    if isinstance(tables, dict):
        for table_name, table_info in tables.items():
            columns = table_info.get("columns", [])
            for col in columns:
                col_name = col.get("name") if isinstance(col, dict) else str(col)
                if col_name:
                    # Generate aliases for column names
                    aliases = _generate_column_aliases(col_name)
                    column_aliases[col_name] = aliases
    elif isinstance(tables, list):
        for table in tables:
            for col in table.get("columns", []):
                col_name = col.get("name")
                if col_name:
                    # Generate aliases for column names
                    aliases = _generate_column_aliases(col_name)
                    column_aliases[col_name] = aliases
    
    return {
        "schema_manifest": schema_manifest,
        "target_dialect": "generic",
        "user_query": user_query,
        "column_aliases": column_aliases,
        "detected_mentions": detected_mentions
    }

def _generate_column_aliases(col_name: str) -> List[str]:
    """
    Generate aliases for column names to improve matching.
    """
    # Convert camelCase/PascalCase to words
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", col_name).replace("_", " ").split()
    parts = [p for p in parts if p]
    
    if not parts:
        return []
    
    phrases = []
    phrases.append(" ".join(parts).lower())
    if len(parts) >= 2:
        phrases.append(" ".join(parts[-2:]).lower())
    phrases.append(parts[-1].lower())
    
    # Drop common suffixes
    drop = {"id", "ids", "key", "keys", "code", "codes"}
    if parts[-1].lower() in drop and len(parts) >= 2:
        phrases.append(" ".join(parts[:-1]).lower())
    
    # Deduplicate
    dedup = []
    for p in phrases:
        if p and p not in dedup:
            dedup.append(p)
    
    return dedup[:5]

def validate_intent_response(intent_result: Dict[str, Any], policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate and sanitize LLM intent response
    
    Handles both old format (flat structure) and new format (params structure)
    
    Ensures:
    1. Required keys exist with correct types
    2. Values are within expected ranges
    3. Ambiguous responses are flagged
    4. Fallback values are provided
    """
    # Get policy defaults
    intent_policy = policy.get("intent", {}) if policy else {}
    default_confidence = intent_policy.get("default_confidence", 0.5)
    fallback_confidence = intent_policy.get("fallback_confidence", 0.1)
    default_action = intent_policy.get("default_action", "SELECT")
    valid_actions = intent_policy.get("valid_actions", ["SELECT", "COUNT", "AGGREGATE", "SEARCH", "COMPARE"])
    default_complexity = intent_policy.get("default_complexity", "simple")
    valid_complexities = intent_policy.get("valid_complexities", ["simple", "moderate", "complex"])
    
    if not isinstance(intent_result, dict):
        logger.warning("LLM returned non-dict response, using fallback")
        return _get_fallback_intent(fallback_confidence)
    
    # Check if this is new format (has params structure)
    is_new_format = "params" in intent_result
    
    # Validate required fields with type checking
    validated = {}
    
    # Action validation
    action = intent_result.get("action", default_action)
    if not isinstance(action, str) or action.upper() not in [a.upper() for a in valid_actions]:
        logger.warning(f"Invalid action '{action}', defaulting to {default_action}")
        action = default_action
    validated["action"] = action.upper()
    
    if is_new_format:
        # Handle new format with params structure
        params = intent_result.get("params", {})
        used_schema = intent_result.get("used_schema", {})
        
        # Tables validation - extract from used_schema
        tables = used_schema.get("tables", [])
        if not isinstance(tables, list):
            logger.warning(f"Tables should be list, got {type(tables)}, using empty list")
            tables = []
        validated["tables"] = [str(t) for t in tables if t]
        
        # Columns validation - extract from used_schema
        columns = used_schema.get("columns", [])
        if not isinstance(columns, list):
            logger.warning(f"Columns should be list, got {type(columns)}, using empty list")
            columns = []
        validated["columns"] = [str(c) for c in columns if c]
        
        # Conditions validation - convert from params.filters
        conditions = []
        filters = params.get("filters", [])
        for filter_item in filters:
            if isinstance(filter_item, dict):
                condition = f"{filter_item.get('column')} {filter_item.get('operator')} '{filter_item.get('value')}'"
                conditions.append(condition)
        validated["conditions"] = conditions
        
        # Extract additional information from params
        if params.get("order_by"):
            validated["order_by"] = params.get("order_by")
        if params.get("limit"):
            validated["limit"] = params.get("limit")
        if "desc" in params:
            validated["desc"] = params.get("desc")
        if params.get("group_by"):
            validated["group_by"] = params.get("group_by")
        if params.get("function"):
            validated["function"] = params.get("function")
        
        # Check for top_n_per_group in post_aggregation
        post_agg = params.get("post_aggregation", {})
        top_n_per_group = post_agg.get("top_n_per_group", {})
        if top_n_per_group:
            validated["top_n_per_group"] = top_n_per_group
        
    else:
        # Handle old format (flat structure)
        # Tables validation - check both root and used_schema
        tables = intent_result.get("tables", [])
        if not tables and "used_schema" in intent_result:
            tables = intent_result["used_schema"].get("tables", [])
        if not isinstance(tables, list):
            logger.warning(f"Tables should be list, got {type(tables)}, using empty list")
            tables = []
        validated["tables"] = [str(t) for t in tables if t]  # Convert to strings, filter empty
        
        # Columns validation - check both root and used_schema
        columns = intent_result.get("columns", [])
        if not columns and "used_schema" in intent_result:
            columns = intent_result["used_schema"].get("columns", [])
        if not isinstance(columns, list):
            logger.warning(f"Columns should be list, got {type(columns)}, using empty list")
            columns = []
        validated["columns"] = [str(c) for c in columns if c]  # Convert to strings, filter empty
        
        # Conditions validation
        conditions = intent_result.get("conditions", [])
        if not isinstance(conditions, list):
            logger.warning(f"Conditions should be list, got {type(conditions)}, using empty list")
            conditions = []
        validated["conditions"] = [str(c) for c in conditions if c]  # Convert to strings, filter empty
        
        # Extract additional fields from old format
        if intent_result.get("aggregation_function"):
            validated["aggregation_function"] = intent_result.get("aggregation_function")
        if intent_result.get("order_by"):
            validated["order_by"] = intent_result.get("order_by")
        if intent_result.get("limit"):
            validated["limit"] = intent_result.get("limit")
        if "desc" in intent_result:
            validated["desc"] = intent_result.get("desc")
        if intent_result.get("group_by"):
            validated["group_by"] = intent_result.get("group_by")
        if intent_result.get("special_requirements"):
            validated["special_requirements"] = intent_result.get("special_requirements")
    
    # Ambiguity detection
    ambiguity_detected = intent_result.get("ambiguity_detected", False)
    if not isinstance(ambiguity_detected, bool):
        logger.warning(f"Ambiguity should be boolean, got {type(ambiguity_detected)}, defaulting to True")
        ambiguity_detected = True
    validated["ambiguity_detected"] = ambiguity_detected
    
    # Clarifying questions validation
    clarifying_questions = intent_result.get("clarifying_questions", [])
    if not isinstance(clarifying_questions, list):
        logger.warning(f"Clarifying questions should be list, got {type(clarifying_questions)}, using empty list")
        clarifying_questions = []
    validated["clarifying_questions"] = [str(q) for q in clarifying_questions if q]
    
    # Complexity level validation
    complexity_level = intent_result.get("complexity_level", default_complexity)
    if not isinstance(complexity_level, str) or complexity_level.lower() not in [c.lower() for c in valid_complexities]:
        logger.warning(f"Invalid complexity level '{complexity_level}', defaulting to {default_complexity}")
        complexity_level = default_complexity
    validated["complexity_level"] = complexity_level.lower()
    
    # Confidence validation
    confidence = intent_result.get("confidence", default_confidence)
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        logger.warning(f"Invalid confidence {confidence}, defaulting to {default_confidence}")
        confidence = default_confidence
    validated["confidence"] = float(confidence)
    
    # Additional validation logic
    _validate_intent_logic(validated)
    
    return validated

def _validate_intent_logic(intent: Dict[str, Any]) -> None:
    """
    Validate logical consistency of intent
    
    Checks for:
    1. Tables exist if columns are specified
    2. Conditions make sense
    3. Action matches complexity
    """
    # If columns specified but no tables, flag as ambiguous
    if intent["columns"] and not intent["tables"]:
        logger.warning("Columns specified but no tables identified - marking as ambiguous")
        intent["ambiguity_detected"] = True
        intent["clarifying_questions"].append("Which table(s) should I query?")
    
    # If complex action but simple complexity, adjust
    if intent["action"] in ["AGGREGATE", "COMPARE"] and intent["complexity_level"] == "simple":
        intent["complexity_level"] = "moderate"
    
    # If no tables and no ambiguity, this is suspicious
    if not intent["tables"] and not intent["ambiguity_detected"]:
        logger.warning("No tables identified but no ambiguity detected - marking as ambiguous")
        intent["ambiguity_detected"] = True
        intent["clarifying_questions"].append("I couldn't identify which table to query. Could you clarify?")

def _get_fallback_intent(fallback_confidence: float) -> Dict[str, Any]:
    """
    Provide safe fallback intent when LLM response is completely invalid
    """
    return {
        "action": "SELECT",
        "tables": [],
        "columns": [],
        "conditions": [],
        "ambiguity_detected": True,
        "clarifying_questions": ["I couldn't understand your query. Could you rephrase it?"],
        "complexity_level": "simple",
        "confidence": fallback_confidence
    }

def prefilter_schema(user_query: str, schema_context: Dict[str, Any], max_tables: int = 8, max_columns_per_table: int = 10) -> Dict[str, Any]:
    """
    Pre-filter schema to select relevant tables and columns based on user query.
    
    This reduces token usage and improves LLM accuracy by focusing on relevant schema elements.
    
    Strategy:
    1. Keyword matching on table/column names and aliases
    2. Semantic similarity using business context
    3. Entity heuristics (e.g., *_id, *_at patterns)
    4. Score-based ranking and selection
    """
    if not schema_context or not user_query:
        return schema_context
    
    # Normalize user query for matching
    query_lower = user_query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Extract schema components
    tables = schema_context.get("tables", {})
    semantic_schema = schema_context.get("semantic_schema", {})
    
    # Score tables based on relevance
    table_scores = _score_tables(query_words, tables, semantic_schema)
    
    # Select top tables
    selected_tables = _select_top_tables(table_scores, max_tables)
    
    # For each selected table, score and select columns
    filtered_schema = {
        "tables": {},
        "semantic_schema": semantic_schema,  # Keep semantic context
        "metadata": schema_context.get("metadata", {})
    }
    
    for table_name in selected_tables:
        table_data = tables.get(table_name, {})
        columns = table_data.get("columns", {})
        
        # Score columns for this table
        column_scores = _score_columns(query_words, columns, semantic_schema.get(table_name, {}))
        
        # Select top columns
        selected_columns = _select_top_columns(column_scores, max_columns_per_table)
        
        # Build filtered table data
        filtered_table = {
            "name": table_name,
            "description": table_data.get("description", ""),
            "row_count_estimate": table_data.get("row_count_estimate", 0),
        }
        
        # Handle both dict and list column formats
        if isinstance(columns, dict):
            filtered_table["columns"] = {col: columns[col] for col in selected_columns if col in columns}
        elif isinstance(columns, list):
            filtered_table["columns"] = [col for col in columns if col.get("name") in selected_columns]
        else:
            filtered_table["columns"] = columns
        
        # Add relationships if they reference selected tables
        relationships = table_data.get("relationships", [])
        filtered_relationships = [
            rel for rel in relationships 
            if rel.get("referenced_table") in selected_tables
        ]
        if filtered_relationships:
            filtered_table["relationships"] = filtered_relationships
        
        filtered_schema["tables"][table_name] = filtered_table
    
    logger.info(f"Schema prefiltered: {len(tables)} → {len(selected_tables)} tables, "
                f"~{sum(len(t.get('columns', {})) for t in filtered_schema['tables'].values())} columns")
    
    return filtered_schema

def _score_tables(query_words: set, tables: Dict[str, Any], semantic_schema: Dict[str, Any]) -> Dict[str, float]:
    """
    Score tables based on relevance to user query.
    
    Scoring factors:
    1. Direct keyword matches in table name/alias
    2. Semantic similarity from business context
    3. Column-level keyword matches
    4. Entity patterns (e.g., user-related tables for user queries)
    """
    table_scores = {}
    
    for table_name, table_data in tables.items():
        score = 0.0
        
        # 1. Table name/alias matching
        table_name_lower = table_name.lower()
        table_aliases = semantic_schema.get(table_name, {}).get("aliases", [])
        
        # Direct table name match
        if any(word in table_name_lower for word in query_words):
            score += 10.0
        
        # Alias matching
        for alias in table_aliases:
            alias_lower = alias.lower()
            if any(word in alias_lower for word in query_words):
                score += 8.0
        
        # 2. Semantic similarity from business context
        business_context = semantic_schema.get(table_name, {})
        business_keywords = set()
        
        # Extract keywords from business context
        description = business_context.get("description", "").lower()
        business_keywords.update(re.findall(r'\b\w+\b', description))
        
        purpose = business_context.get("business_purpose", "").lower()
        business_keywords.update(re.findall(r'\b\w+\b', purpose))
        
        # Score based on business keyword overlap
        keyword_overlap = len(query_words.intersection(business_keywords))
        score += keyword_overlap * 3.0
        
        # 3. Column-level keyword matching
        columns = table_data.get("columns", {})
        column_matches = 0
        
        # Handle both dict and list column formats
        if isinstance(columns, dict):
            columns_iter = columns.items()
        elif isinstance(columns, list):
            columns_iter = [(col.get("name", ""), col) for col in columns]
        else:
            columns_iter = []
        
        for col_name, col_data in columns_iter:
            col_name_lower = col_name.lower()
            col_aliases = col_data.get("aliases", [])
            
            # Column name match
            if any(word in col_name_lower for word in query_words):
                column_matches += 1
            
            # Column alias match
            for alias in col_aliases:
                if any(word in alias.lower() for word in query_words):
                    column_matches += 1
        
        score += column_matches * 2.0
        
        # 4. Entity pattern matching
        if _matches_entity_patterns(table_name, query_words):
            score += 5.0
        
        # 5. Relationship bonus (tables that connect to other relevant tables)
        relationships = table_data.get("relationships", [])
        for rel in relationships:
            if rel.get("referenced_table") in tables:
                score += 1.0
        
        # 6. Row count bonus (prefer tables with data)
        row_count = table_data.get("row_count_estimate", 0)
        if row_count > 0:
            score += min(row_count / 1000, 2.0)  # Cap at 2.0
        
        table_scores[table_name] = score
    
    return table_scores

def _score_columns(query_words: set, columns: Dict[str, Any], table_semantic: Dict[str, Any]) -> Dict[str, float]:
    """
    Score columns based on relevance to user query.
    
    Scoring factors:
    1. Direct keyword matches in column name/alias
    2. Semantic similarity from business context
    3. Data type relevance (e.g., text columns for search queries)
    4. Entity patterns (e.g., *_id, *_at, *_name)
    """
    column_scores = {}
    
    # Handle both dict and list column formats
    if isinstance(columns, dict):
        columns_iter = columns.items()
    elif isinstance(columns, list):
        columns_iter = [(col.get("name", ""), col) for col in columns]
    else:
        return column_scores
    
    for col_name, col_data in columns_iter:
        score = 0.0
        
        # 1. Column name/alias matching
        col_name_lower = col_name.lower()
        col_aliases = col_data.get("aliases", [])
        
        # Direct column name match
        if any(word in col_name_lower for word in query_words):
            score += 10.0
        
        # Alias matching
        for alias in col_aliases:
            if any(word in alias.lower() for word in query_words):
                score += 8.0
        
        # 2. Semantic similarity from business context
        col_semantic = table_semantic.get("columns", {}).get(col_name, {})
        
        # Business context keywords
        meaning = col_semantic.get("meaning", "").lower()
        meaning_keywords = set(re.findall(r'\b\w+\b', meaning))
        keyword_overlap = len(query_words.intersection(meaning_keywords))
        score += keyword_overlap * 3.0
        
        # Synonyms
        synonyms = col_semantic.get("synonyms", [])
        for synonym in synonyms:
            if any(word in synonym.lower() for word in query_words):
                score += 5.0
        
        # 3. Data type relevance
        data_type = col_data.get("data_type", "").lower()
        if "text" in data_type or "varchar" in data_type or "char" in data_type:
            if any(word in ["search", "find", "look", "name", "title", "description"] for word in query_words):
                score += 3.0
        
        if "date" in data_type or "time" in data_type:
            if any(word in ["date", "time", "when", "recent", "latest", "oldest"] for word in query_words):
                score += 3.0
        
        if "numeric" in data_type or "int" in data_type or "decimal" in data_type:
            if any(word in ["count", "sum", "average", "total", "number", "amount"] for word in query_words):
                score += 3.0
        
        # 4. Entity pattern matching
        if _matches_entity_patterns(col_name, query_words):
            score += 5.0
        
        # 5. Primary key/Foreign key bonus
        if col_data.get("is_primary_key", False):
            score += 2.0
        if col_data.get("is_foreign_key", False):
            score += 1.5
        
        # 6. Default value bonus (columns with meaningful defaults)
        if col_data.get("default") is not None:
            score += 0.5
        
        column_scores[col_name] = score
    
    return column_scores

def _matches_entity_patterns(name: str, query_words: set) -> bool:
    """
    Check if name matches common entity patterns relevant to query.
    """
    name_lower = name.lower()
    
    # Common entity patterns
    patterns = {
        "user": ["user", "customer", "client", "person", "member"],
        "id": ["id", "identifier", "key"],
        "name": ["name", "title", "label"],
        "date": ["date", "time", "created", "updated", "modified"],
        "status": ["status", "state", "condition"],
        "type": ["type", "category", "kind", "class"],
        "count": ["count", "number", "quantity", "amount"],
        "price": ["price", "cost", "amount", "value", "total"]
    }
    
    for pattern_type, keywords in patterns.items():
        # Check if query mentions this pattern type
        if any(keyword in query_words for keyword in keywords):
            # Check if name matches this pattern
            if any(keyword in name_lower for keyword in keywords):
                return True
            
            # Check common suffixes
            if name_lower.endswith(f"_{pattern_type}") or name_lower.endswith(f"_{pattern_type}s"):
                return True
    
    return False

def _select_top_tables(table_scores: Dict[str, float], max_tables: int) -> List[str]:
    """
    Select top-scoring tables, ensuring we have enough for the query.
    """
    if not table_scores:
        return []
    
    # Sort by score (descending)
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Always include at least 2 tables if available
    min_tables = min(2, len(sorted_tables))
    selected_tables = [table for table, score in sorted_tables[:max_tables]]
    
    # If we have very low scores, include more tables for context
    if len(selected_tables) < min_tables:
        selected_tables = [table for table, score in sorted_tables[:min_tables]]
    
    return selected_tables

def _select_top_columns(column_scores: Dict[str, float], max_columns: int) -> List[str]:
    """
    Select top-scoring columns for a table.
    """
    if not column_scores:
        return []
    
    # Sort by score (descending)
    sorted_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Always include primary keys and foreign keys
    selected_columns = []
    for col_name, score in sorted_columns:
        if len(selected_columns) >= max_columns:
            break
        selected_columns.append(col_name)
    
    return selected_columns

def interpret_plan_node(state: AppState) -> Dict[str, Any]:
    """
    Interpret natural language query and create execution plan using LLM
    
    This node uses LLM to:
    1. Parse the user's intent
    2. Identify relevant tables and columns
    3. Detect ambiguity
    4. Create an execution plan
    """
    try:
        logger.info(f"Interpreting query intent: {state.user_query}")
        
        # Get LLM client
        llm_client = get_llm_adapter()
        
        # Extract schema context
        schema_context = state.schema_context or {}
        
        # Get prefiltering limits from policy
        policy = getattr(state, "policy", None) or {}
        intent_policy = policy.get("intent", {}) if policy else {}
        max_tables = intent_policy.get("max_tables", 8)
        max_columns_per_table = intent_policy.get("max_columns_per_table", 10)
        
        # Bypass prefiltering - pass complete schema directly to LLM
        schema_manifest = schema_context
        
        # Build comprehensive context including mentions (enhanced)
        context = build_context_generic(state.user_query, schema_manifest)
        
        # Interpret query intent using LLM with enhanced context
        try:
            intent_result = llm_client.interpret_query_intent(
                user_query=state.user_query,
                schema_context=schema_manifest  # Use adapted schema only
            )
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            # Get fallback confidence from policy
            policy = getattr(state, "policy", None) or {}
            intent_policy = policy.get("intent", {}) if policy else {}
            fallback_confidence = intent_policy.get("fallback_confidence", 0.1)
            intent_result = _get_fallback_intent(fallback_confidence)
        
        # Validate and sanitize the LLM response
        validated_intent = validate_intent_response(intent_result, policy)
        
        # Enhanced: Analyze query complexity using detected mentions
        detected_mentions = context.get("detected_mentions", [])
        ranking_mentions = [m for m in detected_mentions if m.get("type") == "rank"]
        per_group_mentions = [m for m in detected_mentions if m.get("type") == "per_group"]
        
        # Determine query complexity based on mentions
        if ranking_mentions and per_group_mentions:
            query_complexity = "complex"  # Top N per group
        elif ranking_mentions:
            query_complexity = "moderate"  # Just top N overall
        elif per_group_mentions:
            query_complexity = "moderate"  # Group by without ranking
        else:
            query_complexity = validated_intent.get("complexity_level", "simple")
        
        logger.info(f"Intent interpretation completed: {validated_intent.get('action', 'unknown')}")
        
        # Extract validated intent details
        action = validated_intent["action"]
        tables = validated_intent["tables"]
        columns = validated_intent["columns"]
        conditions = validated_intent["conditions"]
        ambiguity_detected = validated_intent["ambiguity_detected"]
        clarifying_questions = validated_intent["clarifying_questions"]
        confidence = validated_intent["confidence"]
        
        # Create execution plan with enhanced features
        plan = {
            "action": action,
            "tables": tables,
            "columns": columns,
            "conditions": conditions,
            "complexity_level": query_complexity,
            "joins": [],  # Will be determined during SQL generation
            "where_conditions": conditions,
            "order_by": validated_intent.get("order_by"),  # Use from validated intent
            "limit": validated_intent.get("limit"),  # Use from validated intent
            "group_by": validated_intent.get("group_by"),  # Use from validated intent
            "detected_mentions": detected_mentions,  # Enhanced: Include mentions
            "query_complexity": query_complexity  # Enhanced: Include complexity analysis
        }
        
        # Enhanced: Add ranking information if detected
        if ranking_mentions:
            ranking_info = ranking_mentions[0]  # Take first ranking mention
            plan["ranking"] = {
                "type": ranking_info.get("type"),
                "direction": ranking_info.get("direction", "DESC"),
                "n": ranking_info.get("n", 1),
                "metric_hint": ranking_info.get("metric_hint")
            }
        elif validated_intent.get("order_by") and validated_intent.get("limit"):
            # Use ranking info from validated intent if available
            plan["ranking"] = {
                "type": "rank",
                "direction": "DESC" if validated_intent.get("desc", True) else "ASC",
                "n": validated_intent.get("limit", 1),
                "metric_hint": validated_intent.get("order_by")
            }
        
        # Enhanced: Add grouping information if detected
        if per_group_mentions:
            group_info = per_group_mentions[0]  # Take first group mention
            plan["grouping"] = {
                "type": group_info.get("type"),
                "per_type": group_info.get("per_type"),
                "group_hint": group_info.get("group_hint")
            }
        elif validated_intent.get("group_by"):
            # Use grouping info from validated intent if available
            plan["grouping"] = {
                "type": "per_group",
                "per_type": "for_each",
                "group_hint": validated_intent.get("group_by")[0] if validated_intent.get("group_by") else None
            }
        
        # Add window functions if top_n_per_group is present
        if validated_intent.get("top_n_per_group"):
            top_n_info = validated_intent.get("top_n_per_group")
            plan["window_functions"] = {
                "type": "row_number",
                "partition_by": [top_n_info.get("group_field")],
                "order_by": {
                    "column": top_n_info.get("order_by"),
                    "direction": "DESC" if top_n_info.get("desc", True) else "ASC"
                },
                "limit": top_n_info.get("n", 1)
            }
        
        # Determine if plan is acceptable with stricter validation
        # Get policy thresholds
        policy = getattr(state, "policy", None) or {}
        intent_policy = policy.get("intent", {}) if policy else {}
        min_confidence = intent_policy.get("min_confidence", 0.3)
        valid_actions = intent_policy.get("valid_actions", ["SELECT", "COUNT", "AGGREGATE", "SEARCH", "COMPARE"])
        
        plan_ok = (
            not ambiguity_detected and 
            confidence > min_confidence and 
            len(tables) > 0 and
            action in valid_actions
        )
        
        # Update state with results
        patch = create_success_response(
            status="PLANNING",
            intent_json=validated_intent,
            ambiguity=ambiguity_detected,
            clarifying_question=clarifying_questions[0] if clarifying_questions else None,
            plan=plan,
            plan_ok=plan_ok,
            plan_confidence=confidence,
            detected_mentions=detected_mentions,  # Enhanced: Include mentions in state
            query_complexity=query_complexity  # Enhanced: Include complexity in state
        )
        
        logger.info(f"Plan interpretation completed. Plan OK: {plan_ok}, Confidence: {confidence}, Complexity: {query_complexity}")
        return patch
        
    except Exception as e:
        logger.error(f"Plan interpretation failed: {str(e)}")
        return create_error_response(
            error_message=f"Plan interpretation error: {str(e)}",
            status="ERROR",
            intent_json=_get_fallback_intent(0.1),  # Use default fallback
            ambiguity=True,
            plan={},
            plan_ok=False,
            plan_confidence=0.0
        )

def interpret_plan_router(state: AppState) -> str:
    """
    Route based on plan interpretation results.
    
    Routes:
    - 'sql_generate': If plan is acceptable
    - 'human_approval': If ambiguity detected or low confidence
    - 'error_handler': If interpretation failed
    """
    # Check for errors first
    if getattr(state, "status", None) == "ERROR" or getattr(state, "error_message", None):
        logger.warning(f"Plan interpretation failed, routing to error handler: {getattr(state, 'error_message', None)}")
        return "error"
    
    # Check for ambiguity or low confidence
    if getattr(state, "ambiguity", False) or (getattr(state, "plan_confidence", 0) and getattr(state, "plan_confidence", 0) < 0.3):
        logger.info(f"Ambiguity detected or low confidence, routing to human approval")
        return "human_approval"
    
    # Check if plan is acceptable
    if getattr(state, "plan_ok", False):
        logger.info(f"Plan is acceptable, routing to SQL generation")
        return "sql_generate"
    
    # Default to human approval for unclear cases
    logger.info(f"Plan unclear, routing to human approval")
    return "human_approval"

def build_interpret_plan_subgraph() -> StateGraph:
    """
    Build the interpret plan subgraph
    
    This subgraph handles:
    1. Natural language query interpretation
    2. Intent parsing and table/column identification
    3. Ambiguity detection
    4. Execution plan creation
    5. Routing based on confidence and ambiguity
    
    Returns a subgraph that routes to END with route labels for parent workflow.
    """
    workflow = StateGraph(AppState)
    
    # Add nodes
    workflow.add_node("interpret_plan", interpret_plan_node)
    
    # Add edges
    workflow.add_edge(START, "interpret_plan")
    
    # Add conditional routing - route to END with labels for parent workflow
    workflow.add_conditional_edges(
        "interpret_plan",
        interpret_plan_router,
        {
            "sql_generate": END,  # Route to END, parent will handle
            "human_approval": END,  # Route to END, parent will handle
            "error": END  # Route to END, parent will handle
        }
    )
    
    return workflow.compile()

# Export the compiled subgraph
interpret_plan_subgraph = build_interpret_plan_subgraph()
