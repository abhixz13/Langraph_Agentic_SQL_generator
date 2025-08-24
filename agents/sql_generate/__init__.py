"""
SQL Generation Agent Module

This module provides the SQL generation subgraph that converts structured intent
into SQL statements using the intent from interpret_plan for efficiency.
"""

from .subgraph import sql_generate_subgraph

__all__ = ["sql_generate_subgraph"]
