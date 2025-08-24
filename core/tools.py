from typing import Dict, List, Any, Optional, Union
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import sqlglot
import time
from contextlib import contextmanager
from core.safety import SafetyChecker, ReadOnlyConnection

class DatabaseManager:
    """Database connection manager with connection pooling"""
    
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.metadata = MetaData()
        self.safety_checker = SafetyChecker(config)
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling"""
        connection_string = self.config.database.connection_string
        readonly_config = ReadOnlyConnection.get_readonly_config(self.config.database.dialect)
        
        # Merge readonly config with engine config
        engine_config = {
            "pool_size": self.config.database.pool_size,
            "max_overflow": self.config.database.max_overflow,
            "echo": self.config.database.echo,
            **readonly_config
        }
        
        self.engine = create_engine(connection_string, **engine_config)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

class SchemaTools:
    """Schema introspection and management tools"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def get_all_tables(self) -> List[str]:
        """Get all table names in the database"""
        with self.db_manager.get_connection() as conn:
            inspector = inspect(self.db_manager.engine)
            return inspector.get_table_names()
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get detailed schema for a specific table"""
        with self.db_manager.get_connection() as conn:
            inspector = inspect(self.db_manager.engine)
            
            # Get columns
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "default": column["default"],
                    "primary_key": column.get("primary_key", False)
                })
            
            # Get indexes
            indexes = []
            for index in inspector.get_indexes(table_name):
                indexes.append({
                    "name": index["name"],
                    "columns": index["column_names"],
                    "unique": index["unique"]
                })
            
            # Get foreign keys
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    "name": fk["name"],
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"]
                })
            
            return {
                "table_name": table_name,
                "columns": columns,
                "indexes": indexes,
                "foreign_keys": foreign_keys
            }
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Get complete database schema"""
        tables = self.get_all_tables()
        schema = {}
        
        for table in tables:
            schema[table] = self.get_table_schema(table)
        
        return schema

class SQLExecutionTools:
    """SQL execution tools with safety and EXPLAIN capabilities"""
    
    def __init__(self, db_manager: DatabaseManager, config):
        self.db_manager = db_manager
        self.config = config
        self.safety_checker = SafetyChecker(config)
    
    def safe_execute_sql(self, sql: str, dialect: str = "sqlite") -> Dict[str, Any]:
        """Execute SQL safely with policy enforcement"""
        start_time = time.time()
        
        # Safety validation
        is_safe, violations = self.safety_checker.check_query_safety(sql, dialect)
        if not is_safe:
            return {
                "success": False,
                "error": f"Safety violations: {', '.join(violations)}",
                "execution_time": time.time() - start_time
            }
        
        # Inject safety guards
        safe_sql = self.safety_checker.inject_safety_guards(sql, dialect)
        
        try:
            with self.db_manager.get_connection() as conn:
                # Execute query
                result = conn.execute(text(safe_sql))
                
                # Fetch results (with row limit)
                rows = result.fetchmany(self.config.policy.max_result_rows)
                column_names = list(result.keys()) if result.keys() else []
                
                execution_time = time.time() - start_time
                
                return {
                    "success": True,
                    "data": [dict(zip(column_names, row)) for row in rows],
                    "execution_time": execution_time,
                    "row_count": len(rows),
                    "column_names": column_names,
                    "sql": safe_sql
                }
        
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "sql": safe_sql
            }
    
    def explain_query(self, sql: str, dialect: str = "sqlite") -> Dict[str, Any]:
        """Get query execution plan using EXPLAIN"""
        try:
            # Generate EXPLAIN SQL based on dialect
            if dialect == "postgresql":
                explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
            elif dialect == "mysql":
                explain_sql = f"EXPLAIN FORMAT=JSON {sql}"
            else:
                explain_sql = f"EXPLAIN QUERY PLAN {sql}"
            
            with self.db_manager.get_connection() as conn:
                result = conn.execute(text(explain_sql))
                plan = result.fetchall()
                
                return {
                    "dialect": dialect,
                    "plan": plan,
                    "raw_sql": explain_sql,
                    "success": True
                }
        
        except SQLAlchemyError as e:
            return {
                "dialect": dialect,
                "error": str(e),
                "raw_sql": explain_sql,
                "success": False
            }
    
    def dry_run_query(self, sql: str, dialect: str = "sqlite") -> Dict[str, Any]:
        """Perform a dry run of the query without executing it"""
        try:
            # Parse SQL to validate syntax
            parsed = sqlglot.parse(sql, dialect)
            
            # Get execution plan
            explain_result = self.explain_query(sql, dialect)
            
            return {
                "success": True,
                "syntax_valid": True,
                "parsed_statements": len(parsed),
                "execution_plan": explain_result,
                "estimated_cost": self._estimate_query_cost(sql, dialect)
            }
        
        except Exception as e:
            return {
                "success": False,
                "syntax_valid": False,
                "error": str(e)
            }
    
    def _estimate_query_cost(self, sql: str, dialect: str) -> Dict[str, Any]:
        """Estimate query cost based on heuristics"""
        cost_estimate = {
            "complexity_score": 0,
            "estimated_rows": 0,
            "estimated_time": 0
        }
        
        # Simple heuristics
        sql_upper = sql.upper()
        
        # Count JOINs
        join_count = sql_upper.count("JOIN")
        cost_estimate["complexity_score"] += join_count * 10
        
        # Count subqueries
        subquery_count = sql_upper.count("SELECT") - 1
        cost_estimate["complexity_score"] += subquery_count * 20
        
        # Count GROUP BY
        group_by_count = sql_upper.count("GROUP BY")
        cost_estimate["complexity_score"] += group_by_count * 5
        
        # Estimate execution time (very rough)
        cost_estimate["estimated_time"] = cost_estimate["complexity_score"] * 0.1  # seconds
        
        return cost_estimate

class CoreTools:
    """Main tools interface that combines all tool functionality"""
    
    def __init__(self, config):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.schema_tools = SchemaTools(self.db_manager)
        self.sql_tools = SQLExecutionTools(self.db_manager, config)
    
    def get_schema_for_tables(self, table_names: List[str]) -> Dict[str, Any]:
        """Get schema for specific tables"""
        schema = {}
        for table in table_names:
            try:
                schema[table] = self.schema_tools.get_table_schema(table)
            except Exception as e:
                schema[table] = {"error": str(e)}
        return schema
    
    def validate_and_execute(self, sql: str, dialect: str) -> Dict[str, Any]:
        """Validate and execute SQL with full safety checks"""
        return self.sql_tools.safe_execute_sql(sql, dialect)
    
    def get_query_plan(self, sql: str, dialect: str) -> Dict[str, Any]:
        """Get execution plan for SQL query"""
        return self.sql_tools.explain_query(sql, dialect)
    
    def dry_run(self, sql: str, dialect: str) -> Dict[str, Any]:
        """Perform dry run of query"""
        return self.sql_tools.dry_run_query(sql, dialect)
