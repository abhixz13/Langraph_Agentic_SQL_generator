from typing import List, Dict, Any, Optional, Tuple
import re
import sqlglot
from sqlglot import exp

class SafetyChecker:
    """Safety checker for SQL queries with DDL/DML blocklist and LIMIT injection"""
    
    def __init__(self, config):
        self.config = config
        self.blocked_keywords = config.policy.blocked_keywords
        self.blocked_patterns = config.policy.blocked_patterns
        self.max_result_rows = config.policy.max_result_rows
    
    def check_query_safety(self, sql: str, dialect: str = "sqlite") -> Tuple[bool, List[str]]:
        """Comprehensive safety check for SQL queries"""
        violations = []
        
        # Check for blocked keywords
        keyword_violations = self._check_blocked_keywords(sql)
        violations.extend(keyword_violations)
        
        # Check for blocked patterns
        pattern_violations = self._check_blocked_patterns(sql)
        violations.extend(pattern_violations)
        
        # Check for dangerous operations
        operation_violations = self._check_dangerous_operations(sql, dialect)
        violations.extend(operation_violations)
        
        # Check query complexity
        complexity_violations = self._check_query_complexity(sql)
        violations.extend(complexity_violations)
        
        return len(violations) == 0, violations
    
    def _check_blocked_keywords(self, sql: str) -> List[str]:
        """Check for blocked SQL keywords"""
        violations = []
        sql_upper = sql.upper()
        
        for keyword in self.blocked_keywords:
            if keyword in sql_upper:
                violations.append(f"Blocked keyword: {keyword}")
        
        return violations
    
    def _check_blocked_patterns(self, sql: str) -> List[str]:
        """Check for blocked SQL patterns using regex"""
        violations = []
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                violations.append(f"Blocked pattern: {pattern}")
        
        return violations
    
    def _check_dangerous_operations(self, sql: str, dialect: str) -> List[str]:
        """Check for dangerous operations beyond simple keyword matching"""
        violations = []
        
        try:
            # Parse SQL to understand structure
            parsed = sqlglot.parse(sql, dialect)
            
            for statement in parsed:
                # Check for DDL operations
                if isinstance(statement, exp.Create):
                    violations.append("CREATE operations are not allowed")
                
                if isinstance(statement, exp.Drop):
                    violations.append("DROP operations are not allowed")
                
                if isinstance(statement, exp.Alter):
                    violations.append("ALTER operations are not allowed")
                
                # Check for DML operations
                if isinstance(statement, exp.Insert):
                    violations.append("INSERT operations are not allowed")
                
                if isinstance(statement, exp.Update):
                    violations.append("UPDATE operations are not allowed")
                
                if isinstance(statement, exp.Delete):
                    violations.append("DELETE operations are not allowed")
                
                # Check for system operations
                if isinstance(statement, exp.Grant):
                    violations.append("GRANT operations are not allowed")
                
                if isinstance(statement, exp.Revoke):
                    violations.append("REVOKE operations are not allowed")
                
        except Exception as e:
            violations.append(f"SQL parsing error: {str(e)}")
        
        return violations
    
    def _check_query_complexity(self, sql: str) -> List[str]:
        """Check query complexity to prevent resource exhaustion"""
        violations = []
        
        # Simple complexity heuristics
        complexity_score = 0
        
        # Count JOINs
        join_count = sql.upper().count("JOIN")
        complexity_score += join_count * 10
        
        # Count subqueries
        subquery_count = sql.upper().count("SELECT") - 1
        complexity_score += subquery_count * 20
        
        # Count UNIONs
        union_count = sql.upper().count("UNION")
        complexity_score += union_count * 15
        
        # Count GROUP BY
        group_by_count = sql.upper().count("GROUP BY")
        complexity_score += group_by_count * 5
        
        if complexity_score > self.config.policy.max_query_complexity:
            violations.append(f"Query too complex (score: {complexity_score})")
        
        return violations
    
    def inject_safety_guards(self, sql: str, dialect: str = "sqlite") -> str:
        """Inject safety guards into SQL query"""
        try:
            # Parse SQL
            parsed = sqlglot.parse(sql, dialect)
            
            for statement in parsed:
                # Add LIMIT to SELECT statements if not present
                if isinstance(statement, exp.Select) and not statement.args.get("limit"):
                    statement.set("limit", exp.Literal.number(self.max_result_rows))
                
                # Add timeout hints for supported dialects
                if dialect == "postgresql":
                    # Add statement timeout
                    timeout_hint = f"/*+ SET statement_timeout = {self.config.policy.max_execution_time * 1000} */"
                    sql = f"{timeout_hint} {sql}"
                
                elif dialect == "mysql":
                    # Add query timeout
                    timeout_hint = f"/*+ SET SESSION MAX_EXECUTION_TIME = {self.config.policy.max_execution_time * 1000} */"
                    sql = f"{timeout_hint} {sql}"
            
            return sql
            
        except Exception:
            # If parsing fails, return original SQL with basic LIMIT
            if "SELECT" in sql.upper() and "LIMIT" not in sql.upper():
                sql += f" LIMIT {self.max_result_rows}"
            return sql
    
    def make_readonly(self, sql: str, dialect: str = "sqlite") -> str:
        """Convert query to read-only mode"""
        try:
            # Parse SQL
            parsed = sqlglot.parse(sql, dialect)
            
            for statement in parsed:
                # For PostgreSQL, add read-only transaction
                if dialect == "postgresql":
                    sql = f"BEGIN READ ONLY; {sql}; COMMIT;"
                
                # For MySQL, add read-only session
                elif dialect == "mysql":
                    sql = f"SET SESSION TRANSACTION READ ONLY; {sql};"
                
                # For SQLite, rely on connection-level read-only
                elif dialect == "sqlite":
                    # SQLite doesn't have session-level read-only, rely on connection
                    pass
            
            return sql
            
        except Exception:
            # If parsing fails, return original SQL
            return sql

class ReadOnlyConnection:
    """Helper for creating read-only database connections"""
    
    @staticmethod
    def get_readonly_config(dialect: str) -> Dict[str, Any]:
        """Get configuration for read-only connections"""
        if dialect == "postgresql":
            return {
                "connect_args": {"options": "-c default_transaction_read_only=on"}
            }
        elif dialect == "mysql":
            return {
                "connect_args": {"init_command": "SET SESSION TRANSACTION READ ONLY"}
            }
        elif dialect == "sqlite":
            return {
                "connect_args": {"check_same_thread": False}
            }
        else:
            return {}
