# Phase 1 Implementation Plan: Core Foundation

## 🎯 **Phase 1 Objective**

**Primary Goal**: Establish the foundational infrastructure for the SQL Assistant using LangGraph, creating a robust, type-safe, and configurable system that can be extended with agents in subsequent phases.

**Success Criteria**:
- ✅ Complete development environment setup with all dependencies
- ✅ Functional state management system with type safety
- ✅ Configurable policy management system
- ✅ Core database tools for schema retrieval and safe execution
- ✅ Basic project structure ready for agent implementation
- ✅ All components tested and documented

## 🏗️ **Phase 1 Design Philosophy**

### **1. Modular Architecture**
- **Separation of Concerns**: Each component has a single, well-defined responsibility
- **Dependency Injection**: Components are loosely coupled and easily testable
- **Interface-First**: Define clear contracts before implementation

### **2. Type Safety & Validation**
- **Pydantic Models**: All data structures use Pydantic for validation
- **Type Hints**: Comprehensive type annotations throughout
- **Error Handling**: Graceful error handling with meaningful messages

### **3. Configuration-Driven**
- **YAML Configuration**: Human-readable configuration files
- **Environment Variables**: Sensitive data via environment variables
- **Default Values**: Sensible defaults with override capabilities

### **4. Database Agnostic**
- **Dialect Support**: SQLite, PostgreSQL, MySQL support from day one
- **Connection Pooling**: Efficient database connection management
- **Schema Abstraction**: Unified schema representation across dialects

## 📋 **Detailed Implementation Plan**

### **Step 1.1: Environment Setup (Day 1)**

#### **Objective**: Create a clean, isolated development environment with all necessary dependencies.

#### **Sub-steps**:

1. **Project Structure Creation**
   ```
   sql_assistant/
   ├── core/
   │   ├── __init__.py
   │   ├── state.py
   │   ├── config.py
   │   ├── tools.py
   │   └── memory.py
   ├── agents/
   │   └── __init__.py
   ├── workflows/
   │   └── __init__.py
   ├── prompts/
   │   └── __init__.py
   ├── ui/
   │   └── __init__.py
   ├── tests/
   │   ├── __init__.py
   │   ├── test_state.py
   │   ├── test_config.py
   │   └── test_tools.py
   ├── config/
   │   ├── policy.yaml
   │   └── database.yaml
   ├── data/
   │   ├── memory/
   │   └── checkpoints/
   ├── requirements.txt
   ├── requirements-dev.txt
   ├── pyproject.toml
   ├── .env.example
   ├── .gitignore
   └── README.md
   ```

2. **Virtual Environment Setup**
   - Create Python 3.11+ virtual environment
   - Install core dependencies
   - Setup development tools (pytest, black, flake8, mypy)

3. **Dependency Management**
   ```python
   # requirements.txt
   langgraph>=0.2.0
   langchain>=0.2.0
   langchain-openai>=0.1.0
   pydantic>=2.0.0
   pyyaml>=6.0
   sqlalchemy>=2.0.0
   psycopg2-binary>=2.9.0
   pymysql>=1.1.0
   sqlglot>=20.0.0
   python-dotenv>=1.0.0
   ```

4. **Configuration Files**
   - `.env.example` with all required environment variables
   - `pyproject.toml` for project metadata and tool configuration
   - `.gitignore` for Python and project-specific files

### **Step 1.2: State Schema Implementation (Day 2)**

#### **Objective**: Create a comprehensive, type-safe state management system that can handle the entire workflow lifecycle.

#### **Sub-steps**:

1. **Core State Models**
   ```python
   # core/state.py
   from typing import Optional, List, Dict, Any, Union
   from pydantic import BaseModel, Field, validator
   from enum import Enum
   from datetime import datetime
   import uuid
   ```

2. **Enums and Constants**
   ```python
   class QueryStatus(str, Enum):
       PLANNING = "planning"
       GENERATING = "generating"
       VALIDATING = "validating"
       EXECUTING = "executing"
       COMPLETED = "completed"
       ERROR = "error"
       HUMAN_APPROVAL = "human_approval"

   class SQLDialect(str, Enum):
       SQLITE = "sqlite"
       POSTGRESQL = "postgresql"
       MYSQL = "mysql"
   ```

3. **Data Models**
   ```python
   class SQLCandidate(BaseModel):
       sql: str
       confidence: float = Field(ge=0.0, le=1.0)
       dialect: SQLDialect
       explanation: str
       metadata: Dict[str, Any] = Field(default_factory=dict)
       created_at: datetime = Field(default_factory=datetime.now)

   class ValidationResult(BaseModel):
       is_valid: bool
       syntax_errors: List[str] = Field(default_factory=list)
       semantic_errors: List[str] = Field(default_factory=list)
       warnings: List[str] = Field(default_factory=list)
       suggestions: List[str] = Field(default_factory=list)
       validation_time: float = Field(ge=0.0)

   class ExecutionResult(BaseModel):
       success: bool
       data: Optional[List[Dict[str, Any]]] = None
       error: Optional[str] = None
       execution_time: Optional[float] = None
       row_count: Optional[int] = None
       column_names: Optional[List[str]] = None
   ```

4. **Main State Class**
   ```python
   class SQLAssistantState(BaseModel):
       # Core identifiers
       session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
       query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
       
       # User input
       user_query: str
       dialect: SQLDialect = SQLDialect.SQLITE
       
       # Workflow state
       status: QueryStatus = QueryStatus.PLANNING
       created_at: datetime = Field(default_factory=datetime.now)
       updated_at: datetime = Field(default_factory=datetime.now)
       
       # Configuration
       gen_k: int = Field(default=3, ge=1, le=10)
       max_loops: int = Field(default=3, ge=1, le=10)
       loop_count: int = Field(default=0, ge=0)
       
       # Planning results
       intent: Optional[Dict[str, Any]] = None
       schema_info: Optional[Dict[str, Any]] = None
       execution_plan: Optional[Dict[str, Any]] = None
       ambiguity_detected: bool = False
       
       # SQL generation results
       sql_candidates: List[SQLCandidate] = Field(default_factory=list)
       selected_sql: Optional[str] = None
       
       # Validation results
       validation_result: Optional[ValidationResult] = None
       
       # Execution results
       execution_result: Optional[ExecutionResult] = None
       
       # Error handling
       error_message: Optional[str] = None
       error_suggestions: List[str] = Field(default_factory=list)
       
       # Memory and context
       conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
       previous_fixes: List[Dict[str, Any]] = Field(default_factory=list)
       
       # Human-in-the-loop
       requires_human_approval: bool = False
       human_feedback: Optional[str] = None
       
       # Performance tracking
       total_tokens_used: int = Field(default=0, ge=0)
       total_execution_time: float = Field(default=0.0, ge=0.0)
       
       @validator('updated_at', pre=True, always=True)
       def update_timestamp(cls, v):
           return datetime.now()
   ```

5. **State Utilities**
   ```python
   class StateManager:
       @staticmethod
       def create_initial_state(user_query: str, dialect: SQLDialect = SQLDialect.SQLITE) -> SQLAssistantState:
           """Create initial state for a new query"""
           return SQLAssistantState(user_query=user_query, dialect=dialect)
       
       @staticmethod
       def update_status(state: SQLAssistantState, new_status: QueryStatus) -> SQLAssistantState:
           """Update state status and timestamp"""
           state.status = new_status
           state.updated_at = datetime.now()
           return state
       
       @staticmethod
       def add_sql_candidate(state: SQLAssistantState, candidate: SQLCandidate) -> SQLAssistantState:
           """Add a new SQL candidate to the state"""
           state.sql_candidates.append(candidate)
           return state
   ```

### **Step 1.3: Configuration Management (Day 3)**

#### **Objective**: Create a flexible, type-safe configuration system that supports multiple environments and easy policy management.

#### **Sub-steps**:

1. **Configuration Models**
   ```python
   # core/config.py
   from typing import Dict, List, Any, Optional
   from pydantic import BaseModel, Field, validator
   import yaml
   import os
   from pathlib import Path
   ```

2. **Policy Configuration**
   ```python
   class PolicyConfig(BaseModel):
       # Generation settings
       gen_k: int = Field(default=3, ge=1, le=10, description="Number of SQL candidates to generate")
       max_loops: int = Field(default=3, ge=1, le=10, description="Maximum validation loops")
       
       # Safety caps
       max_query_complexity: int = Field(default=100, ge=1, description="Maximum query complexity score")
       max_execution_time: int = Field(default=30, ge=1, description="Maximum execution time in seconds")
       max_result_rows: int = Field(default=1000, ge=1, description="Maximum result rows")
       max_tokens_per_query: int = Field(default=4000, ge=1000, description="Maximum tokens per query")
       
       # Blocklist
       blocked_keywords: List[str] = Field(
           default=["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"],
           description="Blocked SQL keywords"
       )
       blocked_patterns: List[str] = Field(
           default=[],
           description="Blocked SQL patterns (regex)"
       )
       
       # Dialect-specific settings
       dialect_configs: Dict[str, Dict[str, Any]] = Field(
           default={
               "sqlite": {"case_sensitive": False, "supports_cte": True},
               "postgresql": {"case_sensitive": True, "supports_cte": True},
               "mysql": {"case_sensitive": False, "supports_cte": True}
           },
           description="Dialect-specific configurations"
       )
       
       # Validation settings
       enable_syntax_validation: bool = Field(default=True, description="Enable SQL syntax validation")
       enable_semantic_validation: bool = Field(default=True, description="Enable SQL semantic validation")
       enable_performance_analysis: bool = Field(default=True, description="Enable performance analysis")
       
       @validator('blocked_keywords')
       def validate_blocked_keywords(cls, v):
           return [kw.upper() for kw in v]
   ```

3. **Database Configuration**
   ```python
   class DatabaseConfig(BaseModel):
       host: str = Field(default="localhost")
       port: int = Field(default=5432, ge=1, le=65535)
       database: str
       username: str
       password: str
       dialect: str = Field(default="postgresql")
       pool_size: int = Field(default=5, ge=1, le=20)
       max_overflow: int = Field(default=10, ge=0)
       echo: bool = Field(default=False, description="Enable SQL logging")
       
       @property
       def connection_string(self) -> str:
           """Generate database connection string"""
           if self.dialect == "sqlite":
               return f"sqlite:///{self.database}"
           elif self.dialect == "postgresql":
               return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
           elif self.dialect == "mysql":
               return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
           else:
               raise ValueError(f"Unsupported dialect: {self.dialect}")
   ```

4. **Configuration Manager**
   ```python
   class ConfigManager:
       def __init__(self, config_dir: str = "config"):
           self.config_dir = Path(config_dir)
           self.policy = self.load_policy()
           self.database = self.load_database_config()
       
       def load_policy(self) -> PolicyConfig:
           """Load policy configuration from YAML file"""
           policy_path = self.config_dir / "policy.yaml"
           if not policy_path.exists():
               return PolicyConfig()  # Use defaults
           
           with open(policy_path, 'r') as f:
               config_data = yaml.safe_load(f)
           return PolicyConfig(**config_data)
       
       def load_database_config(self) -> DatabaseConfig:
           """Load database configuration from YAML file"""
           db_path = self.config_dir / "database.yaml"
           if not db_path.exists():
               # Try environment variables
               return DatabaseConfig(
                   host=os.getenv("DB_HOST", "localhost"),
                   port=int(os.getenv("DB_PORT", "5432")),
                   database=os.getenv("DB_NAME", "sql_assistant"),
                   username=os.getenv("DB_USER", "postgres"),
                   password=os.getenv("DB_PASSWORD", ""),
                   dialect=os.getenv("DB_DIALECT", "postgresql")
               )
           
           with open(db_path, 'r') as f:
               config_data = yaml.safe_load(f)
           return DatabaseConfig(**config_data)
       
       def validate_query(self, sql: str) -> tuple[bool, List[str]]:
           """Check if query violates policy"""
           violations = []
           sql_upper = sql.upper()
           
           # Check blocked keywords
           for keyword in self.policy.blocked_keywords:
               if keyword in sql_upper:
                   violations.append(f"Blocked keyword: {keyword}")
           
           # Check blocked patterns
           import re
           for pattern in self.policy.blocked_patterns:
               if re.search(pattern, sql, re.IGNORECASE):
                   violations.append(f"Blocked pattern: {pattern}")
           
           return len(violations) == 0, violations
       
       def get_dialect_config(self, dialect: str) -> Dict[str, Any]:
           """Get dialect-specific configuration"""
           return self.policy.dialect_configs.get(dialect, {})
   ```

5. **Configuration Files**
   ```yaml
   # config/policy.yaml
   gen_k: 3
   max_loops: 3
   max_query_complexity: 100
   max_execution_time: 30
   max_result_rows: 1000
   max_tokens_per_query: 4000
   
   blocked_keywords:
     - DROP
     - DELETE
     - TRUNCATE
     - ALTER
     - CREATE
     - INSERT
     - UPDATE
   
   dialect_configs:
     sqlite:
       case_sensitive: false
       supports_cte: true
     postgresql:
       case_sensitive: true
       supports_cte: true
     mysql:
       case_sensitive: false
       supports_cte: true
   ```

### **Step 1.4: Core Tools Implementation (Day 4-5)**

#### **Objective**: Create database connection tools, schema retrieval tools, and safe SQL execution utilities that work across multiple dialects.

#### **Sub-steps**:

1. **Database Connection Management**
   ```python
   # core/tools.py
   from typing import Dict, List, Any, Optional, Union
   from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
   from sqlalchemy.engine import Engine
   from sqlalchemy.exc import SQLAlchemyError
   import sqlglot
   import time
   from contextlib import contextmanager
   ```

2. **Database Manager**
   ```python
   class DatabaseManager:
       def __init__(self, config: DatabaseConfig):
           self.config = config
           self.engine = None
           self.metadata = MetaData()
           self._initialize_engine()
       
       def _initialize_engine(self):
           """Initialize SQLAlchemy engine with connection pooling"""
           self.engine = create_engine(
               self.config.connection_string,
               pool_size=self.config.pool_size,
               max_overflow=self.config.max_overflow,
               echo=self.config.echo
           )
       
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
   ```

3. **Schema Retrieval Tools**
   ```python
   class SchemaTools:
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
   ```

4. **SQL Execution Tools**
   ```python
   class SQLExecutionTools:
       def __init__(self, db_manager: DatabaseManager, config: PolicyConfig):
           self.db_manager = db_manager
           self.config = config
       
       def safe_execute_sql(self, sql: str, dialect: str = "sqlite") -> ExecutionResult:
           """Execute SQL safely with policy enforcement"""
           start_time = time.time()
           
           # Policy validation
           is_valid, violations = self.config.validate_query(sql)
           if not is_valid:
               return ExecutionResult(
                   success=False,
                   error=f"Policy violations: {', '.join(violations)}",
                   execution_time=time.time() - start_time
               )
           
           # Add safety guards
           safe_sql = self.add_safety_guards(sql, dialect)
           
           try:
               with self.db_manager.get_connection() as conn:
                   # Execute query
                   result = conn.execute(text(safe_sql))
                   
                   # Fetch results (with row limit)
                   rows = result.fetchmany(self.config.max_result_rows)
                   column_names = list(result.keys()) if result.keys() else []
                   
                   execution_time = time.time() - start_time
                   
                   return ExecutionResult(
                       success=True,
                       data=[dict(zip(column_names, row)) for row in rows],
                       execution_time=execution_time,
                       row_count=len(rows),
                       column_names=column_names
                   )
           
           except SQLAlchemyError as e:
               execution_time = time.time() - start_time
               return ExecutionResult(
                   success=False,
                   error=str(e),
                   execution_time=execution_time
               )
       
       def add_safety_guards(self, sql: str, dialect: str) -> str:
           """Add safety guards to SQL query"""
           # Parse SQL to understand structure
           try:
               parsed = sqlglot.parse(sql, dialect)
               
               # Add LIMIT if not present and it's a SELECT
               if "SELECT" in sql.upper() and "LIMIT" not in sql.upper():
                   sql += f" LIMIT {self.config.max_result_rows}"
               
               # Add timeout hint if supported
               if dialect == "postgresql":
                   sql = f"/*+ SET statement_timeout = {self.config.max_execution_time * 1000} */ {sql}"
               
               return sql
           
           except Exception:
               # If parsing fails, return original SQL
               return sql
       
       def explain_query(self, sql: str, dialect: str = "sqlite") -> Dict[str, Any]:
           """Get query execution plan"""
           try:
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
                       "raw_sql": explain_sql
                   }
           
           except SQLAlchemyError as e:
               return {
                   "dialect": dialect,
                   "error": str(e),
                   "raw_sql": explain_sql
               }
   ```

5. **Tool Integration**
   ```python
   class CoreTools:
       def __init__(self, config_manager: ConfigManager):
           self.config_manager = config_manager
           self.db_manager = DatabaseManager(config_manager.database)
           self.schema_tools = SchemaTools(self.db_manager)
           self.sql_tools = SQLExecutionTools(self.db_manager, config_manager.policy)
       
       def get_schema_for_tables(self, table_names: List[str]) -> Dict[str, Any]:
           """Get schema for specific tables"""
           schema = {}
           for table in table_names:
               try:
                   schema[table] = self.schema_tools.get_table_schema(table)
               except Exception as e:
                   schema[table] = {"error": str(e)}
           return schema
       
       def validate_and_execute(self, sql: str, dialect: str) -> ExecutionResult:
           """Validate and execute SQL with full safety checks"""
           return self.sql_tools.safe_execute_sql(sql, dialect)
       
       def get_query_plan(self, sql: str, dialect: str) -> Dict[str, Any]:
           """Get execution plan for SQL query"""
           return self.sql_tools.explain_query(sql, dialect)
   ```

### **Step 1.5: Memory Management (Day 6)**

#### **Objective**: Create a memory system that can store and retrieve synonyms, prior fixes, and conversation context for improved query understanding.

#### **Sub-steps**:

1. **Memory Models**
   ```python
   # core/memory.py
   from typing import Dict, List, Any, Optional
   import json
   import hashlib
   from datetime import datetime, timedelta
   from pathlib import Path
   ```

2. **Memory Manager Implementation**
   ```python
   class MemoryManager:
       def __init__(self, storage_path: str = "data/memory"):
           self.storage_path = Path(storage_path)
           self.storage_path.mkdir(parents=True, exist_ok=True)
           
           self.synonyms_file = self.storage_path / "synonyms.json"
           self.fixes_file = self.storage_path / "fixes.json"
           self.conversations_file = self.storage_path / "conversations.json"
           
           self.synonyms = self.load_synonyms()
           self.prior_fixes = self.load_prior_fixes()
           self.conversations = self.load_conversations()
       
       def load_synonyms(self) -> Dict[str, List[str]]:
           """Load table/column synonyms"""
           if self.synonyms_file.exists():
               with open(self.synonyms_file, 'r') as f:
                   return json.load(f)
           return {}
       
       def save_synonyms(self):
           """Save synonyms to file"""
           with open(self.synonyms_file, 'w') as f:
               json.dump(self.synonyms, f, indent=2)
       
       def load_prior_fixes(self) -> List[Dict[str, Any]]:
           """Load previous query fixes"""
           if self.fixes_file.exists():
               with open(self.fixes_file, 'r') as f:
                   return json.load(f)
           return []
       
       def save_prior_fixes(self):
           """Save prior fixes to file"""
           with open(self.fixes_file, 'w') as f:
               json.dump(self.prior_fixes, f, indent=2)
       
       def load_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
           """Load conversation history"""
           if self.conversations_file.exists():
               with open(self.conversations_file, 'r') as f:
                   return json.load(f)
           return {}
       
       def save_conversations(self):
           """Save conversation history to file"""
           with open(self.conversations_file, 'w') as f:
               json.dump(self.conversations, f, indent=2)
       
       def get_synonyms(self, term: str) -> List[str]:
           """Get synonyms for a term"""
           return self.synonyms.get(term.lower(), [])
       
       def add_synonym(self, term: str, synonyms: List[str]):
           """Add synonyms for a term"""
           term_lower = term.lower()
           if term_lower not in self.synonyms:
               self.synonyms[term_lower] = []
           self.synonyms[term_lower].extend(synonyms)
           self.save_synonyms()
       
       def add_fix(self, original_query: str, fixed_sql: str, fix_type: str, success: bool = True):
           """Record a successful fix"""
           fix_hash = hashlib.md5(original_query.encode()).hexdigest()
           fix_record = {
               "hash": fix_hash,
               "original_query": original_query,
               "fixed_sql": fixed_sql,
               "fix_type": fix_type,
               "success": success,
               "timestamp": datetime.now().isoformat(),
               "usage_count": 0
           }
           
           # Check if similar fix already exists
           existing_fix = next((f for f in self.prior_fixes if f["hash"] == fix_hash), None)
           if existing_fix:
               existing_fix["usage_count"] += 1
               existing_fix["last_used"] = datetime.now().isoformat()
           else:
               self.prior_fixes.append(fix_record)
           
           self.save_prior_fixes()
       
       def find_similar_fixes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
           """Find similar previous fixes"""
           query_hash = hashlib.md5(query.encode()).hexdigest()
           similar_fixes = [fix for fix in self.prior_fixes if fix["hash"] == query_hash]
           
           # Sort by usage count and recency
           similar_fixes.sort(key=lambda x: (x.get("usage_count", 0), x.get("timestamp", "")), reverse=True)
           return similar_fixes[:limit]
       
       def add_conversation_entry(self, session_id: str, entry: Dict[str, Any]):
           """Add conversation entry for a session"""
           if session_id not in self.conversations:
               self.conversations[session_id] = []
           
           entry["timestamp"] = datetime.now().isoformat()
           self.conversations[session_id].append(entry)
           
           # Keep only last 50 entries per session
           if len(self.conversations[session_id]) > 50:
               self.conversations[session_id] = self.conversations[session_id][-50:]
           
           self.save_conversations()
       
       def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
           """Get conversation history for a session"""
           if session_id not in self.conversations:
               return []
           
           return self.conversations[session_id][-limit:]
       
       def cleanup_old_data(self, days: int = 30):
           """Clean up old conversation data"""
           cutoff_date = datetime.now() - timedelta(days=days)
           
           # Clean up old conversations
           for session_id in list(self.conversations.keys()):
               self.conversations[session_id] = [
                   entry for entry in self.conversations[session_id]
                   if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
               ]
               
               if not self.conversations[session_id]:
                   del self.conversations[session_id]
           
           self.save_conversations()
   ```

### **Step 1.6: Testing and Documentation (Day 7)**

#### **Objective**: Create comprehensive tests for all components and document the system for future development.

#### **Sub-steps**:

1. **Unit Tests**
   ```python
   # tests/test_state.py
   import pytest
   from core.state import SQLAssistantState, QueryStatus, SQLDialect, StateManager
   
   def test_state_creation():
       """Test creating initial state"""
       state = SQLAssistantState(user_query="SELECT * FROM users")
       assert state.user_query == "SELECT * FROM users"
       assert state.status == QueryStatus.PLANNING
       assert state.dialect == SQLDialect.SQLITE
       assert state.loop_count == 0
   
   def test_state_validation():
       """Test state validation"""
       # Test invalid gen_k
       with pytest.raises(ValueError):
           SQLAssistantState(user_query="test", gen_k=0)
       
       # Test invalid max_loops
       with pytest.raises(ValueError):
           SQLAssistantState(user_query="test", max_loops=0)
   ```

2. **Configuration Tests**
   ```python
   # tests/test_config.py
   import pytest
   from core.config import ConfigManager, PolicyConfig, DatabaseConfig
   
   def test_policy_validation():
       """Test policy validation"""
       config = ConfigManager()
       
       # Test valid query
       is_valid, violations = config.validate_query("SELECT * FROM users")
       assert is_valid
       assert len(violations) == 0
       
       # Test blocked keyword
       is_valid, violations = config.validate_query("DROP TABLE users")
       assert not is_valid
       assert len(violations) > 0
   ```

3. **Tools Tests**
   ```python
   # tests/test_tools.py
   import pytest
   from core.tools import DatabaseManager, SchemaTools, SQLExecutionTools
   from core.config import DatabaseConfig, PolicyConfig
   
   def test_database_connection():
       """Test database connection"""
       config = DatabaseConfig(
           dialect="sqlite",
           database=":memory:"
       )
       db_manager = DatabaseManager(config)
       assert db_manager.test_connection()
   ```

4. **Integration Tests**
   ```python
   # tests/test_integration.py
   import pytest
   from core.state import SQLAssistantState
   from core.config import ConfigManager
   from core.tools import CoreTools
   from core.memory import MemoryManager
   
   def test_full_workflow():
       """Test complete workflow integration"""
       # Initialize components
       config_manager = ConfigManager()
       tools = CoreTools(config_manager)
       memory = MemoryManager()
       
       # Create state
       state = SQLAssistantState(user_query="SELECT * FROM users")
       
       # Test schema retrieval
       schema = tools.get_schema_for_tables(["users"])
       assert isinstance(schema, dict)
   ```

5. **Documentation**
   ```markdown
   # README.md
   # SQL Assistant - Phase 1 Implementation
   
   ## Overview
   This is the core foundation implementation for the SQL Assistant using LangGraph.
   
   ## Installation
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
   ## Configuration
   1. Copy `.env.example` to `.env`
   2. Update environment variables
   3. Configure `config/policy.yaml` and `config/database.yaml`
   
   ## Usage
   ```python
   from core.config import ConfigManager
   from core.tools import CoreTools
   from core.memory import MemoryManager
   
   # Initialize components
   config = ConfigManager()
   tools = CoreTools(config)
   memory = MemoryManager()
   
   # Use components
   schema = tools.get_schema_for_tables(["users"])
   ```
   
   ## Testing
   ```bash
   pytest tests/
   ```
   ```

## 🎯 **Phase 1 Success Metrics**

### **Functional Requirements**
- ✅ All core components implemented and tested
- ✅ State management system handles all workflow states
- ✅ Configuration system supports multiple environments
- ✅ Database tools work with SQLite, PostgreSQL, and MySQL
- ✅ Memory system stores and retrieves context effectively

### **Quality Requirements**
- ✅ 90%+ test coverage for all components
- ✅ Type safety with comprehensive type hints
- ✅ Error handling for all edge cases
- ✅ Documentation for all public APIs
- ✅ Performance benchmarks established

### **Technical Requirements**
- ✅ Modular architecture with clear separation of concerns
- ✅ Dependency injection for easy testing
- ✅ Configuration-driven behavior
- ✅ Database agnostic design
- ✅ Extensible foundation for future phases

## 🚀 **Next Steps After Phase 1**

1. **Phase 2 Preparation**: The foundation is ready for agent implementation
2. **Agent Development**: Can now build planning, SQL generation, and validation agents
3. **Workflow Integration**: Ready to create LangGraph workflows using the core components
4. **UI Development**: Can build interfaces that use the core tools and state management

This Phase 1 implementation provides a solid, production-ready foundation that follows best practices and is ready for the next phases of development.
