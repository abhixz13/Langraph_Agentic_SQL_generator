"""
Unit tests for the Schema Context Agent

This module tests the schema_context agent's ability to:
1. Reflect database schema from different sources
2. Load and integrate semantic context
3. Build optimized schema context
4. Handle caching (file-based and in-memory)
5. Work with different database dialects
6. Handle errors gracefully
"""

import unittest
import json
import tempfile
import os
import sqlite3
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any

from core.state import AppState
from agents.schema_context.subgraph import schema_context_node


class TestSchemaContextAgent(unittest.TestCase):
    """Test cases for the Schema Context Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary SQLite database for testing
        self.temp_db_path = tempfile.mktemp(suffix='.db')
        self.create_test_database()
        
        # Create test semantic directory
        self.temp_semantic_dir = tempfile.mkdtemp()
        self.create_test_semantic_files()
        
        # Test policy
        self.test_policy = {
            "schema_max_tables": 20,
            "schema_max_columns": 15,
            "schema_include_samples": True,
            "schema_sample_rows": 2,
            "execute_timeout_ms": 5000,
            "schema_cache_enabled": True
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary database
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
        
        # Remove temporary semantic directory
        if os.path.exists(self.temp_semantic_dir):
            import shutil
            shutil.rmtree(self.temp_semantic_dir)
    
    def create_test_database(self):
        """Create a test SQLite database with sample data"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Create test tables
        cursor.execute("""
            CREATE TABLE raw_data (
                CUSTOMER_ID INTEGER PRIMARY KEY,
                CUSTOMER_NAME TEXT NOT NULL,
                COUNTRY TEXT,
                INTERSIGHT_CONSUMPTION TEXT,
                CREATED_AT DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE users (
                USER_ID INTEGER PRIMARY KEY,
                USERNAME TEXT UNIQUE NOT NULL,
                EMAIL TEXT,
                CREATED_AT DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert sample data
        cursor.execute("""
            INSERT INTO raw_data (CUSTOMER_ID, CUSTOMER_NAME, COUNTRY, INTERSIGHT_CONSUMPTION)
            VALUES 
                (1, 'Acme Corp', 'USA', 'pApp'),
                (2, 'Tech Solutions', 'Canada', 'SaaS'),
                (3, 'Global Industries', 'UK', 'pApp'),
                (4, 'Startup Inc', 'USA', 'SaaS')
        """)
        
        cursor.execute("""
            INSERT INTO users (USER_ID, USERNAME, EMAIL)
            VALUES 
                (1, 'admin', 'admin@example.com'),
                (2, 'user1', 'user1@example.com'),
                (3, 'user2', 'user2@example.com')
        """)
        
        conn.commit()
        conn.close()
    
    def create_test_semantic_files(self):
        """Create test semantic context files"""
        # Create business_context.json
        business_context = {
            "raw_data": {
                "display_name": "Customer Data",
                "description": "Customer consumption and usage data",
                "business_purpose": "Track customer consumption patterns and usage metrics",
                "aliases": ["customers", "consumption_data", "customer_data"],
                "columns": {
                    "CUSTOMER_NAME": {
                        "alias": "Customer Name",
                        "meaning": "Name of the customer or client",
                        "description": "Full name of the customer",
                        "data_type": "TEXT",
                        "possible_values": ["Acme Corp", "Tech Solutions", "Global Industries"],
                        "business_context": "Primary identifier for customer records",
                        "synonyms": ["client", "customer_name", "account_name"],
                        "metrics": ["count", "unique_count"],
                        "relationships": [],
                        "business_rules": ["Must be unique per customer"]
                    },
                    "COUNTRY": {
                        "alias": "Country",
                        "meaning": "Country or region where customer is located",
                        "description": "Geographic location of the customer",
                        "data_type": "TEXT",
                        "possible_values": ["USA", "Canada", "UK", "Germany"],
                        "business_context": "Used for regional analysis and reporting",
                        "synonyms": ["location", "region", "territory"],
                        "metrics": ["count", "distribution"],
                        "relationships": [],
                        "business_rules": ["Must be a valid country code"]
                    },
                    "INTERSIGHT_CONSUMPTION": {
                        "alias": "Consumption Type",
                        "meaning": "Type of Intersight consumption or usage",
                        "description": "The specific type of service consumed",
                        "data_type": "TEXT",
                        "possible_values": ["pApp", "SaaS", "Infrastructure"],
                        "business_context": "Key metric for revenue analysis",
                        "synonyms": ["usage_type", "consumption_type", "service_type"],
                        "metrics": ["count", "revenue_impact"],
                        "relationships": [],
                        "business_rules": ["Must be one of predefined types"]
                    }
                }
            }
        }
        
        with open(os.path.join(self.temp_semantic_dir, "business_context.json"), "w") as f:
            json.dump(business_context, f, indent=2)
    
    def test_schema_context_node_success(self):
        """Test successful schema context generation"""
        print("\n🧪 Testing Schema Context Node (Success)...")
        
        # Create test state
        state = AppState(
            database_url=f"sqlite:///{self.temp_db_path}",
            dialect="sqlite",
            semantic_dir=self.temp_semantic_dir,
            policy=self.test_policy
        )
        
        # Test schema_context_node
        result = schema_context_node(state)
        
        print(f"Result Status: {result.get('status')}")
        print(f"Tables Found: {len(result.get('db_schema', {}).get('tables', {}))}")
        print(f"Tables in Context: {len(result.get('schema_context', {}).get('tables', {}))}")
        
        # Print detailed schema information
        db_schema = result.get('db_schema', {})
        schema_context = result.get('schema_context', {})
        
        print(f"\nDatabase Schema:")
        for table_name, table_info in db_schema.get('tables', {}).items():
            print(f"  {table_name}: {len(table_info.get('columns', {}))} columns")
        
        print(f"\nSchema Context:")
        for table_name, table_info in schema_context.get('tables', {}).items():
            print(f"  {table_name}: {len(table_info.get('columns', {}))} columns")
        
        # Assertions
        self.assertEqual(result["status"], "PLANNING")
        self.assertIsNone(result.get("error_message"))
        self.assertIn("raw_data", db_schema.get("tables", {}))
        self.assertIn("users", db_schema.get("tables", {}))
        
        # Check columns exist in the list format
        raw_data_columns = [col["name"] for col in db_schema["tables"]["raw_data"]["columns"]]
        self.assertIn("CUSTOMER_NAME", raw_data_columns)
        self.assertIn("COUNTRY", raw_data_columns)
        
        # Check metrics
        metrics = result.get("metrics", {})
        self.assertGreater(metrics.get("schema_context_ms", 0), 0)
        self.assertFalse(metrics.get("schema_cache_hit", True))  # Should be cache miss on first run
    
    def test_schema_context_node_with_cache(self):
        """Test schema context generation with caching"""
        print("\n🧪 Testing Schema Context Node (With Caching)...")
        
        # Create test state
        state = AppState(
            database_url=f"sqlite:///{self.temp_db_path}",
            dialect="sqlite",
            semantic_dir=self.temp_semantic_dir,
            policy=self.test_policy
        )
        
        # First run - should generate schema
        result1 = schema_context_node(state)
        print(f"First Run - Cache Hit: {result1.get('metrics', {}).get('schema_cache_hit', False)}")
        
        # Second run - should use cache
        result2 = schema_context_node(state)
        print(f"Second Run - Cache Hit: {result2.get('metrics', {}).get('schema_cache_hit', False)}")
        
        # Assertions
        self.assertEqual(result1["status"], "PLANNING")
        self.assertEqual(result2["status"], "PLANNING")
        self.assertFalse(result1.get("metrics", {}).get("schema_cache_hit", True))  # First run should miss
        self.assertTrue(result2.get("metrics", {}).get("schema_cache_hit", False))  # Second run should hit
        
        # Schema should be identical
        self.assertEqual(
            len(result1.get("db_schema", {}).get("tables", {})),
            len(result2.get("db_schema", {}).get("tables", {}))
        )
    
    def test_schema_context_node_error_handling(self):
        """Test error handling in schema context generation"""
        print("\n🧪 Testing Schema Context Node (Error Handling)...")
        
        # Create test state with invalid database URL
        state = AppState(
            database_url="sqlite:///nonexistent_database.db",
            dialect="sqlite",
            semantic_dir=self.temp_semantic_dir,
            policy=self.test_policy
        )
        
        # Test schema_context_node
        result = schema_context_node(state)
        
        print(f"Result Status: {result.get('status')}")
        print(f"Error Message: {result.get('error_message')}")
        
        # Assertions
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNotNone(result["error_message"])
        self.assertIn("tables", result.get("db_schema", {}))
        self.assertEqual(len(result["db_schema"]["tables"]), 0)  # Should be empty on error
    
    def test_schema_context_node_without_semantic(self):
        """Test schema context generation without semantic files"""
        print("\n🧪 Testing Schema Context Node (Without Semantic)...")
        
        # Create test state without semantic directory
        state = AppState(
            database_url=f"sqlite:///{self.temp_db_path}",
            dialect="sqlite",
            semantic_dir="/nonexistent/semantic/dir",
            policy=self.test_policy
        )
        
        # Test schema_context_node
        result = schema_context_node(state)
        
        print(f"Result Status: {result.get('status')}")
        print(f"Tables Found: {len(result.get('db_schema', {}).get('tables', {}))}")
        
        # Assertions
        self.assertEqual(result["status"], "PLANNING")
        self.assertIsNone(result.get("error_message"))
        self.assertIn("raw_data", result.get("db_schema", {}).get("tables", {}))
        self.assertIn("users", result.get("db_schema", {}).get("tables", {}))
        
        # Should still work without semantic context
        schema_context = result.get("schema_context", {})
        self.assertIn("raw_data", schema_context.get("tables", {}))
    
    def test_schema_context_node_policy_configuration(self):
        """Test schema context generation with different policy configurations"""
        print("\n🧪 Testing Schema Context Node (Policy Configuration)...")
        
        # Test with restrictive policy
        restrictive_policy = {
            "schema_max_tables": 1,  # Only 1 table
            "schema_max_columns": 2,  # Only 2 columns per table
            "schema_include_samples": False,
            "schema_sample_rows": 0,
            "execute_timeout_ms": 5000,
            "schema_cache_enabled": False
        }
        
        state = AppState(
            database_url=f"sqlite:///{self.temp_db_path}",
            dialect="sqlite",
            semantic_dir=self.temp_semantic_dir,
            policy=restrictive_policy
        )
        
        result = schema_context_node(state)
        
        print(f"Restrictive Policy Result:")
        print(f"  Tables in DB Schema: {len(result.get('db_schema', {}).get('tables', {}))}")
        print(f"  Tables in Context: {len(result.get('schema_context', {}).get('tables', {}))}")
        
        # Check first table columns
        first_table = list(result.get('schema_context', {}).get('tables', {}).values())[0]
        print(f"  Columns in First Table: {len(first_table.get('columns', {}))}")
        
        # Assertions
        self.assertEqual(result["status"], "PLANNING")
        self.assertLessEqual(len(result.get("schema_context", {}).get("tables", {})), 1)
        
        # Check column limits
        for table_info in result.get("schema_context", {}).get("tables", {}).values():
            self.assertLessEqual(len(table_info.get("columns", {})), 2)


class TestSchemaContextIntegration(unittest.TestCase):
    """Test schema context integration with interpret plan"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary SQLite database
        self.temp_db_path = tempfile.mktemp(suffix='.db')
        self.create_test_database()
        
        # Create test semantic directory
        self.temp_semantic_dir = tempfile.mkdtemp()
        self.create_test_semantic_files()
        
        self.test_policy = {
            "schema_max_tables": 20,
            "schema_max_columns": 15,
            "schema_include_samples": True,
            "schema_sample_rows": 2,
            "execute_timeout_ms": 5000,
            "schema_cache_enabled": True,
            "intent": {
                "min_confidence": 0.3,
                "default_confidence": 0.5,
                "fallback_confidence": 0.1,
                "default_action": "SELECT",
                "valid_actions": ["SELECT", "COUNT", "AGGREGATE", "SEARCH", "COMPARE"],
                "max_tables": 8,
                "max_columns_per_table": 10
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
        if os.path.exists(self.temp_semantic_dir):
            import shutil
            shutil.rmtree(self.temp_semantic_dir)
    
    def create_test_database(self):
        """Create a test SQLite database"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE raw_data (
                CUSTOMER_ID INTEGER PRIMARY KEY,
                CUSTOMER_NAME TEXT NOT NULL,
                COUNTRY TEXT,
                INTERSIGHT_CONSUMPTION TEXT,
                CREATED_AT DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            INSERT INTO raw_data (CUSTOMER_ID, CUSTOMER_NAME, COUNTRY, INTERSIGHT_CONSUMPTION)
            VALUES 
                (1, 'Acme Corp', 'USA', 'pApp'),
                (2, 'Tech Solutions', 'Canada', 'SaaS')
        """)
        
        conn.commit()
        conn.close()
    
    def create_test_semantic_files(self):
        """Create test semantic context files"""
        business_context = {
            "raw_data": {
                "display_name": "Customer Data",
                "description": "Customer consumption and usage data",
                "aliases": ["customers", "consumption_data"],
                "columns": {
                    "CUSTOMER_NAME": {
                        "alias": "Customer Name",
                        "meaning": "Name of the customer or client",
                        "synonyms": ["client", "customer_name"]
                    },
                    "COUNTRY": {
                        "alias": "Country",
                        "meaning": "Country or region where customer is located",
                        "synonyms": ["location", "region"]
                    }
                }
            }
        }
        
        with open(os.path.join(self.temp_semantic_dir, "business_context.json"), "w") as f:
            json.dump(business_context, f, indent=2)
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_schema_context_to_interpret_plan_integration(self, mock_llm_adapter):
        """Test integration between schema context and interpret plan"""
        print("\n🧪 Testing Schema Context to Interpret Plan Integration...")
        
        from agents.interpret_plan.subgraph import interpret_plan_node
        
        # Mock LLM response
        mock_llm_client = MagicMock()
        mock_llm_client.interpret_query_intent.return_value = {
            "action": "SELECT",
            "tables": ["raw_data"],
            "columns": ["CUSTOMER_NAME", "COUNTRY"],
            "conditions": ["COUNTRY = 'USA'"],
            "ambiguity_detected": False,
            "clarifying_questions": [],
            "complexity_level": "simple",
            "confidence": 0.85
        }
        mock_llm_adapter.return_value = mock_llm_client
        
        # Step 1: Generate schema context
        schema_state = AppState(
            database_url=f"sqlite:///{self.temp_db_path}",
            dialect="sqlite",
            semantic_dir=self.temp_semantic_dir,
            policy=self.test_policy
        )
        
        schema_result = schema_context_node(schema_state)
        print(f"Schema Context Status: {schema_result.get('status')}")
        print(f"Tables in Schema: {list(schema_result.get('schema_context', {}).get('tables', {}).keys())}")
        
        # Step 2: Use schema context for interpret plan
        interpret_state = AppState(
            user_query="show me all customers from USA",
            schema_context=schema_result.get("schema_context"),
            policy=self.test_policy
        )
        
        interpret_result = interpret_plan_node(interpret_state)
        print(f"Interpret Plan Status: {interpret_result.get('status')}")
        print(f"Intent: {json.dumps(interpret_result.get('intent_json', {}), indent=2)}")
        
        # Assertions
        self.assertEqual(schema_result["status"], "PLANNING")
        self.assertEqual(interpret_result["status"], "PLANNING")
        self.assertIn("raw_data", schema_result.get("schema_context", {}).get("tables", {}))
        self.assertEqual(interpret_result["intent_json"]["action"], "SELECT")
        self.assertEqual(interpret_result["intent_json"]["tables"], ["raw_data"])


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
