"""
Unit tests for the Interpret Plan Agent

This module tests the interpret_plan agent's ability to:
1. Generate intent from natural language queries
2. Handle different query types (SELECT, COUNT, AGGREGATE, etc.)
3. Detect ambiguity and generate clarifying questions
4. Work with schema prefiltering
5. Use policy-based configuration
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from core.state import AppState, create_success_response, create_error_response
from agents.interpret_plan.subgraph import (
    interpret_plan_node,
    prefilter_schema,
    validate_intent_response,
    _get_fallback_intent
)


class TestInterpretPlanAgent(unittest.TestCase):
    """Test cases for the Interpret Plan Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock schema context for testing
        self.mock_schema_context = {
            "tables": {
                "raw_data": {
                    "name": "raw_data",
                    "description": "Customer consumption data",
                    "row_count_estimate": 1000,
                    "columns": {
                        "CUSTOMER_NAME": {
                            "name": "CUSTOMER_NAME",
                            "data_type": "TEXT",
                            "aliases": ["customer", "name", "client_name"],
                            "is_primary_key": False,
                            "is_foreign_key": False
                        },
                        "COUNTRY": {
                            "name": "COUNTRY",
                            "data_type": "TEXT",
                            "aliases": ["country", "location", "region"],
                            "is_primary_key": False,
                            "is_foreign_key": False
                        },
                        "INTERSIGHT_CONSUMPTION": {
                            "name": "INTERSIGHT_CONSUMPTION",
                            "data_type": "TEXT",
                            "aliases": ["consumption", "usage", "intersight"],
                            "is_primary_key": False,
                            "is_foreign_key": False
                        },
                        "CUSTOMER_ID": {
                            "name": "CUSTOMER_ID",
                            "data_type": "INTEGER",
                            "aliases": ["id", "customer_id"],
                            "is_primary_key": True,
                            "is_foreign_key": False
                        },
                        "CREATED_AT": {
                            "name": "CREATED_AT",
                            "data_type": "DATETIME",
                            "aliases": ["created", "date", "timestamp"],
                            "is_primary_key": False,
                            "is_foreign_key": False
                        }
                    }
                },
                "users": {
                    "name": "users",
                    "description": "User accounts",
                    "row_count_estimate": 500,
                    "columns": {
                        "USER_ID": {
                            "name": "USER_ID",
                            "data_type": "INTEGER",
                            "aliases": ["id", "user_id"],
                            "is_primary_key": True,
                            "is_foreign_key": False
                        },
                        "USERNAME": {
                            "name": "USERNAME",
                            "data_type": "TEXT",
                            "aliases": ["username", "login", "name"],
                            "is_primary_key": False,
                            "is_foreign_key": False
                        }
                    }
                }
            },
            "semantic_schema": {
                "raw_data": {
                    "aliases": ["customers", "consumption_data", "customer_data"],
                    "description": "Customer consumption and usage data",
                    "business_purpose": "Track customer consumption patterns and usage metrics",
                    "columns": {
                        "CUSTOMER_NAME": {
                            "aliases": ["customer", "name", "client_name"],
                            "meaning": "Name of the customer or client",
                            "synonyms": ["client", "customer_name", "account_name"]
                        },
                        "COUNTRY": {
                            "aliases": ["country", "location", "region"],
                            "meaning": "Country or region where customer is located",
                            "synonyms": ["location", "region", "territory"]
                        },
                        "INTERSIGHT_CONSUMPTION": {
                            "aliases": ["consumption", "usage", "intersight"],
                            "meaning": "Type of Intersight consumption or usage",
                            "synonyms": ["usage_type", "consumption_type", "service_type"]
                        }
                    }
                }
            }
        }
        
        # Create test policy
        self.test_policy = {
            "intent": {
                "min_confidence": 0.3,
                "default_confidence": 0.5,
                "fallback_confidence": 0.1,
                "default_action": "SELECT",
                "valid_actions": ["SELECT", "COUNT", "AGGREGATE", "SEARCH", "COMPARE"],
                "default_complexity": "simple",
                "valid_complexities": ["simple", "moderate", "complex"],
                "max_tables": 8,
                "max_columns_per_table": 10
            }
        }
    
    def test_schema_prefiltering(self):
        """Test schema prefiltering functionality"""
        print("\n🧪 Testing Schema Prefiltering...")
        
        # Test 1: Customer-related query
        query1 = "show me all customers from USA"
        filtered1 = prefilter_schema(query1, self.mock_schema_context, max_tables=3, max_columns_per_table=5)
        
        print(f"Query: '{query1}'")
        print(f"Selected tables: {list(filtered1['tables'].keys())}")
        for table_name, table_data in filtered1['tables'].items():
            print(f"  {table_name}: {list(table_data['columns'].keys())}")
        
        # Assertions
        self.assertIn("raw_data", filtered1["tables"])
        self.assertIn("CUSTOMER_NAME", filtered1["tables"]["raw_data"]["columns"])
        self.assertIn("COUNTRY", filtered1["tables"]["raw_data"]["columns"])
        
        # Test 2: Consumption-related query
        query2 = "what is the intersight consumption for each customer"
        filtered2 = prefilter_schema(query2, self.mock_schema_context, max_tables=3, max_columns_per_table=5)
        
        print(f"\nQuery: '{query2}'")
        print(f"Selected tables: {list(filtered2['tables'].keys())}")
        for table_name, table_data in filtered2['tables'].items():
            print(f"  {table_name}: {list(table_data['columns'].keys())}")
        
        # Assertions
        self.assertIn("raw_data", filtered2["tables"])
        self.assertIn("INTERSIGHT_CONSUMPTION", filtered2["tables"]["raw_data"]["columns"])
        self.assertIn("CUSTOMER_NAME", filtered2["tables"]["raw_data"]["columns"])
    
    def test_intent_validation(self):
        """Test intent response validation"""
        print("\n🧪 Testing Intent Validation...")
        
        # Test 1: Valid intent response
        valid_intent = {
            "action": "SELECT",
            "tables": ["raw_data"],
            "columns": ["CUSTOMER_NAME", "COUNTRY"],
            "conditions": ["COUNTRY = 'USA'"],
            "ambiguity_detected": False,
            "clarifying_questions": [],
            "complexity_level": "simple",
            "confidence": 0.85
        }
        
        validated = validate_intent_response(valid_intent, self.test_policy)
        print(f"Valid Intent: {json.dumps(validated, indent=2)}")
        
        # Assertions
        self.assertEqual(validated["action"], "SELECT")
        self.assertEqual(validated["tables"], ["raw_data"])
        self.assertEqual(validated["confidence"], 0.85)
        self.assertFalse(validated["ambiguity_detected"])
        
        # Test 2: Invalid intent response (wrong types)
        invalid_intent = {
            "action": "INVALID_ACTION",
            "tables": "not_a_list",
            "columns": 123,
            "confidence": 1.5
        }
        
        validated = validate_intent_response(invalid_intent, self.test_policy)
        print(f"Invalid Intent (Fixed): {json.dumps(validated, indent=2)}")
        
        # Assertions
        self.assertEqual(validated["action"], "SELECT")  # Should default to SELECT
        self.assertEqual(validated["tables"], [])  # Should be empty list
        self.assertEqual(validated["confidence"], 0.5)  # Should default to 0.5
        self.assertTrue(validated["ambiguity_detected"])  # Should be marked as ambiguous
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_interpret_plan_node_success(self, mock_llm_adapter):
        """Test successful intent interpretation"""
        print("\n🧪 Testing Interpret Plan Node (Success)...")
        
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
        
        # Create test state
        state = AppState(
            user_query="show me all customers from USA",
            schema_context=self.mock_schema_context,
            policy=self.test_policy
        )
        
        # Test interpret_plan_node
        result = interpret_plan_node(state)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Assertions
        self.assertEqual(result["status"], "PLANNING")
        self.assertIsNone(result.get("error_message"))
        self.assertTrue(result["plan_ok"])
        self.assertEqual(result["plan_confidence"], 0.85)
        self.assertFalse(result["ambiguity"])
        
        # Check intent_json
        intent_json = result["intent_json"]
        self.assertEqual(intent_json["action"], "SELECT")
        self.assertEqual(intent_json["tables"], ["raw_data"])
        self.assertEqual(intent_json["columns"], ["CUSTOMER_NAME", "COUNTRY"])
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_interpret_plan_node_ambiguous(self, mock_llm_adapter):
        """Test ambiguous intent interpretation"""
        print("\n🧪 Testing Interpret Plan Node (Ambiguous)...")
        
        # Mock LLM response with ambiguity
        mock_llm_client = MagicMock()
        mock_llm_client.interpret_query_intent.return_value = {
            "action": "SELECT",
            "tables": [],
            "columns": ["CUSTOMER_NAME"],
            "conditions": [],
            "ambiguity_detected": True,
            "clarifying_questions": ["Which table should I query?"],
            "complexity_level": "simple",
            "confidence": 0.2
        }
        mock_llm_adapter.return_value = mock_llm_client
        
        # Create test state
        state = AppState(
            user_query="show me customer names",
            schema_context=self.mock_schema_context,
            policy=self.test_policy
        )
        
        # Test interpret_plan_node
        result = interpret_plan_node(state)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Assertions
        self.assertEqual(result["status"], "PLANNING")
        self.assertIsNone(result.get("error_message"))
        self.assertFalse(result["plan_ok"])  # Should be False due to ambiguity
        self.assertEqual(result["plan_confidence"], 0.2)
        self.assertTrue(result["ambiguity"])
        self.assertIsNotNone(result["clarifying_question"])
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_interpret_plan_node_error(self, mock_llm_adapter):
        """Test error handling in intent interpretation"""
        print("\n🧪 Testing Interpret Plan Node (Error)...")
        
        # Mock LLM error
        mock_llm_client = MagicMock()
        mock_llm_client.interpret_query_intent.side_effect = Exception("LLM API Error")
        mock_llm_adapter.return_value = mock_llm_client
        
        # Create test state
        state = AppState(
            user_query="show me all customers from USA",
            schema_context=self.mock_schema_context,
            policy=self.test_policy
        )
        
        # Test interpret_plan_node
        result = interpret_plan_node(state)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Assertions - error should be handled gracefully
        self.assertEqual(result["status"], "PLANNING")  # Error is handled gracefully
        self.assertIsNone(result.get("error_message"))  # No error message in result
        self.assertFalse(result["plan_ok"])  # Plan should not be OK due to fallback
        self.assertEqual(result["plan_confidence"], 0.1)  # Should use fallback confidence
        self.assertTrue(result["ambiguity"])  # Should be marked as ambiguous
    
    def test_fallback_intent(self):
        """Test fallback intent generation"""
        print("\n🧪 Testing Fallback Intent...")
        
        fallback = _get_fallback_intent(0.1)
        print(f"Fallback Intent: {json.dumps(fallback, indent=2)}")
        
        # Assertions
        self.assertEqual(fallback["action"], "SELECT")
        self.assertEqual(fallback["tables"], [])
        self.assertEqual(fallback["columns"], [])
        self.assertTrue(fallback["ambiguity_detected"])
        self.assertEqual(fallback["confidence"], 0.1)
        self.assertIn("couldn't understand", fallback["clarifying_questions"][0])


class TestQueryScenarios(unittest.TestCase):
    """Test different query scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_schema_context = {
            "tables": {
                "raw_data": {
                    "name": "raw_data",
                    "description": "Customer consumption data",
                    "columns": {
                        "CUSTOMER_NAME": {"name": "CUSTOMER_NAME", "data_type": "TEXT"},
                        "COUNTRY": {"name": "COUNTRY", "data_type": "TEXT"},
                        "INTERSIGHT_CONSUMPTION": {"name": "INTERSIGHT_CONSUMPTION", "data_type": "TEXT"},
                        "CUSTOMER_ID": {"name": "CUSTOMER_ID", "data_type": "INTEGER", "is_primary_key": True}
                    }
                }
            }
        }
        
        self.test_policy = {
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
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_select_query(self, mock_llm_adapter):
        """Test SELECT query interpretation"""
        print("\n🧪 Testing SELECT Query...")
        
        mock_llm_client = MagicMock()
        mock_llm_client.interpret_query_intent.return_value = {
            "action": "SELECT",
            "tables": ["raw_data"],
            "columns": ["CUSTOMER_NAME", "COUNTRY"],
            "conditions": ["COUNTRY = 'USA'"],
            "ambiguity_detected": False,
            "clarifying_questions": [],
            "complexity_level": "simple",
            "confidence": 0.9
        }
        mock_llm_adapter.return_value = mock_llm_client
        
        state = AppState(
            user_query="show me all customers from USA",
            schema_context=self.mock_schema_context,
            policy=self.test_policy
        )
        
        result = interpret_plan_node(state)
        
        print(f"Query: 'show me all customers from USA'")
        print(f"Intent: {json.dumps(result['intent_json'], indent=2)}")
        print(f"Plan OK: {result['plan_ok']}")
        print(f"Confidence: {result['plan_confidence']}")
        
        self.assertEqual(result["intent_json"]["action"], "SELECT")
        self.assertTrue(result["plan_ok"])
        self.assertEqual(result["plan_confidence"], 0.9)
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_count_query(self, mock_llm_adapter):
        """Test COUNT query interpretation"""
        print("\n🧪 Testing COUNT Query...")
        
        mock_llm_client = MagicMock()
        mock_llm_client.interpret_query_intent.return_value = {
            "action": "COUNT",
            "tables": ["raw_data"],
            "columns": ["CUSTOMER_ID"],
            "conditions": ["COUNTRY = 'USA'"],
            "ambiguity_detected": False,
            "clarifying_questions": [],
            "complexity_level": "simple",
            "confidence": 0.8
        }
        mock_llm_adapter.return_value = mock_llm_client
        
        state = AppState(
            user_query="how many customers are from USA",
            schema_context=self.mock_schema_context,
            policy=self.test_policy
        )
        
        result = interpret_plan_node(state)
        
        print(f"Query: 'how many customers are from USA'")
        print(f"Intent: {json.dumps(result['intent_json'], indent=2)}")
        print(f"Plan OK: {result['plan_ok']}")
        print(f"Confidence: {result['plan_confidence']}")
        
        self.assertEqual(result["intent_json"]["action"], "COUNT")
        self.assertTrue(result["plan_ok"])
        self.assertEqual(result["plan_confidence"], 0.8)
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_aggregate_query(self, mock_llm_adapter):
        """Test AGGREGATE query interpretation"""
        print("\n🧪 Testing AGGREGATE Query...")
        
        mock_llm_client = MagicMock()
        mock_llm_client.interpret_query_intent.return_value = {
            "action": "AGGREGATE",
            "tables": ["raw_data"],
            "columns": ["COUNTRY", "INTERSIGHT_CONSUMPTION"],
            "conditions": [],
            "ambiguity_detected": False,
            "clarifying_questions": [],
            "complexity_level": "moderate",
            "confidence": 0.7
        }
        mock_llm_adapter.return_value = mock_llm_client
        
        state = AppState(
            user_query="group customers by country and show their consumption",
            schema_context=self.mock_schema_context,
            policy=self.test_policy
        )
        
        result = interpret_plan_node(state)
        
        print(f"Query: 'group customers by country and show their consumption'")
        print(f"Intent: {json.dumps(result['intent_json'], indent=2)}")
        print(f"Plan OK: {result['plan_ok']}")
        print(f"Confidence: {result['plan_confidence']}")
        
        self.assertEqual(result["intent_json"]["action"], "AGGREGATE")
        self.assertTrue(result["plan_ok"])
        self.assertEqual(result["plan_confidence"], 0.7)
        self.assertEqual(result["intent_json"]["complexity_level"], "moderate")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
