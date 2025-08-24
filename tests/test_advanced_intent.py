"""
Test Advanced Intent Parsing Capabilities

This module tests the enhanced intent parsing with complex queries including:
1. Ranking queries (top/bottom/most/fewest)
2. Period-based grouping (per year/quarter/month)
3. Complex aggregations with ranking
4. Entity recognition
5. Mention detection
"""

import unittest
import json
from unittest.mock import patch, MagicMock

from core.state import AppState
from agents.interpret_plan.subgraph import (
    interpret_plan_node,
    detect_mentions_generic,
    adapt_schema_for_llm,
    build_context_generic
)


class TestAdvancedIntentParsing(unittest.TestCase):
    """Test advanced intent parsing capabilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a comprehensive mock schema context
        self.mock_schema_context = {
            "tables": {
                "raw_data": {
                    "name": "raw_data",
                    "description": "Customer consumption data",
                    "row_count_estimate": 1000,
                    "columns": {
                        "CUSTOMER_ID": {
                            "name": "CUSTOMER_ID",
                            "data_type": "INTEGER",
                            "aliases": ["id", "customer_id"],
                            "is_primary_key": True,
                            "is_foreign_key": False
                        },
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
                        "ACTUAL_BOOKINGS": {
                            "name": "ACTUAL_BOOKINGS",
                            "data_type": "DECIMAL",
                            "aliases": ["bookings", "revenue", "amount"],
                            "is_primary_key": False,
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
                        },
                        "ACTUAL_BOOKINGS": {
                            "aliases": ["bookings", "revenue", "amount"],
                            "meaning": "Actual booking amount or revenue",
                            "synonyms": ["revenue", "sales", "amount", "value"]
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
    
    def test_mention_detection_ranking(self):
        """Test detection of ranking mentions"""
        print("\n🧪 Testing Ranking Mention Detection...")
        
        # Test various ranking patterns
        test_queries = [
            "show me the top 10 customers",
            "find the most profitable customers",
            "get the bottom 5 countries",
            "which customers have the highest bookings",
            "show customers with the lowest revenue"
        ]
        
        for query in test_queries:
            mentions = detect_mentions_generic(query)
            ranking_mentions = [m for m in mentions if m.get("type") == "rank"]
            
            print(f"\nQuery: '{query}'")
            print(f"Ranking mentions: {json.dumps(ranking_mentions, indent=2)}")
            
            # Assertions
            self.assertGreater(len(ranking_mentions), 0, f"No ranking mentions detected in: {query}")
            
            for mention in ranking_mentions:
                self.assertIn("direction", mention)
                self.assertIn("span", mention)
    
    def test_mention_detection_periods(self):
        """Test detection of period mentions"""
        print("\n🧪 Testing Period Mention Detection...")
        
        # Test various period patterns
        test_queries = [
            "show bookings for each year",
            "revenue per quarter",
            "customers in each month",
            "top performers by year",
            "growth across quarters"
        ]
        
        for query in test_queries:
            mentions = detect_mentions_generic(query)
            period_mentions = [m for m in mentions if m.get("type") in ["period", "per_group"]]
            
            print(f"\nQuery: '{query}'")
            print(f"Period mentions: {json.dumps(period_mentions, indent=2)}")
            
            # Assertions
            self.assertGreater(len(period_mentions), 0, f"No period mentions detected in: {query}")
    
    def test_mention_detection_entities(self):
        """Test detection of entity mentions"""
        print("\n🧪 Testing Entity Mention Detection...")
        
        # Test various entity patterns
        test_queries = [
            "show me all customers",
            "which products are popular",
            "revenue by country",
            "accounts in different regions"
        ]
        
        for query in test_queries:
            mentions = detect_mentions_generic(query)
            entity_mentions = [m for m in mentions if m.get("type") == "entity_hint"]
            
            print(f"\nQuery: '{query}'")
            print(f"Entity mentions: {json.dumps(entity_mentions, indent=2)}")
            
            # Assertions
            self.assertGreater(len(entity_mentions), 0, f"No entity mentions detected in: {query}")
    
    def test_schema_adaptation(self):
        """Test schema adaptation to LLM-friendly format"""
        print("\n🧪 Testing Schema Adaptation...")
        
        adapted_schema = adapt_schema_for_llm(self.mock_schema_context)
        
        print(f"Adapted schema structure: {json.dumps(adapted_schema, indent=2)}")
        
        # Assertions
        self.assertIn("tables", adapted_schema)
        self.assertIsInstance(adapted_schema["tables"], list)
        
        # Check first table structure
        first_table = adapted_schema["tables"][0]
        self.assertIn("name", first_table)
        self.assertIn("columns", first_table)
        self.assertIn("primary_key", first_table)
        self.assertIn("foreign_keys", first_table)
        
        # Check columns are in list format
        self.assertIsInstance(first_table["columns"], list)
        for col in first_table["columns"]:
            self.assertIn("name", col)
            self.assertIn("type", col)
            self.assertIn("is_pk", col)
            self.assertIn("not_null", col)
    
    def test_context_building(self):
        """Test comprehensive context building"""
        print("\n🧪 Testing Context Building...")
        
        user_query = "show me the top 10 customers with highest bookings per year"
        schema_manifest = adapt_schema_for_llm(self.mock_schema_context)
        context = build_context_generic(user_query, schema_manifest)
        
        print(f"Built context: {json.dumps(context, indent=2)}")
        
        # Assertions
        self.assertIn("schema_manifest", context)
        self.assertIn("user_query", context)
        self.assertIn("column_aliases", context)
        self.assertIn("detected_mentions", context)
        
        # Check mentions
        mentions = context["detected_mentions"]
        self.assertGreater(len(mentions), 0)
        
        # Should have ranking and period mentions
        ranking_mentions = [m for m in mentions if m.get("type") == "rank"]
        period_mentions = [m for m in mentions if m.get("type") in ["period", "per_group"]]
        
        self.assertGreater(len(ranking_mentions), 0)
        self.assertGreater(len(period_mentions), 0)
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_complex_ranking_query(self, mock_llm_adapter):
        """Test complex ranking query with period grouping"""
        print("\n🧪 Testing Complex Ranking Query...")
        
        # Mock LLM response for complex query
        mock_llm_client = MagicMock()
        mock_llm_client.interpret_query_intent.return_value = {
            "action": "AGGREGATE",
            "tables": ["raw_data"],
            "columns": ["CUSTOMER_NAME", "ACTUAL_BOOKINGS", "CREATED_AT"],
            "conditions": [],
            "ambiguity_detected": False,
            "clarifying_questions": [],
            "complexity_level": "complex",
            "confidence": 0.9
        }
        mock_llm_adapter.return_value = mock_llm_client
        
        # Create test state with complex query
        state = AppState(
            user_query="show me the top 10 customers with highest bookings per year",
            schema_context=self.mock_schema_context,
            policy=self.test_policy
        )
        
        # Test interpret_plan_node
        result = interpret_plan_node(state)
        
        print(f"Complex Query Result: {json.dumps(result, indent=2)}")
        
        # Assertions
        self.assertEqual(result["status"], "PLANNING")
        self.assertIsNone(result.get("error_message"))
        self.assertTrue(result["plan_ok"])
        self.assertEqual(result["plan_confidence"], 0.9)
        self.assertFalse(result["ambiguity"])
        
        # Check enhanced features
        self.assertIn("detected_mentions", result)
        self.assertIn("query_complexity", result)
        self.assertEqual(result["query_complexity"], "complex")
        
        # Check plan has ranking and grouping info
        plan = result["plan"]
        self.assertIn("ranking", plan)
        self.assertIn("grouping", plan)
        
        # Check mentions
        mentions = result["detected_mentions"]
        ranking_mentions = [m for m in mentions if m.get("type") == "rank"]
        period_mentions = [m for m in mentions if m.get("type") in ["period", "per_group"]]
        
        self.assertGreater(len(ranking_mentions), 0)
        self.assertGreater(len(period_mentions), 0)
    
    @patch('agents.interpret_plan.subgraph.get_llm_adapter')
    def test_simple_aggregation_query(self, mock_llm_adapter):
        """Test simple aggregation query"""
        print("\n🧪 Testing Simple Aggregation Query...")
        
        # Mock LLM response for simple query
        mock_llm_client = MagicMock()
        mock_llm_client.interpret_query_intent.return_value = {
            "action": "AGGREGATE",
            "tables": ["raw_data"],
            "columns": ["COUNTRY", "ACTUAL_BOOKINGS"],
            "conditions": [],
            "ambiguity_detected": False,
            "clarifying_questions": [],
            "complexity_level": "moderate",
            "confidence": 0.8
        }
        mock_llm_adapter.return_value = mock_llm_client
        
        # Create test state with simple query
        state = AppState(
            user_query="show total bookings by country",
            schema_context=self.mock_schema_context,
            policy=self.test_policy
        )
        
        # Test interpret_plan_node
        result = interpret_plan_node(state)
        
        print(f"Simple Aggregation Result: {json.dumps(result, indent=2)}")
        
        # Assertions
        self.assertEqual(result["status"], "PLANNING")
        self.assertTrue(result["plan_ok"])
        self.assertEqual(result["plan_confidence"], 0.8)
        self.assertEqual(result["query_complexity"], "moderate")
        
        # Check plan structure
        plan = result["plan"]
        self.assertEqual(plan["action"], "AGGREGATE")
        self.assertIn("raw_data", plan["tables"])
        self.assertIn("COUNTRY", plan["columns"])
        self.assertIn("ACTUAL_BOOKINGS", plan["columns"])


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
