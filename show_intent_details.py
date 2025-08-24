#!/usr/bin/env python3
"""
Display Intent Details

This script shows the intent that was generated and passed to SQL Generation
from our test query: "Show me top customer for SaaS in all years."
"""

import json
from pprint import pprint

def show_intent_details():
    """Display the intent details from our test results"""
    
    print("🔍 INTENT ANALYSIS")
    print("=" * 80)
    print("Query: 'Show me top customer for SaaS in all years.'")
    print("=" * 80)
    
    # Intent JSON from our test results
    intent_json = {
        "action": "SELECT",
        "tables": ["customers", "sales_data"],
        "columns": ["customer_name", "revenue", "sale_date"],
        "conditions": ["product_category = 'SaaS'"],
        "ambiguity_detected": False,
        "clarifying_questions": [],
        "complexity_level": "moderate",
        "confidence": 0.85
    }
    
    # Execution Plan from our test results
    execution_plan = {
        "action": "SELECT",
        "tables": ["customers", "sales_data"],
        "columns": ["customer_name", "revenue", "sale_date"],
        "conditions": ["product_category = 'SaaS'"],
        "complexity_level": "moderate",
        "joins": [],
        "where_conditions": ["product_category = 'SaaS'"],
        "order_by": None,
        "limit": None,
        "group_by": None,
        "detected_mentions": [
            {
                "text": "top",
                "type": "rank",
                "n": None,
                "direction": "DESC",
                "metric_hint": None,
                "span": [8, 11]
            },
            {
                "text": "years",
                "type": "period",
                "span": [37, 42]
            },
            {
                "text": "customer",
                "type": "entity_hint",
                "span": [12, 20]
            }
        ],
        "query_complexity": "moderate",
        "ranking": {
            "type": "rank",
            "direction": "DESC",
            "n": None,
            "metric_hint": None
        }
    }
    
    print("\n📋 INTENT JSON (LLM Response)")
    print("-" * 50)
    print(json.dumps(intent_json, indent=2))
    
    print("\n📊 EXECUTION PLAN (Enhanced with Mentions)")
    print("-" * 50)
    print(json.dumps(execution_plan, indent=2))
    
    print("\n🎯 KEY INSIGHTS")
    print("-" * 50)
    
    # Analyze the intent
    print(f"1. Action Type: {intent_json['action']}")
    print(f"2. Target Tables: {', '.join(intent_json['tables'])}")
    print(f"3. Selected Columns: {', '.join(intent_json['columns'])}")
    print(f"4. Filter Conditions: {', '.join(intent_json['conditions'])}")
    print(f"5. Complexity Level: {intent_json['complexity_level']}")
    print(f"6. Confidence Score: {intent_json['confidence']}")
    print(f"7. Ambiguity Detected: {intent_json['ambiguity_detected']}")
    
    print("\n🔍 DETECTED MENTIONS")
    print("-" * 50)
    for i, mention in enumerate(execution_plan['detected_mentions'], 1):
        print(f"{i}. {mention['type'].upper()}: '{mention['text']}' (span: {mention['span']})")
        if mention['type'] == 'rank':
            print(f"   - Direction: {mention['direction']}")
            print(f"   - Limit: {mention['n'] or 'None (top 1 implied)'}")
    
    print("\n📈 RANKING ANALYSIS")
    print("-" * 50)
    ranking = execution_plan['ranking']
    print(f"Ranking Type: {ranking['type']}")
    print(f"Direction: {ranking['direction']} (highest to lowest)")
    print(f"Limit: {ranking['n'] or 'Top 1 (implied)'}")
    print(f"Metric Hint: {ranking['metric_hint'] or 'Revenue (implied)'}")
    
    print("\n🔄 FLOW TO SQL GENERATION")
    print("-" * 50)
    print("1. Intent JSON → SQL Generation Node")
    print("2. SQL Generation receives:")
    print("   - User Query: 'Show me top customer for SaaS in all years.'")
    print("   - Schema Context: customers, sales_data tables")
    print("   - Plan: SELECT with ranking and filtering")
    print("   - Tables: ['customers', 'sales_data']")
    print("   - Columns: ['customer_name', 'revenue', 'sale_date']")
    print("   - Conditions: ['product_category = 'SaaS'']")
    print("   - Ranking: DESC order, top 1")
    
    print("\n✅ VALIDATION RESULTS")
    print("-" * 50)
    print("Plan OK: True")
    print("Confidence: 0.85 (above threshold of 0.3)")
    print("Ambiguity: False")
    print("Routing: sql_generate (successful)")
    
    print("\n" + "=" * 80)
    print("🎉 INTENT PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    show_intent_details()
