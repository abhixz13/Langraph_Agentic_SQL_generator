#!/usr/bin/env python3
"""
Investigate why schema context is not being generated and loaded properly
"""

import sys
import os
from pathlib import Path
import json
import time

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_database_connection():
    """Test direct database connection"""
    print("1. Testing Database Connection...")
    
    try:
        import sqlite3
        conn = sqlite3.connect('data/raw_data.db')
        cursor = conn.cursor()
        
        # Check if database exists and has tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"   Database tables found: {[table[0] for table in tables]}")
        
        # Check raw_data table structure
        if tables:
            cursor.execute("PRAGMA table_info('raw_data')")
            columns = cursor.fetchall()
            print(f"   raw_data table columns: {len(columns)}")
            for col in columns[:5]:  # Show first 5 columns
                print(f"     {col[1]} ({col[2]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   Database connection error: {e}")
        return False

def test_schema_reflector():
    """Test the schema reflector directly"""
    print("\n2. Testing Schema Reflector...")
    
    try:
        from agents.schema_context.subgraph import SchemaReflector
        
        # Test with different database URLs
        test_urls = [
            "sqlite:///data/raw_data.db",
            "sqlite:///./data/raw_data.db",
            "sqlite:////Users/abhijeetsinghx2/SQL_Assistant_Langchain/data/raw_data.db"
        ]
        
        for url in test_urls:
            print(f"   Testing URL: {url}")
            try:
                reflector = SchemaReflector(url, "sqlite", sample_rows=2, timeout_ms=5000)
                result = reflector.reflect()
                
                tables = result.get("tables", {})
                print(f"     Tables found: {len(tables)}")
                for table_name in tables:
                    cols = tables[table_name].get("columns", [])
                    print(f"       {table_name}: {len(cols)} columns")
                    
            except Exception as e:
                print(f"     Error: {e}")
                
    except Exception as e:
        print(f"   Schema reflector error: {e}")

def test_schema_context_node():
    """Test the schema context node directly"""
    print("\n3. Testing Schema Context Node...")
    
    try:
        from agents.schema_context.subgraph import schema_context_node
        
        # Test with dictionary state
        dict_state = {
            "user_query": "test query",
            "dialect": "sqlite",
            "database_url": "sqlite:///data/raw_data.db",
            "semantic_dir": "data/semantic",
            "policy": {
                "schema_cache_enabled": False,
                "schema_max_tables": 20,
                "schema_max_columns": 30,
                "schema_include_samples": True,
                "schema_sample_rows": 2
            }
        }
        
        print("   Testing with dictionary state...")
        result = schema_context_node(dict_state)
        
        print(f"     Status: {result.get('status')}")
        print(f"     DB Schema Tables: {len(result.get('db_schema', {}).get('tables', {}))}")
        print(f"     Schema Context Tables: {len(result.get('schema_context', {}).get('tables', {}))}")
        
        # Show metrics
        metrics = result.get('metrics', {})
        for key, value in metrics.items():
            print(f"     {key}: {value}")
            
    except Exception as e:
        print(f"   Schema context node error: {e}")
        import traceback
        traceback.print_exc()

def test_workflow_integration():
    """Test how the workflow calls the schema context"""
    print("\n4. Testing Workflow Integration...")
    
    try:
        from workflows.main_graph import SQLAssistantWorkflow
        
        # Create workflow
        workflow = SQLAssistantWorkflow()
        
        # Test with a simple query
        user_query = "Show me customers"
        
        print(f"   Testing workflow with query: {user_query}")
        
        # Run the workflow
        result = workflow.run(user_query, dialect="sqlite")
        
        print(f"     Workflow status: {result.get('status')}")
        print(f"     Error: {result.get('error', 'None')}")
        
        # Check schema context in result
        db_schema = result.get('db_schema', {})
        schema_context = result.get('schema_context', {})
        
        print(f"     DB Schema Tables: {len(db_schema.get('tables', {}))}")
        print(f"     Schema Context Tables: {len(schema_context.get('tables', {}))}")
        
        # Show metrics
        metrics = result.get('metrics', {})
        for key, value in metrics.items():
            if key.startswith('schema_'):
                print(f"     {key}: {value}")
                
    except Exception as e:
        print(f"   Workflow integration error: {e}")
        import traceback
        traceback.print_exc()

def test_environment():
    """Test environment and configuration"""
    print("\n5. Testing Environment...")
    
    # Check current directory
    print(f"   Current directory: {os.getcwd()}")
    
    # Check if database file exists
    db_path = "data/raw_data.db"
    print(f"   Database file exists: {os.path.exists(db_path)}")
    if os.path.exists(db_path):
        print(f"   Database file size: {os.path.getsize(db_path)} bytes")
    
    # Check environment variables
    print(f"   DATABASE_URL env var: {os.getenv('DATABASE_URL', 'Not set')}")
    print(f"   DB_DIALECT env var: {os.getenv('DB_DIALECT', 'Not set')}")
    
    # Check semantic directory
    semantic_dir = "data/semantic"
    print(f"   Semantic directory exists: {os.path.exists(semantic_dir)}")
    if os.path.exists(semantic_dir):
        files = os.listdir(semantic_dir)
        print(f"   Semantic files: {files}")

def main():
    """Run all investigations"""
    print("=" * 80)
    print("INVESTIGATING SCHEMA CONTEXT LOADING ISSUES")
    print("=" * 80)
    
    test_database_connection()
    test_schema_reflector()
    test_schema_context_node()
    test_workflow_integration()
    test_environment()
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
