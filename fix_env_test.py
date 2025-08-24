#!/usr/bin/env python3
"""
Fix environment variables and test schema context
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def fix_and_test():
    """Fix environment variables and test schema context"""
    
    print("1. Fixing Environment Variables...")
    
    # Temporarily set correct environment variables
    os.environ['DATABASE_URL'] = 'sqlite:///data/raw_data.db'
    os.environ['DB_DIALECT'] = 'sqlite'
    
    print(f"   Set DATABASE_URL to: {os.environ.get('DATABASE_URL')}")
    print(f"   Set DB_DIALECT to: {os.environ.get('DB_DIALECT')}")
    
    print("\n2. Testing Schema Context Node...")
    
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
            
        # Show actual tables
        db_tables = result.get('db_schema', {}).get('tables', {})
        if db_tables:
            print(f"\n     DB Schema Tables:")
            for table_name, table_info in db_tables.items():
                cols = table_info.get('columns', [])
                print(f"       {table_name}: {len(cols)} columns")
        
        schema_tables = result.get('schema_context', {}).get('tables', {})
        if schema_tables:
            print(f"\n     Schema Context Tables:")
            for table_name, table_info in schema_tables.items():
                cols = table_info.get('columns', [])
                print(f"       {table_name}: {len(cols)} columns")
                
        return result
        
    except Exception as e:
        print(f"   Schema context node error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_workflow_with_fixed_env():
    """Test the workflow with fixed environment variables"""
    
    print("\n3. Testing Workflow with Fixed Environment...")
    
    try:
        from workflows.main_graph import SQLAssistantWorkflow
        
        # Create workflow
        workflow = SQLAssistantWorkflow()
        
        # Test with the original query
        user_query = "Show me top 2 customers for intersight SaaS for each year"
        
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
        
        # Show SQL candidates if available
        sql_candidates = result.get('sql_candidates', [])
        if sql_candidates:
            print(f"\n     SQL Candidates ({len(sql_candidates)}):")
            for i, sql in enumerate(sql_candidates, 1):
                print(f"       {i}. {sql}")
        
        # Show metrics
        metrics = result.get('metrics', {})
        for key, value in metrics.items():
            if key.startswith('schema_'):
                print(f"     {key}: {value}")
                
    except Exception as e:
        print(f"   Workflow integration error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the fix and test"""
    print("=" * 80)
    print("FIXING ENVIRONMENT VARIABLES AND TESTING SCHEMA CONTEXT")
    print("=" * 80)
    
    # Fix environment and test schema context
    result = fix_and_test()
    
    # Test workflow with fixed environment
    test_workflow_with_fixed_env()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    
    if result and result.get('db_schema', {}).get('tables'):
        print("✅ SUCCESS: Schema context is now working!")
        print("   The issue was incorrect environment variables in .env file:")
        print("   - DATABASE_URL was pointing to wrong database")
        print("   - DB_DIALECT was set to postgres instead of sqlite")
    else:
        print("❌ FAILED: Schema context still not working")

if __name__ == "__main__":
    main()
