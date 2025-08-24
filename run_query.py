#!/usr/bin/env python3
"""
Run the user query and get SQL output
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def run_user_query():
    """Run the user query and get SQL output"""
    
    # Set correct environment variables
    os.environ['DATABASE_URL'] = 'sqlite:///data/raw_data.db'
    os.environ['DB_DIALECT'] = 'sqlite'
    
    print("=" * 80)
    print("RUNNING USER QUERY")
    print("=" * 80)
    
    user_query = "Show me the top 2 customers of Intersight SaaS for each year"
    print(f"User Query: {user_query}")
    print()
    
    try:
        from workflows.main_graph import SQLAssistantWorkflow
        
        # Create workflow
        workflow = SQLAssistantWorkflow()
        
        # Run the query
        print("Running query through AI assistant...")
        result = workflow.run(user_query, dialect="sqlite")
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        # Check status
        status = result.get('status', 'UNKNOWN')
        print(f"Status: {status}")
        
        # Check for errors
        if result.get('error'):
            print(f"Error: {result.get('error')}")
            return
        
        # Show SQL candidates
        sql_candidates = result.get('sql_candidates', [])
        if sql_candidates:
            print(f"\nGenerated SQL Queries ({len(sql_candidates)}):")
            print("-" * 80)
            for i, sql in enumerate(sql_candidates, 1):
                print(f"\n{i}. {sql}")
        else:
            print("\nNo SQL candidates generated")
        
        # Show primary SQL
        primary_sql = result.get('sql')
        if primary_sql:
            print(f"\nPrimary SQL:")
            print("-" * 80)
            print(primary_sql)
        
        # Show execution results if available
        exec_preview = result.get('exec_preview')
        if exec_preview:
            print(f"\nExecution Preview:")
            print(f"  Row count: {exec_preview.get('row_count', 'N/A')}")
            print(f"  Execution time: {exec_preview.get('execution_time', 'N/A')}s")
        
        result_sample = result.get('result_sample')
        if result_sample:
            print(f"\nResult Sample ({len(result_sample)} rows):")
            for i, row in enumerate(result_sample, 1):
                print(f"  {i}. {row}")
        
        # Show metrics
        metrics = result.get('metrics', {})
        if metrics:
            print(f"\nMetrics:")
            for key, value in metrics.items():
                if key.startswith('schema_'):
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error running query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_user_query()
