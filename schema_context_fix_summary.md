# Schema Context Loading Issue - Investigation Summary

## Problem Identified

The schema context was not being generated and loaded properly in the AI assistant workflow, causing SQL generation to fail with the error "Cannot generate SQL: schema_context empty".

## Root Cause

The issue was caused by **incorrect environment variables** in the `.env` file:

1. **DATABASE_URL** was set to `sqlite:///./data/sql_assistant_langchain.db` (wrong database)
2. **DB_DIALECT** was set to `postgres` (wrong dialect)

The schema context node uses these environment variables as fallbacks when the state doesn't provide a database URL:

```python
database_url = getattr(state, "database_url", None) or os.getenv("DATABASE_URL", "sqlite:///data/sample.db")
```

## Investigation Results

### ✅ What Works
- Database connection to `data/raw_data.db` works fine
- Schema reflector can detect tables from all database URLs
- SQL generation works when given correct schema context
- Schema context node works when environment variables are correct

### ❌ What Was Broken
- Schema context node was using wrong database URL from environment
- Workflow was failing to generate SQL due to empty schema context
- Generated SQL was incorrect due to missing schema information

## Solution

### Fix the `.env` file:

Change these lines in `.env`:

```bash
# FROM (incorrect):
DATABASE_URL=sqlite:///./data/sql_assistant_langchain.db
DB_DIALECT=postgres

# TO (correct):
DATABASE_URL=sqlite:///data/raw_data.db
DB_DIALECT=sqlite
```

### Alternative: Override in Code

If you can't modify the `.env` file, you can override the environment variables in your code:

```python
import os
os.environ['DATABASE_URL'] = 'sqlite:///data/raw_data.db'
os.environ['DB_DIALECT'] = 'sqlite'
```

## Verification

After fixing the environment variables:

1. **Schema Context Node**: ✅ Now detects 1 table (raw_data) with 24 columns
2. **Workflow Integration**: ✅ Now generates SQL candidates
3. **SQL Generation**: ✅ Now produces SQL queries (though they need refinement)

## Expected SQL for the Query

For the query "Show me top 2 customers for intersight SaaS for each year", the expected SQL should be:

```sql
WITH RankedCustomers AS (
    SELECT 
        CUSTOMER_NAME,
        YEAR,
        ACTUAL_BOOKINGS,
        ROW_NUMBER() OVER (
            PARTITION BY YEAR 
            ORDER BY ACTUAL_BOOKINGS DESC
        ) as rank
    FROM raw_data 
    WHERE IntersightConsumption = 'SaaS'
)
SELECT 
    CUSTOMER_NAME,
    YEAR,
    ACTUAL_BOOKINGS
FROM RankedCustomers 
WHERE rank <= 2
ORDER BY YEAR, rank
```

## Next Steps

1. **Fix the `.env` file** with the correct database URL and dialect
2. **Test the workflow** with the original query
3. **Refine the SQL generation** to produce more accurate queries for complex requirements like window functions

## Files to Clean Up

After fixing the issue, you can delete these investigation files:
- `investigate_schema.py`
- `fix_env_test.py`
- `schema_context_fix_summary.md`
