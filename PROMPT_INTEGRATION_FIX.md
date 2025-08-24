# Prompt Integration Fix

## Issue Identified

The optimized SQL generation implementation in `agents/sql_generate/tools.py` was **not using the existing `prompts/sql_generator.txt` template** and was instead using hardcoded prompts. This was inconsistent with the rest of the codebase and missed the opportunity to leverage the detailed prompt template.

### **Original Problem**

```python
# OLD: Hardcoded system prompt
system_message="You are an expert SQL generator. Generate only valid SQL queries."

# OLD: Hardcoded user prompt
prompt = f"""Generate {k} SQL query candidates based on the following plan:
Plan DSL: {json.dumps(plan_dsl, indent=2)}
...
"""
```

### **Root Cause**

The implementation was inconsistent with the existing codebase pattern used in `core/llm_adapter.py`:

```python
# CORRECT PATTERN (from llm_adapter.py)
with open("prompts/sql_generator.txt", "r") as f:
    prompt_template = f.read()

system_prompt = get_system_prompt("sql_assistant")

response = self.client.json(
    prompt=prompt,
    system_message=system_prompt,
    temperature=0.2
)
```

## Solution Implemented

### **1. System Prompt Integration**

**Before:**
```python
system_message="You are an expert SQL generator. Generate only valid SQL queries."
```

**After:**
```python
from core.llm_adapter import get_system_prompt
system_prompt = get_system_prompt("sql_assistant")
```

### **2. User Prompt Template Integration**

**Before:**
```python
prompt = f"""Generate {k} SQL query candidates based on the following plan:
Plan DSL: {json.dumps(plan_dsl, indent=2)}
...
"""
```

**After:**
```python
# Load the existing prompt template
with open("prompts/sql_generator.txt", "r") as f:
    prompt_template = f.read()

# Convert plan DSL to natural language description
plan_description = _plan_dsl_to_query_description(plan_dsl)

# Manual string replacement to avoid conflicts with JSON braces
prompt = prompt_template.replace("{user_query}", plan_description)
prompt = prompt.replace("{schema_context}", json.dumps(schema_context, indent=2))
prompt = prompt.replace("{dialect}", dialect)
prompt = prompt.replace("{num_candidates}", str(k))
```

### **3. Plan DSL to Natural Language Conversion**

Added `_plan_dsl_to_query_description()` function to convert structured plan DSL back to natural language for the template:

```python
def _plan_dsl_to_query_description(plan_dsl: Dict[str, Any]) -> str:
    """Convert plan DSL to a natural language query description for the prompt template."""
    
    # Example conversion:
    # Plan DSL: {"action": "SELECT", "tables": ["raw_data"], "columns": ["CUSTOMER_NAME"]}
    # → Natural Language: "SELECT from raw_data columns: CUSTOMER_NAME"
```

## Key Improvements

### **✅ Consistency**
- Now uses the same prompt template as the rest of the codebase
- Follows the established pattern from `llm_adapter.py`
- Maintains consistency across all SQL generation calls

### **✅ System Prompt**
- Uses `get_system_prompt("sql_assistant")` instead of hardcoded string
- Leverages the centralized system prompt management
- Consistent with other LLM calls in the codebase

### **✅ Template Usage**
- Loads `prompts/sql_generator.txt` template
- Converts plan DSL to natural language for template compatibility
- Handles JSON brace conflicts with manual string replacement

### **✅ Robustness**
- Maintains fallback behavior if template loading fails
- Graceful error handling
- Backward compatibility

### **✅ Efficiency**
- Still uses structured intent instead of re-processing user query
- Maintains the performance benefits of the optimized approach
- Adds template consistency without performance penalty

## Technical Details

### **Template Compatibility**

The `sql_generator.txt` template expects:
- `{user_query}` - Natural language query
- `{schema_context}` - Database schema
- `{dialect}` - SQL dialect
- `{num_candidates}` - Number of candidates

**Solution:** Convert plan DSL to natural language description to fit the template format.

### **JSON Brace Handling**

The template contains JSON examples with curly braces that conflict with string formatting:
```json
{
    "sql_candidates": [
        "SELECT ... FROM ... WHERE ..."
    ]
}
```

**Solution:** Use manual string replacement instead of `format_prompt()` to avoid conflicts.

### **Few-shot Integration**

Few-shot examples are conditionally added to the template:
```python
if fewshots:
    fewshot_section = f"\n\nFew-shot Examples:\n{json.dumps(fewshots, indent=2)}"
    prompt_template = prompt_template.replace(
        "Generate the SQL candidates now:",
        f"{fewshot_section}\n\nGenerate the SQL candidates now:"
    )
```

## Benefits

### **1. Consistency**
- All SQL generation now uses the same prompt template
- Unified system prompt management
- Consistent behavior across the application

### **2. Maintainability**
- Single source of truth for SQL generation prompts
- Centralized prompt management
- Easier to update and maintain

### **3. Quality**
- Leverages the detailed instructions in `sql_generator.txt`
- Better error handling and validation
- More comprehensive prompt structure

### **4. Efficiency**
- Maintains the performance benefits of using structured intent
- No additional LLM calls
- Same execution speed with better prompt quality

## Testing

The fix was validated with comprehensive tests:

- ✅ Plan DSL to natural language conversion
- ✅ Prompt template loading and formatting
- ✅ System prompt retrieval
- ✅ Fallback behavior
- ✅ Template compatibility

## Conclusion

The prompt integration fix successfully addresses the inconsistency issue while maintaining all the performance benefits of the optimized SQL generation approach. The implementation now:

1. **Uses the existing `sql_generator.txt` template**
2. **Retrieves system prompts from `get_system_prompt()`**
3. **Converts plan DSL to natural language for template compatibility**
4. **Maintains fallback behavior for robustness**
5. **Preserves the efficiency gains of structured intent usage**

This fix ensures the optimized SQL generation is both **efficient** and **consistent** with the rest of the codebase.
