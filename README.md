# Simple SQL Assistant

A basic SQL assistant that converts natural language to SQL and executes it safely.

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**:
   ```bash
   # Copy the example file
   cp env_example.txt .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **Test the system**:
   ```bash
   python main.py
   ```

## Usage

### Command Line
```python
from main import run_query

result = run_query("Show me all users")
print(result)
```

### With UI (coming in Day 3)
```bash
python ui.py
```

## What it does

1. Takes a natural language query
2. Generates SQL using OpenAI
3. Executes the SQL safely (SELECT only)
4. Returns results

## Current Features

- ✅ Natural language to SQL conversion
- ✅ Basic safety (SELECT queries only)
- ✅ Error handling
- ✅ Simple workflow

## Next Steps

- Real database connection
- Better SQL generation
- Web interface
- Query validation
