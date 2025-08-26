# 🔒 Secure Integration Guide - LLM Proxy Gateway

## 📋 Overview

This guide shows how to integrate the **LLM Proxy Gateway** with your existing SQL Assistant to prevent data leakage and ensure privacy compliance when communicating with external LLM services.

## 🎯 **What the LLM Proxy Gateway Solves**

### **Before (Unsecured):**
```
User Query: "Show me top 2 customers for Intersight SaaS for each year"
↓
Schema Context: Complete database structure with real table/column names
↓
OpenAI API: Receives sensitive business information
↓
Risk: Data exposure, privacy violations, compliance issues
```

### **After (Secured):**
```
User Query: "Show me top 2 customers for Intersight SaaS for each year"
↓
Sanitized Query: "Show me top 2 entities for metric for each year"
↓
Obfuscated Schema: Generic table/column names (table_001, col_001)
↓
OpenAI API: Receives anonymized, non-sensitive information
↓
Result: Secure, compliant, privacy-protected
```

---

## 🚀 **Quick Integration**

### **Step 1: Install Dependencies**
```bash
pip install cryptography
```

### **Step 2: Replace LLM Client**
```python
# OLD: Direct OpenAI client
from services.llm.client import LLMClient
client = LLMClient(api_key="your-key")

# NEW: Secure proxy gateway
from llm_proxy_gateway import SecureLLMClient, SecurityConfig

config = SecurityConfig(
    max_requests_per_minute=30,
    max_tokens_per_request=2000
)

client = SecureLLMClient(api_key="your-key", config=config)
```

### **Step 3: Update Environment Variables**
```bash
# Add to .env file
LLM_PROXY_ENCRYPTION_KEY=your-encryption-key-here
```

---

## 🔧 **Detailed Integration Steps**

### **1. Update LLM Service Factory**

**File:** `services/llm/factory.py`

```python
# Add import
from llm_proxy_gateway import SecureLLMClient, SecurityConfig

def create_secure_llm_client(
    api_key: Optional[str] = None,
    config: Optional[SecurityConfig] = None,
    **kwargs
):
    """Create a secure LLM client with proxy gateway"""
    try:
        # Get API key
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMConfigError("OpenAI API key not found")
        
        # Get configuration
        if not config:
            config = SecurityConfig(
                max_requests_per_minute=int(os.getenv("LLM_MAX_REQUESTS_PER_MINUTE", "30")),
                max_tokens_per_request=int(os.getenv("LLM_MAX_TOKENS_PER_REQUEST", "2000"))
            )
        
        # Create secure client
        client = SecureLLMClient(api_key=api_key, config=config)
        logger.info(f"Secure LLM client created with proxy gateway")
        
        return client
        
    except Exception as e:
        raise LLMConfigError(f"Failed to create secure LLM client: {str(e)}")
```

### **2. Update SQL Generation Agent**

**File:** `agents/sql_generate/subgraph.py`

```python
# Add import
from llm_proxy_gateway import LLMProxyGateway, SecurityConfig

def sql_generate_node(state: AppState) -> Dict[str, Any]:
    """Generate SQL with secure proxy gateway"""
    try:
        # Get schema context
        schema_context = getattr(state, "schema_context", {})
        user_query = getattr(state, "user_query", "")
        
        # Create secure proxy
        config = SecurityConfig(
            max_requests_per_minute=30,
            max_tokens_per_request=2000
        )
        
        proxy = LLMProxyGateway(
            api_key=os.getenv("OPENAI_API_KEY"),
            config=config
        )
        
        # Generate SQL securely
        result = proxy.secure_sql_generation(
            user_query=user_query,
            schema_context=schema_context
        )
        
        if result["success"]:
            return {
                "sql": result["deobfuscated_sql"],
                "sql_candidates": [result["deobfuscated_sql"]],
                "status": "GENERATED"
            }
        else:
            return {
                "error": result["error"],
                "status": "ERROR"
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "status": "ERROR"
        }
```

### **3. Update Interpret Plan Agent**

**File:** `agents/interpret_plan/subgraph.py`

```python
# Add import
from llm_proxy_gateway import LLMProxyGateway, SecurityConfig

def interpret_plan_node(state: AppState) -> Dict[str, Any]:
    """Interpret plan with secure proxy gateway"""
    try:
        # Get inputs
        user_query = getattr(state, "user_query", "")
        schema_context = getattr(state, "schema_context", {})
        
        # Create secure proxy
        config = SecurityConfig(
            max_requests_per_minute=30,
            max_tokens_per_request=2000
        )
        
        proxy = LLMProxyGateway(
            api_key=os.getenv("OPENAI_API_KEY"),
            config=config
        )
        
        # Create secure prompt
        secure_prompt = f"""
        Interpret the following user query and create an execution plan:
        
        User Query: {user_query}
        
        Generate a structured plan for SQL generation.
        """
        
        # Make secure API call
        result = proxy.secure_text_completion(
            prompt=secure_prompt,
            system_message="You are a SQL query interpretation assistant."
        )
        
        if result["success"]:
            # Parse the result (implement based on your format)
            intent_json = parse_intent_result(result["result"])
            plan = create_plan_from_intent(intent_json)
            
            return {
                "intent_json": intent_json,
                "plan": plan,
                "status": "INTERPRETED"
            }
        else:
            return {
                "error": result["error"],
                "status": "ERROR"
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "status": "ERROR"
        }
```

---

## 🧪 **Testing the Integration**

### **Test Script: `test_secure_integration.py`**

```python
#!/usr/bin/env python3
"""
Test secure integration with LLM Proxy Gateway
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_proxy_gateway import LLMProxyGateway, SecurityConfig

def test_secure_integration():
    """Test the secure integration"""
    
    print("🔒 Testing LLM Proxy Gateway Integration")
    print("=" * 60)
    
    # Set up configuration
    config = SecurityConfig(
        max_requests_per_minute=30,
        max_tokens_per_request=2000
    )
    
    # Create proxy gateway
    proxy = LLMProxyGateway(
        api_key=os.getenv("OPENAI_API_KEY"),
        config=config
    )
    
    # Test 1: Schema Obfuscation
    print("\n1. Testing Schema Obfuscation")
    print("-" * 40)
    
    original_schema = {
        "tables": {
            "raw_data": {
                "columns": [
                    {"name": "CUSTOMER_NAME", "type": "TEXT"},
                    {"name": "ACTUAL_BOOKINGS", "type": "REAL"},
                    {"name": "YEAR", "type": "INTEGER"}
                ]
            }
        },
        "business_context": {
            "synonyms": {"revenue": ["ACTUAL_BOOKINGS"]},
            "metrics": {"ACTUAL_BOOKINGS": "Revenue amount in USD"}
        }
    }
    
    schema_result = proxy.secure_schema_processing(original_schema)
    
    if schema_result["success"]:
        print("✅ Schema obfuscation successful")
        print(f"   Original tables: {list(original_schema['tables'].keys())}")
        print(f"   Obfuscated tables: {list(schema_result['obfuscated_schema']['tables'].keys())}")
        print(f"   Business context removed: {len(schema_result['obfuscated_schema']['business_context']) == 0}")
    else:
        print(f"❌ Schema obfuscation failed: {schema_result['error']}")
    
    # Test 2: Query Sanitization
    print("\n2. Testing Query Sanitization")
    print("-" * 40)
    
    original_query = "Show me top 2 customers for Intersight SaaS for each year"
    sanitized_query = proxy.sanitizer.sanitize_query(original_query)
    
    print(f"   Original: {original_query}")
    print(f"   Sanitized: {sanitized_query}")
    print(f"   Sensitive terms replaced: {'customer' not in sanitized_query.lower()}")
    
    # Test 3: Secure SQL Generation
    print("\n3. Testing Secure SQL Generation")
    print("-" * 40)
    
    sql_result = proxy.secure_sql_generation(
        user_query=original_query,
        schema_context=original_schema
    )
    
    if sql_result["success"]:
        print("✅ Secure SQL generation successful")
        print(f"   Original SQL: {sql_result['original_sql']}")
        print(f"   Deobfuscated SQL: {sql_result['deobfuscated_sql']}")
        print(f"   Tokens used: {sql_result['tokens_used']}")
        print(f"   Processing time: {sql_result['processing_time']:.2f}s")
    else:
        print(f"❌ Secure SQL generation failed: {sql_result['error']}")
    
    # Test 4: Security Statistics
    print("\n4. Security Statistics")
    print("-" * 40)
    
    stats = proxy.get_security_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Rate limit remaining: {stats['rate_limit_remaining']}")
    print(f"   Total tokens used: {stats['total_tokens_used']}")
    print(f"   Tables obfuscated: {stats['obfuscation_mappings']['tables']}")
    print(f"   Columns obfuscated: {stats['obfuscation_mappings']['columns']}")
    
    print("\n✅ Secure integration test completed!")

if __name__ == "__main__":
    test_secure_integration()
```

---

## 🔐 **Security Features Implemented**

### **1. Data Classification**
- **Automatic sensitivity detection** for queries and schema
- **Pattern-based classification** (HIGH/MEDIUM/LOW)
- **SQL keyword detection** for structure analysis

### **2. Schema Obfuscation**
- **Table name anonymization** (raw_data → table_001)
- **Column name anonymization** (CUSTOMER_NAME → col_001)
- **Business context removal** (synonyms, metrics)
- **Sample data removal** (no real data exposure)

### **3. Query Sanitization**
- **Sensitive term replacement** (customer → entity)
- **Business term obfuscation** (revenue → metric)
- **SQL comment removal** (no metadata leakage)

### **4. Rate Limiting & Cost Control**
- **Request rate limiting** (configurable per minute)
- **Token usage limits** (prevent excessive costs)
- **Cost estimation** and monitoring

### **5. Audit Logging**
- **Complete audit trail** of all requests
- **Data classification tracking**
- **Cost monitoring** and analysis
- **Compliance reporting** capabilities

### **6. Encryption**
- **Data encryption** for sensitive information
- **Secure key management** via environment variables
- **Encrypted storage** for audit logs

---

## 📊 **Before vs After Comparison**

| Aspect | Before (Unsecured) | After (Secured) |
|--------|-------------------|-----------------|
| **Schema Exposure** | Complete table/column names | Generic identifiers (table_001, col_001) |
| **Business Context** | Full synonyms and metrics | Completely removed |
| **Query Terms** | Original business terms | Sanitized generic terms |
| **Data Classification** | None | Automatic HIGH/MEDIUM/LOW |
| **Audit Trail** | None | Complete request logging |
| **Rate Limiting** | None | Configurable limits |
| **Cost Control** | None | Token usage monitoring |
| **Compliance** | Non-compliant | GDPR/SOC2 ready |

---

## 🚨 **Security Benefits**

### **1. Data Privacy Protection**
- **No sensitive data** leaves your system
- **Schema obfuscation** prevents structure exposure
- **Query sanitization** removes business context

### **2. Compliance Ready**
- **GDPR compliance** through data minimization
- **SOC 2 compliance** with audit logging
- **Industry standards** adherence

### **3. Cost Control**
- **Rate limiting** prevents API abuse
- **Token monitoring** tracks usage
- **Cost estimation** for budgeting

### **4. Risk Mitigation**
- **Data classification** for risk assessment
- **Audit logging** for incident response
- **Encryption** for data protection

---

## 🔧 **Configuration Options**

### **Environment Variables**
```bash
# Required
OPENAI_API_KEY=your-openai-api-key
LLM_PROXY_ENCRYPTION_KEY=your-encryption-key

# Optional (with defaults)
LLM_MAX_REQUESTS_PER_MINUTE=30
LLM_MAX_TOKENS_PER_REQUEST=2000
LLM_PROXY_LOG_LEVEL=INFO
```

### **Security Configuration**
```python
config = SecurityConfig(
    # Rate limiting
    max_requests_per_minute=30,
    max_tokens_per_request=2000,
    
    # Data retention
    max_data_age_hours=24,
    max_cache_size_mb=100,
    
    # Custom sensitive patterns
    sensitive_patterns=[
        r'\b(customer|client)\b',
        r'\b(revenue|sales)\b',
        # Add your custom patterns
    ]
)
```

---

## 🎯 **Next Steps**

1. **Install the proxy gateway** and dependencies
2. **Update your LLM service** to use the secure client
3. **Test the integration** with the provided test script
4. **Monitor audit logs** for compliance
5. **Adjust configuration** based on your security requirements

The LLM Proxy Gateway provides a **comprehensive security layer** that addresses all the data exposure risks identified in our security evaluation while maintaining full functionality of your SQL Assistant.
