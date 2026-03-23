# 03 — Tool Schema Design

> *The quality of your tool schema is the quality of your agent — poorly described tools cause wrong decisions.*

---

## 3.1 Why Schema Design Matters

The LLM reads your tool schemas and uses them to decide:
1. **Whether** to call this tool
2. **When** to call this tool (vs. a different tool or no tool)
3. **What arguments** to pass

A bad schema causes:
- The LLM calling the wrong tool
- The LLM passing wrong or invalid arguments
- The LLM not calling the tool when it should
- Runtime errors from malformed arguments

**Tool schema design is prompt engineering for tools.**

---

## 3.2 The Three Parts of a Tool Schema

```
┌──────────────────────────────────────────────────────────┐
│  PART 1: NAME                                            │
│  • What the tool is called                               │
│  • Must be unique, lowercase, snake_case                 │
├──────────────────────────────────────────────────────────┤
│  PART 2: DESCRIPTION                                     │
│  • What the tool does                                    │
│  • When to use it (vs. when NOT to)                      │
│  • This is the most important part — LLM reads this      │
├──────────────────────────────────────────────────────────┤
│  PART 3: PARAMETERS                                      │
│  • JSON Schema of all inputs                             │
│  • Types, constraints, descriptions for each param       │
│  • Required vs optional parameters                       │
└──────────────────────────────────────────────────────────┘
```

---

## 3.3 Naming Conventions

```python
# ❌ Bad names
"search"         # Too generic — search for what?
"doStuff"        # camelCase, vague
"tool1"          # Meaningless
"process_data"   # What kind of processing?

# ✅ Good names — verb_noun format, specific
"web_search"         # Web search specifically
"query_database"     # SQL query on database
"send_email"         # SMTP email send
"get_stock_price"    # Get specific price data
"calculate_roi"      # Calculate specific metric
"read_local_file"    # Read a file from disk
"execute_python_code"# Run Python in sandbox
```

**Format**: `verb_noun` — describes the action and the target. Under 30 characters. No spaces.

---

## 3.4 Writing Great Tool Descriptions

The description is the most read part of the schema. It must answer:

1. **What does it do?** (simple, concrete)
2. **When should the LLM use it?** (triggers)
3. **When should it NOT use it?** (negative examples — often omitted but critical)
4. **What format are arguments expected in?** (for complex params)

### Weak vs Strong Descriptions

```python
# ❌ WEAK — too short, ambiguous
"description": "Search the web"

# ✅ STRONG — specific, with triggers and counter-examples
"description": """Search the web for current information using a search engine.

USE THIS TOOL when:
- The question involves real-time data (prices, news, scores, weather)
- The question asks about events after your knowledge cutoff (2024+)
- The user asks to 'look up', 'search for', or 'find' something current
- You need to verify a fact that may have changed

DO NOT use this tool when:
- The question is about well-established historical facts (before 2023)
- The question requires calculation or reasoning (no search needed)
- The answer is clearly within your training knowledge

For best results: use specific, targeted queries. Include the year if time-sensitive.
Example good query: 'GPT-5 release date 2025'
Example bad query: 'OpenAI stuff'"""
```

---

## 3.5 Parameter Description Engineering

Each parameter needs its own description that answers:
- What is this value for?
- What format/values are acceptable?
- What's the default if not provided?
- An example (very helpful for LLMs)

```python
# ❌ WEAK parameter descriptions
"parameters": {
    "properties": {
        "query": {"type": "string"},
        "n": {"type": "integer"}
    }
}

# ✅ STRONG parameter descriptions
"parameters": {
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query. Be specific and targeted. Use quotes for exact phrases. Example: 'Python 3.13 new features list'"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of search results to return (1-10). Default: 5. Use 3 for specific lookups, 10 for broad research tasks.",
            "minimum": 1,
            "maximum": 10,
            "default": 5
        }
    },
    "required": ["query"]
}
```

---

## 3.6 JSON Schema Constraints — Validation Built into the Schema

JSON Schema supports constraints that the LLM respects in Strict mode:

```python
# String constraints
"email": {
    "type": "string",
    "format": "email",           # email format validation
    "maxLength": 255
}

# Number constraints  
"temperature": {
    "type": "number",
    "minimum": -273.15,          # Absolute zero minimum
    "maximum": 1000,
    "description": "Temperature in Celsius"
}

# Integer constraints
"page_number": {
    "type": "integer",
    "minimum": 1,
    "description": "Page number starting from 1"
}

# Enum (controlled vocabulary)
"status": {
    "type": "string",
    "enum": ["active", "inactive", "pending"],
    "description": "Account status. Must be exactly one of the listed values."
}

# Arrays
"tags": {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 10,
    "description": "List of tags. 1-10 items, each a single word."
}

# Nested objects
"date_range": {
    "type": "object",
    "properties": {
        "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
        "end": {"type": "string", "description": "End date in YYYY-MM-DD format"}
    },
    "required": ["start", "end"]
}
```

---

## 3.7 Tool Granularity — How to Split Functions

### Too Coarse (Bad)
```python
# One god-function doing everything — LLM doesn't know what it'll do
def do_database_stuff(operation, table, data=None, filter=None, ...):
    ...
```

### Too Fine (Bad)
```python
# 20 micro-functions — LLM has too many choices, gets confused
get_user_by_id()
get_user_by_email()
get_user_by_phone()
get_user_by_username()
# ... 16 more variants
```

### Just Right ✅
```python
# Single-responsibility, composable tools
query_database(table, filters, limit)    # Read data
update_record(table, id, changes)        # Write single record
delete_record(table, id)                 # Delete with confirmation
```

**Rule of thumb**: One tool = one coherent action. If you can't describe it in 2 sentences, split it.

---

## 3.8 Designing for Reliability — Schema Anti-Patterns

| Anti-Pattern | Problem | Fix |
|---|---|---|
| **Overloaded parameter** | `action="create\|read\|update\|delete"` | Separate tools for each action |
| **Free-form JSON param** | `data: {"any": "structure"}` | Use typed properties with strict schema |
| **Ambiguous name** | `process()`, `handle()`, `manage()` | Verb_noun: `send_email()`, `query_db()` |
| **Missing optional indicator** | All fields in `required` | Only put truly required fields in `required` |
| **No examples in description** | LLM guesses format | Add "Example: 'Python 3.13 features'" |
| **Tool overlap** | `search_web` and `google_search` | One tool per capability |

---

## 3.9 Schema Validation in Code

Always validate the schema before using it — and validate LLM-generated arguments before executing:

```python
from jsonschema import validate, ValidationError

def validate_tool_call(tool_name: str, args: dict, tools: list) -> tuple[bool, str]:
    """Validate LLM-generated arguments against defined schema."""
    # Find the tool schema
    tool_schema = next((t for t in tools if t["function"]["name"] == tool_name), None)
    if not tool_schema:
        return False, f"Tool '{tool_name}' not found in registry"
    
    parameters_schema = tool_schema["function"]["parameters"]
    
    try:
        validate(instance=args, schema=parameters_schema)
        return True, "Valid"
    except ValidationError as e:
        return False, f"Invalid args: {e.message}"

# Usage
is_valid, error = validate_tool_call("get_stock_price", {"ticker": "AAPL"}, TOOLS)
print(f"Valid: {is_valid}, Message: {error}")

is_valid, error = validate_tool_call("get_stock_price", {"wrong_field": "AAPL"}, TOOLS)
print(f"Valid: {is_valid}, Message: {error}")
```

---

## 3.10 Minimal Schema vs Full Schema Comparison

```python
# MINIMAL (works, but less reliable)
{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
        }
    }
}

# FULL PRODUCTION SCHEMA (reliable, clear, validated)
{
    "type": "function",
    "function": {
        "name": "send_email",
        "strict": True,
        "description": """Send an email to one or more recipients.

USE when: the user explicitly asks to send, compose, or draft an email.
DO NOT use when: user just wants to discuss email content without sending.

IMPORTANT: Always confirm with the user before calling this function 
if not explicitly authorized to send emails automatically.""",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "minItems": 1,
                    "description": "List of recipient email addresses"
                },
                "subject": {
                    "type": "string",
                    "maxLength": 200,
                    "description": "Email subject line (max 200 chars)"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content. Can include plain text or Markdown."
                },
                "cc": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "description": "Optional CC recipients. Default: empty list.",
                    "default": []
                },
                "priority": {
                    "type": "string",
                    "enum": ["normal", "high", "urgent"],
                    "description": "Email priority. Default: normal",
                    "default": "normal"
                }
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False
        }
    }
}
```

---

## 📌 Key Takeaways

1. **Description is the most critical part** — LLM reads it to decide when and how to call the tool
2. **Include "when NOT to use"** — prevents wrong tool selection
3. **Name format**: `verb_noun` — specific, lowercase, snake_case
4. **Parameter descriptions**: type + purpose + format + example
5. **Use constraints**: `enum`, `minimum`, `maximum`, `maxLength` in the schema
6. **One tool = one action**: avoid multi-purpose overloaded functions
7. **Validate at runtime**: always check LLM-generated args before executing
8. **`strict: True` = guaranteed compliance** — use in production
