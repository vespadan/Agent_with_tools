# AI Agent Instructions for Agent_with_tools

## Project Overview
Agentic RAG system with Gradio UI that integrates Ollama LLMs with LangChain agents and extensible tool calling. Main components: `Agentic_RAG.py` (UI/orchestration) and `tools.py` (tool registry and implementations).

## Critical Architecture Patterns

### Tool Registration System
All tools MUST use the `@register_function` decorator from `tools.py` to be discoverable:
```python
@register_function
def my_tool(query: str) -> str:
    """Docstring becomes tool description for the agent."""
    return result
```

Tools are automatically registered in `registered_functions` dict and wrapped as LangChain Tools via `_make_langchain_tools()`. The tool's docstring becomes its LangChain description.

### LLM Configuration
Ollama integration uses singleton pattern via `get_ollama_llm()` accessor:
- **OLLAMA_HOST**: Default `http://10.100.14.73:11434` (network deployment)
- **OLLAMA_MODEL**: Default `Qwen2.5:32b-instruct`
- **EMBEDDINGS_MODEL**: Default `nomic-embed-text:latest` for RAG

Never instantiate `OllamaLLM()` directly - always use `get_ollama_llm()` to ensure singleton consistency across RAG tool and main agent.

### Agent Prompt Engineering
The agent uses a custom Thought/Action/Observation loop defined in `LC_PROMPT` (assembled from `PREFIX`, `FORMAT_INSTRUCTIONS`, `SUFFIX`). Key characteristics:
- Expects JSON-formatted `Action Input`
- Supports up to 3 Thought/Action/Observation cycles
- Requires `Final Answer:` format when no tool needed
- Enforces Italian responses via `SUFFIX`
- Verbosity instruction: minimum 500 words in responses

### Conversation History Management
Per-user file-based persistence in `results/agent_histories/history_{user_id}.json`:
- Format: List of dicts with `{"role": "user/assistant", "content": "..."}`
- Functions: `load_history()`, `save_history()`, `history_filepath()`
- Global agent state uses `_AGENT_EXECUTOR` and `_AGENT_MEMORY` singletons

### Tool Input Parsing
`tools.run_callable()` implements flexible argument handling:
1. **Dict args**: Pass as `**kwargs`
2. **JSON string**: Parse and dispatch by type (dict→kwargs, list→positional, number→single param)
3. **Comma-separated values**: Split and convert to appropriate types
4. **Parameter introspection**: Uses `inspect.signature()` to intelligently bind arguments when function expects single parameter named 'query' or similar

This complexity handles LangChain agents providing varied input formats (e.g., bare strings vs structured JSON).

## Development Workflows

### Adding New Tools
1. Define function with clear docstring in `tools.py`
2. Add `@register_function` decorator
3. Add corresponding tool schema dict at bottom of `tools.py` (format: `{name}_tool`)
4. Update `tool_names` list in `Agentic_RAG.py` with `/{name}` prefix
5. No manual registration needed - auto-discovered by `_make_langchain_tools()`

Example:
```python
@register_function
def my_tool(query: str) -> dict:
    """Use this tool to..."""
    return {"status": "ok", "result": "..."}

my_tool_schema = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "...",
        "parameters": {"type": "object", "properties": {...}, "required": [...]}
    }
}
```

### RAG Tool Integration
The `/RAG` tool retrieves context from `docs/risorgimento.txt` (Italian history document):
- Uses FAISS vectorstore with Ollama embeddings
- **Critical**: Must access shared `ollama_llm` via `get_ollama_llm()` from `Agentic_RAG` module
- Imports `Agentic_RAG` dynamically to avoid circular dependencies
- Vectorstore initialized lazily on first query if not pre-built

### Running the Application
```bash
# Set environment variables (optional, has defaults)
export OLLAMA_HOST="http://10.100.14.73:11434"
export OLLAMA_MODEL="Qwen2.5:32b-instruct"
export LANGSMITH_TRACING="false"  # Enable for debugging

# Run the Gradio interface
python Agentic_RAG.py
# Opens at http://127.0.0.1:7860
```

### Dependencies
Install via `requirements.txt` (302 lines). Key packages:
- `gradio` (5.49.1): Web UI
- `langchain-ollama`, `langchain-community`: Agent framework
- `faiss-cpu`: Vector similarity search
- `wolframalpha`, `wikipedia`: External API tools
- `httpx`: HTTP client for weather/news tools

External requirement: Ollama server running at configured host (install from https://ollama.com/download)

## Project-Specific Conventions

### Error Handling in Tools
Return dict with `{"error": "message"}` rather than raising exceptions - allows agent to handle gracefully:
```python
if not query:
    return {"error": "Missing query parameter"}
```

### Multi-language Support
- Agent responses default to Italian (enforced in `SUFFIX`)
- Wikipedia tool uses Italian API wrapper
- Google News tool defaults to `country="IT"`

### Agent Debugging
Set `LANGSMITH_TRACING=true` and provide `LANGSMITH_API_KEY` for LangChain tracing. Verbose mode already enabled in agent initialization (`verbose=True`).

### Safe Copies Directory
`safecopy/` contains dated backups with descriptive names (e.g., `251104 copia consolidata/`). Pattern: `YYMMDD backup description/`. Preserve this for rollback capability.

## Common Pitfalls

1. **Tool input format**: Agent may send bare strings instead of JSON - `run_callable()` handles this, but ensure your tool accepts flexible inputs
2. **LLM initialization**: Never create multiple `OllamaLLM` instances - breaks singleton pattern for RAG tool
3. **Import cycles**: RAG tool imports `Agentic_RAG` dynamically at runtime to access shared state
4. **Parameter binding**: If agent provides unexpected argument structure, `run_callable()` introspects function signature - ensure parameter names match common patterns like 'query', 'city', etc.
5. **Gradio state management**: Use `gr.State` dict with `{"user_id": ..., "messages": [...]}` structure for session persistence

## Key Files Reference
- `Agentic_RAG.py`: Main orchestrator (542 lines) - LLM wrapper, agent initialization, Gradio UI, history management
- `tools.py`: Tool registry (716 lines) - all tool implementations and schemas
- `docs/risorgimento.txt`: RAG knowledge base for Italian history queries
- `results/agent_histories/`: Per-user conversation persistence
- `requirements.txt`: Full dependency specification

## Testing Approach
No formal test suite detected. Manual testing workflow:
1. Start Ollama server with required models
2. Run `Agentic_RAG.py`
3. Test tool invocation via chat interface (e.g., "What's the weather in Rome?")
4. Verify history persistence across sessions
5. Check LangSmith traces if enabled
