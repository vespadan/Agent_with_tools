"""
Simple test harness to exercise the tool wrapper behavior in Agentic_RAG.
Runs a few cases for 'square' and 'add' to validate parsing and invocation.
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports like `tutorials.agents_with_tools` work
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import types

# Insert lightweight dummy modules for optional heavy deps so we can import
# `tools.py` in this test environment without installing large packages.
_dummy_names = [
    'cohere', 'httpx', 'langchain', 'langchain.chains', 'langchain_community',
    'langchain_community.tools', 'langchain_community.utilities', 'langchain_ollama',
    'langchain_community.document_loaders', 'langchain_community.vectorstores',
    'tomlkit', 'wolframalpha'
]
for _n in _dummy_names:
    if _n not in sys.modules:
        sys.modules[_n] = types.ModuleType(_n)

# Provide minimal attributes that tools.py expects from these optional packages
sys.modules['cohere'].Tool = lambda *a, **k: None
sys.modules['langchain.chains'].RetrievalQA = type('RetrievalQA', (), {'from_llm': staticmethod(lambda **k: None)})
sys.modules['langchain_community.tools'].WikipediaQueryRun = type('WQR', (), {'run': lambda self, q: f'wiki result for {q}'})
sys.modules['langchain_community.utilities'].WikipediaAPIWrapper = lambda *a, **k: None
sys.modules['langchain_ollama'].OllamaEmbeddings = lambda *a, **k: None
sys.modules['langchain_community.document_loaders'].TextLoader = lambda *a, **k: type('DL', (), {'load': lambda self: []})
sys.modules['langchain_community.vectorstores'].FAISS = type('FAISS', (), {'from_documents': staticmethod(lambda docs, emb: None)})
sys.modules['tomlkit'].string = str

from tutorials.agents_with_tools import tools


def parse_input(text: str) -> dict:
    import json
    import re

    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        out = {}
        pattern = re.compile(r"(\w+)\s*=\s*(\".*?\"|'.*?'|\S+)")
        for match in pattern.finditer(text):
            k = match.group(1)
            v = match.group(2)
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            if v.isdigit():
                out[k] = int(v)
            else:
                out[k] = v
        return out


# Reuse the same fallback logic implemented in Agentic_RAG for two-params
def make_args_for_tool(name: str, input_str: str):
    args = parse_input(input_str)
    if not args and (input_str or "").strip():
        try:
            fn = tools.registered_functions.get(name)
            if fn:
                import inspect
                sig = inspect.signature(fn)
                params = [
                    p.name
                    for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                ]
                if len(params) == 1:
                    args = {params[0]: input_str.strip()}
                elif len(params) == 2:
                    s = input_str.strip()
                    import re as _re
                    parts = _re.split(r"\s*,\s*|\s+", s, maxsplit=1)
                    if len(parts) == 2:
                        args = {params[0]: parts[0], params[1]: parts[1]}
                    else:
                        args = {params[0]: s, params[1]: s}
                elif "query" in params:
                    args = {"query": input_str.strip()}
        except Exception:
            pass
    return args


def run_case(name: str, raw_input: str):
    args = make_args_for_tool(name, raw_input)
    print(f"Tool: {name!r}, raw_input: {raw_input!r} -> args: {args}")
    res = tools.run_callable(name, args)
    print(f"Result: {res}\n")


if __name__ == '__main__':
    print("Running tests for tool wrapper...\n")
    # square: single param
    run_case('square', '5')
    run_case('square', '{"a": 6}')

    # add: two params (comma, whitespace, json, explicit kv)
    run_case('add', '2,3')
    run_case('add', '2 3')
    run_case('add', 'num1=7 num2=8')

    # edge cases
    run_case('add', '42')
    run_case('square', '')
