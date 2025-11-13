# """
# What Are Tools?
# Tools are methods (functions) provided to the model (or to an agent), making them available 
# for use when the agent needs to perform actions such as collecting data from the internet, 
# making calculations, retrieving emails from a mailbox, and more. Since the main goal is to 
# make agents increasingly autonomous, we cannot respond to every concern an agent may have. 
 
# Instead, we aim to equip the agent with tools they can use independently.
# Let's explore how to build such tools and how to utilize tools that are integrated into external 
# Python libraries like Ollama and OpenAI.
# """


from cohere import Tool

import httpx
import calendar
import datetime
import importlib
import inspect
import json
import traceback
import os
from typing import Callable, Optional, Union
from langchain.chains import RetrievalQA

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from tomlkit import string

EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "nomic-embed-text:latest")

os.environ["WOLFRAMALPHA_APPID"] = "79G95VH2T5"

registered_functions = {}

def register_function(func) -> Callable:
    registered_functions[func.__name__] = func
    return func


def run_callable(name: str, arguments: dict) -> Union[Callable, dict]:
    try:
        # Prefer the registered_functions dictionary defined in this module.
        func = registered_functions.get(name)

        if not callable(func):
            return {"error": f"Function '{name}' is not callable or not found"}

        # If the caller passed a mapping/dict, use keyword expansion.
        if isinstance(arguments, dict):
            return func(**arguments)

        # If arguments is a JSON string, attempt to parse it into python types
        # (dict -> kwargs, list/tuple -> positional args, number -> single arg)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                # immediate handling based on parsed type to avoid later
                # mis-binding (e.g. parsed dict passed as single positional arg)
                if isinstance(parsed, dict):
                    return func(**parsed)
                if isinstance(parsed, (list, tuple)):
                    return func(*parsed)
                # numeric -> treat as single value
                if isinstance(parsed, (int, float)):
                    # if function accepts single parameter, bind to it, else pass as positional
                    try:
                        sig = inspect.signature(func)
                        params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
                        if len(params) == 0:
                            return func()
                        if len(params) == 1:
                            return func(**{params[0].name: parsed})
                    except Exception:
                        pass
                    return func(parsed)
                # otherwise continue with the parsed value
                arguments = parsed
            except Exception:
                # Not valid JSON: allow a simple comma-separated list of numbers
                if "," in arguments:
                    parts = [p.strip() for p in arguments.split(",") if p.strip() != ""]
                    converted = []
                    for p in parts:
                        try:
                            if any(ch in p for ch in ('.', 'e', 'E')):
                                converted.append(float(p))
                            else:
                                converted.append(int(p))
                        except Exception:
                            converted.append(p)
                    # use list of values as positional args
                    arguments = converted
                else:
                    # leave as plain string (will be handled below)
                    pass

        # If arguments is None, call without arguments (if possible).
        if arguments is None:
            return func()

        # If arguments is a list/tuple, pass as positional args.
        if isinstance(arguments, (list, tuple)):
            return func(*arguments)

        # For single non-mapping values (e.g. an int), attempt to bind
        # intelligently to the function's parameters. If the function
        # accepts exactly one parameter, call it using that parameter's
        # name (as a keyword). Otherwise try calling as single positional.
        try:
            sig = inspect.signature(func)
            params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
            if len(params) == 0:
                # function expects no args
                return func()
            if len(params) == 1:
                # use the single parameter's name so keywords like 'a' or 'query' work
                param_name = params[0].name
                return func(**{param_name: arguments})
        except Exception:
            # fallback: try calling as single positional argument
            try:
                return func(arguments)
            except Exception as e:
                traceback.print_exc()
                return {"error": f"Error executing {name}: {str(e)}"}

        # Default fallback: try calling as single positional
        try:
            return func(arguments)
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error executing {name}: {str(e)}"}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Error executing {name}: {str(e)}"}


# ################################################################# 
# # step 1: create functions & register them globally
# ################################################################# 

@register_function
def weather(city: str) -> str:
    """
    Use this tool to check updated weather for a given city.
    Remember to replace diacritics with neutral consotants or vowels, e.g. Kraków -> Krakow You need to provide city name.

    Arguments:
    city (str): The city name.

    Returns:
    str: Response containing the weather data for the provided city, or in case of error: a str containint error message.
    """

    base_url: str = f"http://wttr.in/{city}?format=j1"
    response = httpx.Client(follow_redirects=True, default_encoding="utf-8").get(base_url)
    response.raise_for_status()

    if response.status_code == 200:
        data = response.json()
        temp = data["current_condition"][0]["temp_C"]
        return f"The temperature in {city} is {temp} Celsius degree"
    else:
        return f"Could not retrieve weather data for {city}."



@register_function
def today_is() -> str:
    """
    Use this tool to check today's time and date.
    Remember to replace diacritics with neutral consotants or vowels, e.g. Kraków -> Krakow You need to provide city name.

    Returns:
    str: A string representing timestamp made of time and date in format 'YYYY-MM-DD HH:MM:SS'.
    """
    return f"The time and date now is: {str(datetime.datetime.now().replace(microsecond=0))}"


@register_function
def day_of_week() -> str:
    """
    Use to get name of today's day.

    Arguments:
    None

    Returns: name of the day of the week for today.
    """
    d = datetime.date.today()
    weekday_index = calendar.weekday(d.year, d.month, d.day)
    weekday_names = list(calendar.day_name)
    return str(weekday_names[weekday_index].capitalize())


@register_function
def add(num1: string, num2: string) -> string:
    """Add two numbers."""
    return str(int(num1) + int(num2))

@register_function
def subtract(num1: string, num2: string) -> string:
    """Subtract two numbers."""
    return str(int(num1) - int(num2))

@register_function
def multiply(num1: string, num2: string) -> string:
    """Multiply two numbers."""
    return str(int(num1) * int(num2))

@register_function
def divide(num1: string, num2: string) -> string:
    """Divide two numbers."""
    return str(int(num1) / int(num2))

@register_function
def square(a: Optional[Union[str, int]] = None, query: Optional[Union[str, int]] = None) -> str:
    """Calculates the square of a number.

    Accepts either parameter name `a` (as defined in the tool schema) or a
    single-parameter named `query`. Handles integer and floating-point inputs
    provided as numbers or numeric strings. Always returns a string (either the
    numeric result converted to str, or an error message string).
    """
    # prefer explicit 'a' parameter, otherwise fall back to 'query'
    val = a if a is not None else query

    if val is None:
        return "Error: missing required parameter 'a' or 'query'"

    # normalize strings
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return "Error: empty string provided as input"

    # try to parse as int or float
    try:
        # Accept integers like '5' or numeric types
        if isinstance(val, (int,)):
            num = val
        else:
            # if it contains a decimal point or exponent, parse as float
            s = str(val)
            if any(ch in s for ch in ('.', 'e', 'E')):
                num = float(s)
            else:
                num = int(s)

        res = num * num

        # normalize result: drop trailing .0 for integer results
        if isinstance(res, float) and res.is_integer():
            res = int(res)

        return str(res)
    except Exception as e:
        return f"Error: invalid numeric input: {e}"

@register_function
def wikipedia(query: str) -> dict:
    """Use this tool to search Wikipedia (italiano) for general knowledge.

    Returns a JSON-serializable dict with keys:
      - status: "ok" on success
      - query: the original query string
      - result: the raw result returned by the WikipediaQueryRun
    On error returns: {"error": <message>}.
    """
    # Basic validation: ensure the caller provided a non-empty query string.
    if query is None:
        return {"error": "Missing required parameter 'query'"}

    query = str(query).strip()
    if not query:
        return {"error": "Empty 'query' parameter provided"}

    try:
        api_wrapper = WikipediaAPIWrapper()
        result = WikipediaQueryRun(api_wrapper=api_wrapper).run(query)
        return {"status": "ok", "query": query, "result": result}
    except Exception as e:
        return {"error": str(e)}
     
@register_function
def RAG (query: str) -> str:
    """
    Use this tool to search with indexed text using a Retrieval-Augmented Generation (RAG) approach.
    Args: query (str): The user query string to search for relevant documents.
    Returns: str: The response generated by the RAG chain, or an error message if an exception occurs.
    Raises: Exception: If there is an error during the retrieval or generation process, an error message is returned instead of raising.
    Side Effects: Relies on global variables 'vectorstore' and 'ollama_llm' which must be initialized before calling this function.
    """
    # Prefer to use the shared objects from the Agentic_RAG module so the
    # LLM singleton and vectorstore are the same used by the UI/agent.
    try:
        agent_mod = importlib.import_module("tutorials.agents_with_tools.Agentic_RAG")
    except Exception:
        try:
            # fallback to alternative import path if run from different CWD
            agent_mod = importlib.import_module("Agentic_RAG")
        except Exception as e:
            return {"error": f"Could not import Agentic_RAG module: {e}"}

    # obtain the shared LLM singleton via accessor if available
    llm = None
    if hasattr(agent_mod, "get_ollama_llm") and callable(getattr(agent_mod, "get_ollama_llm")):
        try:
            llm = agent_mod.get_ollama_llm()
        except Exception:
            llm = None
    # fallback to direct attribute
    if llm is None and hasattr(agent_mod, "ollama_llm"):
        llm = getattr(agent_mod, "ollama_llm")

    if llm is None:
        return {"error": "Ollama LLM is not initialized. Ensure Agentic_RAG.get_ollama_llm() has been called or the app has been started."}

    # get the vectorstore from the Agentic_RAG module
    vectorstore = getattr(agent_mod, "vectorstore", None)
    if vectorstore is None:
        # creazione embeddings dal testo fornito
        documents = TextLoader("/home/vespa/VStudioProjects/Agent_with_tools/docs/risorgimento.txt").load()
        vectorstore = FAISS.from_documents(documents, OllamaEmbeddings(model=EMBEDDINGS_MODEL))
    
    if vectorstore is None:
        return {"error": "Vectorstore is not initialized. Please run build_demo() or ensure vectorstore is created before calling RAG."}

    try:
        rag_chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        # use run() for LangChain chains
        return rag_chain.run(query)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    

@register_function
def wolfram(query: str, appid: Optional[str] = None, timeout: int = 30) -> dict:
    """
    Query Wolfram Alpha using WolframAlphaTool (if available) or the wolframalpha.Client.
    Provide either appid parameter or set WOLFRAMALPHA_APPID in environment.
    Returns a dict with keys: status (ok/error), query and result (string).
    """
    if query is None or str(query).strip() == "":
        return {"error": "Missing or empty 'query' parameter"}

    try:
        # Try to use WolframAlphaTool if the package provides it (import locally)
        try:
            from wolframalpha import WolframAlphaTool

            try:
                tool = WolframAlphaTool()
                if hasattr(tool, "run"):
                    result = tool.run(query)
                    return {"status": "ok", "query": query, "result": result}
            except Exception:
                # if the tool instantiation or run fails, fall back to Client
                pass
        except Exception:
            # wolframalpha package may not expose WolframAlphaTool; continue to fallback
            pass

        # Fallback: use wolframalpha.Client
        try:
            from wolframalpha import Client
        except Exception:
            return {"error": "Package 'wolframalpha' is not available. Install it with 'pip install wolframalpha'."}

        appid = appid or os.environ.get("WOLFRAMALPHA_APPID")
        if not appid:
            return {"error": "Missing WolframAlpha APPID. Pass 'appid' or set WOLFRAMALPHA_APPID env var."}

        client = Client(appid)
        answer = None
        try:
            # Primary attempt: use the wolframalpha client
            res = client.query(query, timeout=timeout)
            try:
                if hasattr(res, "results") and res.results is not None:
                    answer = next(res.results).text
            except Exception:
                answer = None

            if not answer:
                answer = str(res)

            return {"status": "ok", "query": query, "result": answer}

        except AssertionError as ae:
            # The upstream wolframalpha package may assert on unexpected
            # Content-Type headers. Fall back to calling the HTTP API
            # directly (output=json) which is more tolerant and easier to
            # parse reliably.
            try:
                url = "https://api.wolframalpha.com/v2/query"
                params = {
                    "appid": appid,
                    "input": query,
                    "output": "json",
                    "format": "plaintext",
                }
                resp = httpx.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                pods = data.get("queryresult", {}).get("pods", [])
                parts = []
                for pod in pods:
                    for sub in pod.get("subpods", []):
                        txt = sub.get("plaintext")
                        if txt:
                            parts.append(txt)
                if parts:
                    return {"status": "ok", "query": query, "result": "\n\n".join(parts)}
                # If no plaintext parts, return raw body
                return {"status": "ok", "query": query, "result": resp.text}
            except Exception as e2:
                traceback.print_exc()
                return {"error": f"Wolfram client AssertionError and HTTP fallback failed: {e2}"}

        except Exception as e:
            # Generic fallback: try direct HTTP API as well, but surface original
            # exception if HTTP fallback also fails.
            try:
                url = "https://api.wolframalpha.com/v2/query"
                params = {
                    "appid": appid,
                    "input": query,
                    "output": "json",
                    "format": "plaintext",
                }
                resp = httpx.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                pods = data.get("queryresult", {}).get("pods", [])
                parts = []
                for pod in pods:
                    for sub in pod.get("subpods", []):
                        txt = sub.get("plaintext")
                        if txt:
                            parts.append(txt)
                if parts:
                    return {"status": "ok", "query": query, "result": "\n\n".join(parts)}
                return {"status": "ok", "query": query, "result": resp.text}
            except Exception as e2:
                traceback.print_exc()
                return {"error": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@register_function
def recent_news(category: str, days: int = 7, limit: int = 5, country: str = "IT") -> dict:
    """
    Recupera le notizie principali relative a `category` degli ultimi `days` giorni usando
    il feed RSS di Google News. Restituisce una lista di articoli (title, link,
    pubDate, description, source) e suggerimenti di interrogazioni follow-up che
    l'agente può porre per approfondire.

    Parametri:
      - category (str): tema/termine di ricerca
      - days (int): numero di giorni nel passato da considerare (default 7)
      - limit (int): numero massimo di risultati da restituire
      - country (str): country code usato per localizzare Google News (es. 'IT')

    Restituisce:
      dict con chiavi 'status'/'error', 'query', 'results' (lista) e 'suggestions' (lista)
    """
    if category is None or str(category).strip() == "":
        return {"error": "Missing or empty 'category' parameter"}

    try:
        q = f"{category} when:{days}d"
        params = {"q": q, "hl": "it", "gl": country, "ceid": f"{country}:it"}
        url = "https://news.google.com/rss/search"
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()

        # Parse RSS (XML) without adding extra deps
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.text)
        items = []
        for item in root.findall('.//item')[:limit]:
            title = item.findtext('title') or ""
            link = item.findtext('link') or ""
            pubDate = item.findtext('pubDate') or ""
            description = item.findtext('description') or ""
            source = None
            src_el = item.find('source')
            if src_el is not None:
                source = src_el.text

            items.append({
                "title": title,
                "link": link,
                "pubDate": pubDate,
                "description": description,
                "source": source,
            })

        # Suggerimenti su quali interrogazioni eseguire a valle
        suggestions = [
            f"Approfondire le fonti principali: cerca dichiarazioni ufficiali o comunicati relativi a '{category}'",
            f"Verificare le timeline: cerca 'cronologia {category} in ordine' o 'timeline {category}'",
            f"Cercare opinioni e analisi: 'analisi {category}' o 'commenti esperti {category}'",
            f"Verificare eventuali video o interviste: 'video {category}'",
            f"Cercare aggiornamenti locali: 'ultime notizie {category} {country}'",
        ]

        return {"status": "ok", "query": category, "results": items, "suggestions": suggestions}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ################################################################# 
# step 2: add tools schema so the agent is able to use it
# ################################################################# 

today_is_tool = {
    "type": "function",
    "function": {
        "name": "today_is",
        "description": "Get today's date",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

weather_tool = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Get current weather for a specific city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name"}},
            "required": ["city"],
        },
    },
}

day_of_week_tool = {
    "type": "function",
    "function": {
        "name": "day_of_week",
        "description": "Get today's the day of the week",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

add_tool = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "string", "description": "The first number to add"},
                "num2": {"type": "string", "description": "The second number to add"}
            },
            "required": ["num1", "num2"],
        },
    },
}

subtract_tool = {
    "type": "function",
    "function": {
        "name": "subtract",
        "description": "Subtract two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "string", "description": "The first number"},
                "num2": {"type": "string", "description": "The second number to subtract"}
            },
            "required": ["num1", "num2"],
        },
    },
}
            
multiply_tool = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "string", "description": "The first number to multiply"},
                "num2": {"type": "string", "description": "The second number to multiply"}
            },
            "required": ["num1", "num2"],
        },
    },
}

divide_tool = {
    "type": "function",
    "function": {
        "name": "divide",
        "description": "Divide two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "string", "description": "Dividendo"},
                "num2": {"type": "string", "description": "Divisore"}
            },
            "required": ["num1", "num2"],
        },
    },
}

square_tool = {
    "type": "function",
    "function": {
        "name": "square",
        "description": "square of a number.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "The number to square"}
            },
            "required": ["a"],
        },
    },
}


wikipedia_tool = {
    "type": "function",
    "function": {
        "name": "wikipedia",
        "description": (
            "Search Wikipedia for a query and return a JSON-serializable result.\n"
            "Implementation returns a dict with keys: 'status' ('ok' on success), 'query' (the input),"
            " and 'result' (the raw result returned by the WikipediaQueryRun)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user query (non-empty)"}
            },
            "required": ["query"],
        },
    },
}

RAG_tool = {
    "type": "function",
    "function": {
        "name": "RAG",
        "description": "Get relevant documents for a specific query from the vectorstore",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The RAG document"}},
            "required": ["query"],
        },
    },
}

wolfram_tool = {
    "type": "function",
    "function": {
        "name": "wolfram",
        "description": "Query Wolfram Alpha for computation and factual answers. Requires WOLFRAMALPHA_APPID env var or parameter 'appid'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The natural-language query to Wolfram Alpha"},
                "appid": {"type": "string", "description": "Optional Wolfram Alpha AppID (default from WOLFRAMALPHA_APPID env var)"},
                "timeout": {"type": "integer", "description": "Optional timeout in seconds", "default": 30}
            },
            "required": ["query"],
        },
    },
}

recent_news_tool = {
    "type": "function",
    "function": {
        "name": "recent_news",
        "description": "Recupera le notizie recenti (RSS Google News) per un category e suggerisce interrogazioni di approfondimento.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic" or "category": {"type": "string", "description": "Termine o tema da cercare (es. 'cronaca')"},
                "days": {"type": "integer", "description": "Numero di giorni nel passato da considerare (default 7)", "default": 7},
                "limit" or "count": {"type": "integer", "description": "Numero massimo di risultati da restituire (default 5)", "default": 5},
                "country": {"type": "string", "description": "Codice paese per localizzare Google News (es. 'IT')", "default": "IT"}
            },
            "required": ["category"],
        },
    },
}