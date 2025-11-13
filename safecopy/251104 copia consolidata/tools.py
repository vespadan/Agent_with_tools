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
import traceback
import os
from typing import Callable, Optional, Union
from langchain.chains import RetrievalQA

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

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

        if callable(func):
            return func(**arguments)
        else:
            return {"error": f"Function '{name}' is not callable or not found"}
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
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return int(a) + int(b)

@register_function
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return int(a) - int(b)

@register_function
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return int(a) * int(b)

@register_function
def divide(a: int, b: int) -> int:
    """Divide two numbers."""
    return int(a) / int(b)

@register_function
def square(a) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return int(a) * int(a)

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
        res = client.query(query, timeout=timeout)
        # Extract a readable answer from pods/results when possible
        answer = None
        try:
            if hasattr(res, "results") and res.results is not None:
                answer = next(res.results).text
        except Exception:
            answer = None

        if not answer:
            # Fallback to string representation
            answer = str(res)

        return {"status": "ok", "query": query, "result": answer}
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
                "a": {"type": "integer", "description": "The first number to add"},
                "b": {"type": "integer", "description": "The second number to add"}
            },
            "required": ["a", "b"],
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
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number to subtract"}
            },
            "required": ["a", "b"],
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
                "a": {"type": "integer", "description": "The first number to multiply"},
                "b": {"type": "integer", "description": "The second number to multiply"}
            },
            "required": ["a", "b"],
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
                "a": {"type": "integer", "description": "Dividendo"},
                "b": {"type": "integer", "description": "Divisore"}
            },
            "required": ["a", "b"],
        },
    },
}

square_tool = {
    "type": "function",
    "function": {
        "name": "square",
        "description": "square of a numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The number to square"}
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