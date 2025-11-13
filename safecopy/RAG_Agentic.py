
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from tools import (
    run_callable,
    today_is_tool,
    weather_tool,
    day_of_week_tool,
    add_tool,
    multiply_tool,
    divide_tool,
    square_tool,
    wiki_tool
)

# NOTE: tools.py exposes function-calling schemas as plain dicts (for defining json-schema
# parameters). initialize_agent/Agents expect Tool objects. This module wraps those dicts
# into langchain 'Tool' instances that call run_callable(name, args) at runtime.


def load_model(model: str = None) -> ChatOllama:
    # ritorna un'istanza LLM ChatOllama
    return ChatOllama(model="llama3.2:latest")

LLM_Ollama=load_model("llama3.2:latest")

def build_agent():
    # carica documenti e costruisci vectorstore
    documents = TextLoader("/home/vespa/VStudioProjects/Agent_with_tools/docs/trascrizione.txt").load()
    vectorstore = FAISS.from_documents(documents, OllamaEmbeddings(model="llama3.2:latest"))

    # Tools definiti localmente
    # RAG chain
    rag_chain = RetrievalQA.from_llm(
        llm=LLM_Ollama,
        retriever=vectorstore.as_retriever()
    )

    # RAG tool
    rag_tool = Tool(
        name="RAG",
        func=rag_chain.run,
        description="Use this tool to answer questions from company docs and the vector store."
    )

    # tools disponibili     
    # The local 'tools.py' exposes function schemas (dicts) for function-calling style tools.
    # Initialize actual langchain Tool objects that wrap a call to run_callable(name, args).
    def make_function_tool(schema: dict) -> Tool:
        name = schema.get("function", {}).get("name")
        description = schema.get("function", {}).get("description", "")
        # Inspect parameter schema to map free-text input to the correct argument names
        param_props = schema.get("function", {}).get("parameters", {}).get("properties", {}) or {}
        required_keys = schema.get("function", {}).get("parameters", {}).get("required", []) or list(param_props.keys())
     
        return Tool(name=name, func=run_callable, description=description)

    tools = [
        today_is_tool,
        weather_tool,
        day_of_week_tool,
        add_tool,
        multiply_tool,
        divide_tool,
        square_tool,
        wiki_tool,
        # rag_tool,
    ]

    custom_prompt = """You are an intelligent assistant that can use company docs, Wikipedia or other tools
    Decide which tool is most appropriate. For multi-step questions, reason step-by-step, using tools as needed, and synthesize a final answer."""
  
    # inizializza l'agente con ChatOllama
    # Ensure every entry in `tools` is a langchain `Tool` instance.
    normalized_tools = []
    for t in tools:
        # If it's already a Tool, keep it
        if isinstance(t, Tool):
            normalized_tools.append(t)
            continue

        # If it's a dict schema (from tools.py), convert it to a Tool
        if isinstance(t, dict):
            try:
                normalized_tools.append(make_function_tool(t))
                continue
            except Exception as e:
                # fallback: create a dummy Tool that returns an error when called
                def _err(s: str, _e=e):
                    return f"Invalid tool schema: {_e}"
                normalized_tools.append(Tool(name=str(t.get('function', {}).get('name', 'unknown_tool')),
                                             func=_err,
                                             description="Invalid tool schema - auto-wrapped"))
                continue

        # Unknown type: wrap into a Tool that reports the type error at runtime
        def _unknown(s: str, _obj=t):
            return f"Unsupported tool object of type {type(_obj)}: {_obj}"
        normalized_tools.append(Tool(name=f"unsupported_{len(normalized_tools)}",
                                     func=_unknown,
                                     description=f"Automatically-wrapped unsupported tool of type {type(t)}"))

    # use normalized_tools when initializing the agent
    tools = normalized_tools

    agent = initialize_agent(
        tools=tools,
        llm=LLM_Ollama,
        agent="zero-shot-react-description",
        verbose=False,
        agent_kwargs={"prefix": custom_prompt},
        max_iterations=100,  # Limit the number of agent iterations (tool-calls / reasoning steps)
    )
    return agent

# costruisci l'agente 
agent = build_agent()

def agentic_rag_interface(user_query):
    try:
        return agent.run(user_query)
    except Exception as e:
        return f"Agent error: {e}"

if __name__ == "__main__":

    try:
        import gradio as gr

        demo = gr.Interface(
            fn=agentic_rag_interface,
            inputs="text",
            outputs="text",
            title="Agentic RAG Demo",
            description="Fai domande complesse! L'agente userà retrieval, Wikipedia, e calcolatrice se serve."
        )
        demo.launch()
        
    except Exception as e:
        print(f"Gradio not available or failed to launch demo: {e}")
