import gradio as gr
import httpx
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.tools import WikipediaQueryRun  
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import os
os.environ['USER_AGENT'] = 'myagent'

# Configure remote Ollama server and model via environment variables.
# Example: export OLLAMA_HOST="http://10.0.0.1:11434"; export OLLAMA_MODEL="llama3.2:latest"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://10.100.14.73:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "Qwen2.5:32b-instruct")   
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

# Ensure libraries that read env vars can pick this up
os.environ.setdefault("OLLAMA_HOST", OLLAMA_HOST)
os.environ.setdefault("OLLAMA_MODEL", OLLAMA_MODEL)

import asyncio
from ollama import AsyncClient

async def main():
    client = AsyncClient(host=OLLAMA_HOST)
    response = await client.chat(
        model=OLLAMA_MODEL,
        messages=[{'role': 'user', 'content': 'Ciao, come stai?'}],
    )
    print('Response:', response['message']['content'])

asyncio.run(main())

# abilitazione log in LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

os.environ["LANGSMITH_PROJECT"] = "default" 
# per evitare problemi di memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Module-level singletons to preserve agent and memory across calls
_agent_singleton = None
_memory_store = None

def calculator_run(expression: str) -> str:
    """ Calcolatrice semplice e limitata da utilizzare come strumento: valuta l'espressione utilizzando le funzioni e gli operatori del modulo matematico, disabilitando al contempo i componenti predefiniti.  """
    import math
    try:
        # restrict globals to math functions only
        result = eval(expression, {"__builtins__": None}, math.__dict__)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def wikipedia_search(query: str) -> str:
    """ Ricerche sulla base dati wikipedia in italiano """
    try:
        api_wrapper = WikipediaAPIWrapper(language="it")
        return WikipediaQueryRun(api_wrapper=api_wrapper).run(query)
    except Exception as e:
        return f"Error: {e}"
    
def weather_run(city: str) -> str:
    """ Ricerche meteo su una città specifica """
    try:
        base_url: str = f"http://wttr.in/{city}?format=j1"
        response = httpx.Client(follow_redirects=True, default_encoding="utf-8").get(base_url)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            temp = data["current_condition"][0]["temp_C"]
            return f"The temperature in {city} is {temp} Celsius degree"
        else:
            return f"Could not retrieve weather data for {city}."
    except Exception as e:
        return f"Error: {e}"
    
def history_chain (query: str) -> str:
    """ Ricerche nel contesto delle precedenti conversazioni """
    try:
        if _memory_store is None:
            return "La memoria della conversazione non è ancora stata inizializzata."
        return _memory_store.load_memory_variables({})["history"]
    except Exception as e:
        return f"Error: {e}"

def load_model(model: str | None = None) -> ChatOllama:
    # ritorna un'istanza LLM ChatOllama
    # allow passing model name or use configured OLLAMA_MODEL
    model_name = model if model and not model.startswith("OLLAMA_") else OLLAMA_MODEL
    # make sure langchain_ollama picks the configured host
    os.environ.setdefault("OLLAMA_HOST", OLLAMA_HOST)
    return ChatOllama(model=model_name)

def build_agent(user_memory=None):
    # Persist agent and memory so they survive across user queries.
    # This avoids recreating the agent (and losing conversation history) on every call.
    global _agent_singleton, _memory_store

    if _agent_singleton is not None and user_memory is None:
        return _agent_singleton

    LLM_Ollama = load_model()
    documents = TextLoader("/home/vespa/VStudioProjects/Agent_with_tools/docs/trascrizione.txt").load()
    vectorstore = FAISS.from_documents(documents, OllamaEmbeddings(model=OLLAMA_MODEL))
    
    rag_chain = RetrievalQA.from_llm(
        llm=LLM_Ollama,
        retriever=vectorstore.as_retriever()
    )
 
    rag_tool = Tool(
        name="RAG",
        func=rag_chain.run,
        description="Utilizza questo tool per rispondere alle domande sui documenti aziendali e sul vectorstore. Rispondi sempre in italiano."
    )
    
    calc_tool = Tool(
        name="Calculator",
        func=calculator_run,
        description="Utilizza questo tool per operazioni matematiche e calcoli."
    )
    
    wiki_tool = Tool(
        name="wikipedia",
        func= wikipedia_search,
        description="Utilizza questo tool per cercare su Wikipedia informazioni di carattere generale."
    )
    
    weather_tool = Tool(
        name="weather",
        func=weather_run,
        description="Utilizza questo tool per ottenere le previsioni meteo per una città specifica."
    )
    
    history_tool = Tool(
        name="Contesto",
        func=history_chain,
        description="Utilizza questo tool per acceder alla precedente conversazione."
    )   
    
    # caricamento tools disponibili
    tools = [history_tool, rag_tool, calc_tool, wiki_tool, weather_tool]
    

    custom_prompt = """Sei un assistente intelligente in grado di utilizzare tool per utilizzare: documenti aziendali, calcolatrice, Wikipedia, previsioni meteo, la history delle precedenti conversazioni.
    Rispondi sempre in italiano.
    Utilizza sempre per primo il tool history_tool per ottenere il contesto delle conversazioni precedenti.
    Decidi poi quale altro tool è da utilizzare.
    Per domande articolate in più passaggi, ragiona passo dopo passo, utilizzando i tools disponibili.
    Al termine genera una sintesi di tutte le informazioni raccolte.
    """

    prompt_template = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template=custom_prompt + "\n{input}\n{agent_scratchpad}"
    )
    
    # create a persistent memory store (keeps conversation across calls)
    if user_memory is not None:
        memory = user_memory
    else:
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        _memory_store = memory

    _agent_singleton = initialize_agent(
        tools=tools,
        llm=LLM_Ollama,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=60,
        early_stopping_method="generate",
        memory=memory,
        prompt_template=prompt_template,
    )
     
    return _agent_singleton
  
def agentic_rag_interface(user_query, user_memory=None):
    """
    Handle a single user query while preserving per-session memory.

    Gradio will keep `user_memory` (a ConversationBufferMemory instance) per session
    when we declare it as a `gr.State()` input/output. If `user_memory` is None,
    we create a fresh ConversationBufferMemory for that session.

    Note:
        `user_memory` must be a ConversationBufferMemory instance with memory_key="history"
        to match the agent's expectations. If not provided, a new ConversationBufferMemory
        with memory_key="history" will be created.
    """
    # If no per-user memory provided, create one — gr.State will keep it per session
    if user_memory is None:
        user_memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Build the agent with the correct memory instance
    agent = build_agent(user_memory=user_memory)
    result = agent.run({"input": user_query})
    return result, user_memory


# Use gr.State() to keep a ConversationBufferMemory instance per Gradio session.
demo = gr.Interface(
    fn=agentic_rag_interface,
    inputs=[gr.Textbox(lines=2, label="Query"), gr.State()],
    outputs=[gr.Textbox(label="Answer"), gr.State()],
    title="Agentic RAG con Ollama LLM",
    description="Fai domande complesse! L'agente userà i tool più adatti per rispondere."
)

if __name__ == "__main__":
    demo.launch()
