"""
Agentic_RAG.py

Gradio chatbot interface for an agent that can call tools defined in `tools.py`.

Features implemented:
- Per-user persisted conversation history (file-based)
- Reset button to clear history for a user
- Ollama integration using environment variables OLLAMA_HOST and OLLAMA_MODEL.

Defaults: OLLAMA_HOST=http://10.100.14.73:11434, OLLAMA_MODEL=Qwen2.5:32b-instruct

This file provides a minimal, self-contained UI.
"""

from __future__ import annotations

from email import message
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import getpass
import inspect

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

import gradio as gr
import tools

# Try to import LangChain pieces; we'll fall back if LangChain isn't available
try:
    from langchain.agents import initialize_agent, AgentType
    try:
        # newer versions
        from langchain.agents import Tool as LC_Tool
    except Exception:
        # older versions
        from langchain.tools import Tool as LC_Tool
    from langchain.memory import ConversationBufferMemory
    from langchain.llms.base import LLM
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# abilitazione log in LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "default" 

# per evitare problemi di memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



# 
# GESTIONE LLM OLLAMA
#

# Read Ollama host and model from environment with sensible defaults
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://10.100.14.73:11434")
# OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# use the exact model requested by the user if not overridden by env
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "Qwen2.5:32b-instruct")
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "Mixtral:8x7B")
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:32b")

# Controls for encouraging longer, more detailed replies. These are best-effort
# instructions injected into the system prompt; some Ollama clients may also
# accept explicit generation params but we avoid calling unknown kwargs.
OLLAMA_MIN_WORDS = int(os.environ.get("OLLAMA_MIN_WORDS", "100"))
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.9"))
OLLAMA_VERBOSE_INSTRUCTION = os.environ.get("OLLAMA_VERBOSE_INSTRUCTION", """Rispondi in modo esteso e dettagliato. Verifica con attenzione la veridicità del materiale raccolto. Fornisci spiegazioni passo-passo e esempi dove utili.""") 
# Cerca di produrre almeno {OLLAMA_MIN_WORDS} parole.")


# tool_names is a comma-separated string of tool names, each prefixed with '/', e.g. '/tool1, /tool2, /tool3'
# This format is used in prompts and should be kept consistent for parsing and display.
tool_names = ['/RAG', '/add', '/day_of_week', '/divide', '/multiply', '/square', '/subtract', '/today_is', '/weather', '/wikipedia', '/wolfram']


# Gestione PROMPT

PREFIX = """
You are a highly intelligent assistant capable of understanding and executing specific actions based on user requests. 
Your goal is to assist the user as efficiently and accurately as possible without deviating from their instructions.
"""

FORMAT_INSTRUCTIONS = """
You have access to the following tools:

{tool_names}

Please follow these instructions carefully:

1. If you need to perform an action to answer the user's question, use the following format:
'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action in valid JSON to pass as dictionary for a given tool
Observation: the result of the action 
... (this Thought/Action/Action Input/Observation can repeat 3 times)
'''

2. If you can answer the user's question without performing any additional actions, or when missing Action after Thought, use the following format:
'''
Thought: Do I need to use a tool? No
Final Answer:  the final answer to the original input question
'''

Your responses should be concise and directly address the user's query. Avoid generating new questions or unnecessary information.
"""

SUFFIX = """
End of instructions. Please proceed with answering the user's question following the guidelines provided above.
Question: {input}
Thought:{agent_scratchpad}
Rispondi sempre e solo in lingua italiana.
"""

# Example of how to use this template in a LangChain agent setup
template = PREFIX + FORMAT_INSTRUCTIONS + SUFFIX


OLLAMA_SYSTEM_PROMPT = os.environ.get(
"OLLAMA_SYSTEM_PROMPT",
"""System prompt: Sei un agente autonomo che può usare strumenti locali {tool_names} per raccogliere informazioni, eseguire calcoli e compiere azioni banali. 
Seguire questo formato strutturato quando decidi cosa fare:
Thought: descrivi quale strategia stai considerando e perché.
Action: indica il nome del tool che intendi usare (formato: /toolname). Se non intendi usare alcun tool indica LLM
Action Input: fornisci l'input per il tool. L'input deve essere JSON valido o una lista di coppie chiave=valore (es. key="val").
Observation: incolla qui il risultato restituito dal tool (questo campo è popolato dall'ambiente che esegue il tool).
...(Ripeti Thought / Action / Action Input / Observation al massimo 5 volte)
Thought: quando dichiari di conoscere la risposta finale, fermati qui e non fare altre chiamate a tool.
Final answer: fornisci la risposta finale in lingua italiana.

Regole e vincoli:
Inizia sempre con una ricerca tramite RAG tool.
Poi fai seguire una ricerca con il tool Wikipedia per raccogliere ulteriori informazioni.
Usa gli altri tool solo quando necessario: prima di chiamare un tool, verifica che l'azione sia utile e che non si possa rispondere senza eseguire il tool.
Struttura dell'invocazione tool: usa esattamente il formato Action e Action Input indicato più sopra; l'ambiente eseguirà il tool e fornirà Observation.
Input dei tool: preferisci JSON per input complessi; se usi key=value, assicurati che siano non ambigui (usa virgolette per stringhe con spazi).
Error handling: se un tool fallisce o restituisce un errore, includi nell'Observation il nome del tool ed il messaggio d'errore. 
Genera quindi un nuovo Thought che decide se riprovare (magari con parametri diversi) o cambiare strategia.
Limite iterazioni: non superare 5 iterazioni Thought / Action / Action Input / Observation per singola risposta; se non risolvi entro tale limite, torna indietro e fornisci una spiegazione parziale e i passi successivi consigliati.
Non inventare risultati: tuttavia, se un tool fornisce informazioni incomplete, chiariscilo nel Thought e indica quali ulteriori tool o input sarebbero necessari.
Riservatezza e sicurezza: non invocare tool che possano esfiltrare dati sensibili se non autorizzato; se non sei sicuro, chiedi chiarimenti all'utente.
Formattazione della Final answer: per ogni affermazione fattuale, quando possibile, cita la fonte (es. nome del tool e/o file) tra parentesi quadre inline. 
Alla fine fornisci un breve livello di confidenza su una scala 0-100%.
Linguaggio: rispondi sempre in italiano.
Non riportare nella risposta indicazioni sul troubleshooting o errori interni dell'agente. """)


CUSTOM_PROMPT = os.environ.get("CUSTOM_PROMPT", 
f"""Custom prompt: Sei un assistente che può usare i tool locali: {tool_names} per rispondere alle domande. 
Rispondi sempre e solo in lingua italiana.                                     
Decidi autonomamente quale tool utilizzare.
Per domande articolate in più passaggi, ragiona passo dopo passo, utilizzando i tools disponibili.
Al termine genera una sintesi di tutte le informazioni raccolte. 
""")

# Try to import Ollama clients; prefer sync Client if available
OLLAMA_AVAILABLE = False
try:
    from ollama import Client as OllamaClient  # type: ignore
    OLLAMA_AVAILABLE = True
    
except Exception:
    try:
        from ollama import AsyncClient as OllamaAsyncClient  # type: ignore
        OLLAMA_AVAILABLE = True
        
    except Exception:
        OLLAMA_AVAILABLE = False

def ollama_chat_sync(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
#     """Call Ollama synchronously. messages is a list of dicts with role/content.
#     Uses blocking Client if available, otherwise runs AsyncClient via asyncio.
#     Returns the assistant reply string or an error string.
#     """

    """ Chiama Ollama in modo sincrono. messages è un elenco di dizionari con ruolo/contenuto.
    Utilizza il client sincrono se disponibile, altrimenti esegue il client asincrono tramite asyncio.
    Restituisci la stringa di risposta dell'assistente o una stringa di errore.
    """

    model_to_use = model or OLLAMA_MODEL
    if not OLLAMA_AVAILABLE:
        return "(Ollama not available)"
    try:
        if 'OllamaClient' in globals():
            client = OllamaClient(host=OLLAMA_HOST) if OLLAMA_HOST else OllamaClient()
            try: 
                resp = client.chat(model=model_to_use, messages=messages)
                            
            except Exception as e:
                return f"(Ollama error: {e})"
            
            # Normalize different client response shapes to a single string
            if hasattr(resp, 'content') and resp.content:
                return str(resp.content)
            
            if hasattr(resp, 'message') and hasattr(resp.message, 'content') and resp.message.content:
                return str(resp.message.content)
            return str(resp)
        else:
            import asyncio

            async def _call():
                client = OllamaAsyncClient(host=OLLAMA_HOST) if OLLAMA_HOST else OllamaAsyncClient()
                resp = await client.chat(model=model_to_use, messages=messages)
                if hasattr(resp, 'content') and resp.content:
                    return str(resp.content)
                if hasattr(resp, 'message') and hasattr(resp.message, 'content') and resp.message.content:
                    return str(resp.message.content)
                return str(resp)

            try:
                loop = asyncio.get_running_loop()
                # If we're here, there's an event loop running (e.g., Jupyter)
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(_call())
            except RuntimeError:
                # No event loop running
                return asyncio.run(_call())
    except Exception as e:
        return f"(Errore Ollama: {e})"


# --- LangChain LLM wrapper around the ollama_chat_sync function ---
if LANGCHAIN_AVAILABLE:
    try:
        # Simple Base LLM that forwards prompts to Ollama chat API
        class OllamaLLM(LLM):
            model: str = OLLAMA_MODEL
            host: Optional[str] = OLLAMA_HOST

            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:  # type: ignore[override]
                # Build a basic conversation and include a verbosity instruction so the
                # model is encouraged to generate longer, more detailed outputs.
                # system_base = os.environ.get("OLLAMA_SYSTEM_PROMPT", OLLAMA_SYSTEM_PROMPT)
                
                # system_content = f"{system_base}\n{OLLAMA_VERBOSE_INSTRUCTION}" if system_base else OLLAMA_VERBOSE_INSTRUCTION
                system_content = template
                                
                msgs = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
                resp = ollama_chat_sync(msgs, model=self.model)
                
                # ollama_chat_sync now normalizes to a string
                return str(resp)

            @property
            def _identifying_params(self):
                return {"model": self.model, "host": self.host}

            @property
            def _llm_type(self) -> str:
                return "ollama_custom"
    except Exception:
        LANGCHAIN_AVAILABLE = False


# Global singleton for Ollama LLM instance. Use the accessor get_ollama_llm()
# to lazily initialize it once and make it available to other modules/tools
# (for example the /RAG tool).
ollama_llm: Optional[LLM] = None


def get_ollama_llm() -> Optional[LLM]:
    """Return a singleton Ollama LLM instance (lazy init).

    If LangChain LLM wrapper is not available this will return None.
    """
    global ollama_llm
    if ollama_llm is None and LANGCHAIN_AVAILABLE:
        try:
            ollama_llm = OllamaLLM()
        except Exception:
            # If instantiation fails, keep None and let callers handle it
            ollama_llm = None
    return ollama_llm


# Helper: create LangChain Tools from the local `tools.registered_functions`
def _make_langchain_tools():
    """
    Create LangChain Tool objects from the locally registered functions in tools.registered_functions.

    Note: The closure issue with `make_fn` is handled by binding the current tool name as a default argument (n=name)
    to ensure each tool uses the correct function reference.
    """
    lc_tools = []

    def parse_input(text: str) -> dict:
        import json

        text = (text or "").strip()
        if not text:
            return {}
        # try JSON first
        try:
            return json.loads(text)
        except Exception:
            # try key=value pairs: a=1 b=2 or key="some value"
            out = {}
            import shlex
            import re
            # Use regex to find key="value" or key=value pairs
            pattern = re.compile(r'(\w+)\s*=\s*(".*?"|\'.*?\'|\S+)')
            for match in pattern.finditer(text):
                k = match.group(1)
                v = match.group(2)
                # Remove surrounding quotes if present
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                # try to cast numbers
                if v.isdigit():
                    out[k] = int(v)
                else:
                    out[k] = v
            return out

    for name in sorted(list(tools.registered_functions.keys())):
        def make_fn(n=name):
            def _call_tool(input_str: str) -> str:
                args = parse_input(input_str)
                # Fallback: if parsing returned empty but the input is a non-empty
                # plain string (e.g. "Albert Einstein"), try to interpret the
                # whole string as the single string parameter most tools expect
                # (commonly named 'query'). This prevents a TypeError like
                # "missing 1 required positional argument: 'query'" when the
                # agent provides a bare string.
                if not args and (input_str or "").strip():
                    try:
                        fn = tools.registered_functions.get(n)
                        if fn:
                            sig = inspect.signature(fn)
                            params = [p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
                            if len(params) == 1 :
                                args = {params[0]: input_str.strip()}
                            else:
                                # If the function explicitly declares 'query', prefer that
                                if "query" in params:
                                    args = {"query": input_str.strip()}
                    except Exception:
                        # if introspection fails, keep args as parsed (possibly empty)
                        pass

                res = tools.run_callable(n, args)
                try:
                    import json

                    return json.dumps(res, ensure_ascii=False)
                except Exception:
                    return str(res)

            return _call_tool

        description = f"Local tool wrapper for {name}"
        try:
            lc_tools.append(LC_Tool(name=name, func=make_fn(), description=description))
        except Exception:
            # fallback to old Tool signature
            try:
                from langchain.agents import Tool as ToolOld

                lc_tools.append(ToolOld(name=name, func=make_fn(), description=description))
            except Exception:
                # Give up adding this tool
                pass

    return lc_tools


# Directory to persist conversation histories per-user
HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "agent_histories")
os.makedirs(HIST_DIR, exist_ok=True)

# Memory key for conversation buffer memory (configurable)
MEMORY_KEY = "chat_history"

# 
# GESTIONE HISTORY 
#

def history_filepath(user_id: Optional[str]) -> str:
    if not user_id:
        user_id = getpass.getuser() or "default"
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    return os.path.join(HIST_DIR, f"history_{safe_id}.json")


def load_history(user_id: Optional[str]) -> List[Dict[str, Any]]:
    path = history_filepath(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return []
    return []

def save_history(user_id: Optional[str], messages: List[Dict[str, Any]]):
    path = history_filepath(user_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 
# CHATBOT COLLOQUIALE
#    

def bot_response(user_message: str, state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    # funzione di risposta alla generica interazione utente
    user_id = state.get("user_id") or getpass.getuser()
    messages: List[Dict[str, Any]] = state.get("messages") or []
    messages_answer: List[Dict[str, Any]] = state.get("messages") or []

    messages.append({"role": "user", "content": user_message})
    save_history(user_id, messages)

    answer = ""  # Ensure answer is always defined
    
    # build a LangChain agent that can use local tools and has a conversation memory.
    try:
        # Create or reuse a global agent stored on the module (simple singleton)
        global _AGENT_EXECUTOR, _AGENT_MEMORY
        global ollama_llm

        if "_AGENT_EXECUTOR" not in globals() or _AGENT_EXECUTOR is None:
            # lazily initialize the shared Ollama LLM singleton
            ollama_llm = get_ollama_llm()
            lc_tools = _make_langchain_tools()
            # Create conversation memory
            _AGENT_MEMORY = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
            _AGENT_EXECUTOR = initialize_agent(
                tools=lc_tools,
                llm=ollama_llm,
                # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=_AGENT_MEMORY,
                verbose=True,
                handle_parsing_errors=True,
                # max_iterations=10,
                # max_execution_time=600,
            )

        # Build a prompt that includes recent history so the agent can reference it (we also have memory enabled)
        hist_text = ""
        for m in messages[-20:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                hist_text += f"User: {content}\n"
            else:
                hist_text += f"Assistant: {content}\n"

        # Run the agent with history + current message
        prompt_for_agent = f"{CUSTOM_PROMPT}\n{hist_text}User: {user_message}\nAssistant:"
        agent_output = _AGENT_EXECUTOR.run(prompt_for_agent)
        assistant_reply = agent_output
        
    except Exception as e:
        assistant_reply = f"(Errore agent LangChain: {e})\nTool disponibili: {', '.join(sorted(list(tools.registered_functions.keys())))}"

    # aggiornamento conversazione
    messages_answer = {"role": "assistant", "content": assistant_reply}
    messages.append(messages_answer)
        
    # aggiornamento stato agente
    state["messages"] = messages
    
    # salvataggio conversazione
    save_history(user_id, messages)
     
    # clear the input question textbox by returning an empty string as third output
    return messages, state, ""

# 
# RESET HISTORY
#
def reset_conversation(user_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # cancellazione history
    path = history_filepath(user_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    state = {"user_id": user_id, "messages": []}
    
    return [], state

# 
# USER INTERFACE
#
def build_demo():
    global vectorstore    
    
    with gr.Blocks(title="Agentic RAG - tools + Ollama") as demo:
        gr.Markdown("""
        ## Agentic RAG
        Invia messaggi all'agente che ti risponderà a breve.
        """)

        chatbot = gr.Chatbot(label="Conversazione", type="messages", autoscroll=True, show_copy_all_button=True)
        state = gr.State({"user_id": None, "messages": []})
        
        with gr.Row():
            with gr.Column(scale=9):
                question = gr.Textbox(label="Domanda", placeholder="Scrivi qui la domanda")
                
            with gr.Column(scale=1):
                send = gr.Button("Invia")
                reset_btn = gr.Button("Reset")

                # gestione sessione per utente collegato
                user_id_inp = gr.Textbox(label="User ID:", value=getpass.getuser(), visible=False)
            
        def set_user_and_load(uid: str, current_state: Dict[str, Any]):
            if uid and uid.strip():
                messages = load_history(uid)
                return {"user_id": uid, "messages": messages}, messages
            return current_state, current_state.get("messages") or []

        user_id_inp.change(fn=set_user_and_load, inputs=[user_id_inp, state], outputs=[state, chatbot])

        # after sending, update chatbot and state and clear the question textbox
        send.click(fn=bot_response, inputs=[question, state], outputs=[chatbot, state, question])

        # also submit the same callback when pressing Enter/Return inside the textbox
        # this makes Enter act like pressing the "Invia" button
        question.submit(fn=bot_response, inputs=[question, state], outputs=[chatbot, state, question])

        def reset_for_user(uid: str):
            _, state = reset_conversation(uid if uid and uid.strip() else None)
            # also clear the question textbox by returning empty string as third output
            return [], state, ""

        # rinizializza la conversazione e pulisce il campo di input
        reset_btn.click(fn=reset_for_user, inputs=[user_id_inp], outputs=[chatbot, state, question])
        
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1")
