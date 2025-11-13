"""
Agentic_RAG.py

Gradio chatbot interface for an agent that can call tools defined in `tools.py`.

Features implemented:
- Per-user persisted conversation history (file-based)
- Reset button to clear history for a user
- Ollama integration using environment variables OLLAMA_HOST and OLLAMA_MODEL.

Defaults: OLLAMA_HOST=http://10.100.14.73:11434, OLLAMA_MODEL=Qwen2.5:32b-instruct

This file is based on an example in the repository and provides a minimal, self-contained UI.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

import tools

# abilitazione log in LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

os.environ["LANGSMITH_PROJECT"] = "default" 

# per evitare problemi di memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# 
# GESTIONE LLM OLLAMA
#

# Read Ollama host and model from environment with sensible defaults
# OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://10.100.14.73:11434")
# use the exact model requested by the user if not overridden by env
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "Qwen2.5:32b-instruct")

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
    """Call Ollama synchronously. messages is a list of dicts with role/content.
    Uses blocking Client if available, otherwise runs AsyncClient via asyncio.
    Returns the assistant reply string or an error string.
    """
    model_to_use = model or OLLAMA_MODEL
    if not OLLAMA_AVAILABLE:
        return "(Ollama not available)"
    try:
        if 'OllamaClient' in globals():
            client = OllamaClient(host=OLLAMA_HOST) if OLLAMA_HOST else OllamaClient()
            resp = client.chat(model=model_to_use, messages=messages)
            return getattr(resp, 'content', str(resp)) or str(resp), resp.message.content
        else:
            import asyncio

            async def _call():
                client = OllamaAsyncClient(host=OLLAMA_HOST) if OLLAMA_HOST else OllamaAsyncClient()
                resp = await client.chat(model=model_to_use, messages=messages)
                return getattr(resp, 'content', str(resp)) or str(resp)

            return asyncio.run(_call())
    except Exception as e:
        return f"(Errore Ollama: {e})"

# Directory to persist conversation histories per-user
HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "agent_histories")
os.makedirs(HIST_DIR, exist_ok=True)

# 
# GESTIONE HISTORY 
#

def history_filepath(user_id: Optional[str]) -> str:
    if not user_id:
        user_id = "anonymous"
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
    user_id = state.get("user_id")
    messages: List[Dict[str, Any]] = state.get("messages") or []
    messages_answer: List[Dict[str, Any]] = state.get("messages") or []

    messages.append({"role": "user", "content": user_message})
    save_history(user_id, messages)

    answer = ""  # Ensure answer is always defined
    
    # custom_prompt2 = """Sei un assistente intelligente in grado di utilizzare tool per rispondere alle domande: 
    # documenti aziendali, calcolatrice, Wikipedia, previsioni meteo, la history delle precedenti conversazioni.
    # Rispondi sempre in italiano.
    # Decidi autonomamente quale altro tool è da utilizzare.
    # Per domande articolate in più passaggi, ragiona passo dopo passo, utilizzando i tools disponibili.
    # Al termine genera una sintesi di tutte le informazioni raccolte.
    # A conclusione indica tra parentesi gli eventuali tool che hai utilizzato per rispondere alla domanda. Fai precedere l'elenco da 'Tools utilizzati:'
    # """

    custom_prompt = """Sei un assistente che può usare i tool locali come:  /today_is, /weather, day_of_week, /add, /subtract, /multiply,  /divide, /square, /wikipedia_search per rispondere alle domande
    Rispondi sempre in italiano.
    Decidi autonomamente quale tool è da utilizzare.
    Per domande articolate in più passaggi, ragiona passo dopo passo, utilizzando i tools disponibili.
    Al termine genera una sintesi di tutte le informazioni raccolte.
    A conclusione indica tra parentesi gli eventuali tool locali che hai utilizzato per rispondere alla domanda. Fai precedere l'elenco da 'Tools utilizzati:'
    """
    
    # use Ollama if available
    if OLLAMA_AVAILABLE:
        try:
            convo = [{"role": m["role"], "content": m["content"]} for m in messages[-12:]]
            system_msg = {"role": "system", "content": custom_prompt}
            ollama_msgs = [system_msg] + convo
            assistant_reply, answer = ollama_chat_sync(ollama_msgs, model=OLLAMA_MODEL)
        except Exception as e:
            assistant_reply = f"(Errore Ollama: {e})\nSe vuoi usare un tool locale invia /toolname ...\nTool disponibili: {', '.join(sorted(list(tools.registered_functions.keys())))}"
    else:
        assistant_reply = (
            "Ho ricevuto il tuo messaggio. Se vuoi usare uno dei tool locali, invia un comando nel formato:\n"
            "/toolname arg1=val1 arg2=val2  oppure /toolname {\"arg1\": val}\n"
            "Tool disponibili: " + ", ".join(sorted(list(tools.registered_functions.keys())))
        )

    # aggiornamento conversazione
    messages_answer = {"role": "assistant", "content": answer}
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
    with gr.Blocks(title="Agentic RAG - tools + Ollama") as demo:
        gr.Markdown("""
        ## Agentic RAG
        Invia messaggi all'agente che ti risponderà a breve.
        """)

        chatbot = gr.Chatbot(label="Conversazione", type="messages", autoscroll=True, show_copy_all_button=True)
        state = gr.State({"user_id": None, "messages": []})
        
        with gr.Row():
            question = gr.Textbox(label="Messaggio", placeholder="Scrivi qui la domanda")
            
        with gr.Row():
            user_id_inp = gr.Textbox(label="User ID (opzionale)", placeholder="Inserisci ID per persistere la conversazione", visible=False)
            send = gr.Button("Invia")
            reset_btn = gr.Button("Reset")
 
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
