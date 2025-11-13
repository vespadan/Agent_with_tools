"""
Agentic_RAG.py

Gradio chatbot wrapper for an agent that can call tools defined in `tools.py`.

Features implemented:
- Per-user persisted conversation history (file-based)
- Simple tool invocation syntax: /toolname key=val or /toolname {"key": val}
- Dropdown to run a tool manually with JSON args
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
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "default" 

# per evitare problemi di memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Read Ollama host and model from environment with sensible defaults
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://10.100.14.73:11434")
# use the exact model requested by the user if not overridden by env
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "Qwen2.5:32b-instruct")

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


def parse_tool_invocation(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse a tool invocation in the form:
    /toolname key=val key2=val2
    or
    /toolname {"key": "val"}
    Returns (tool_name, args_dict) or None.
    """
    text = text.strip()
    m = re.match(r"^/([\w_\-]+)(?:\s+(.*))?$", text)
    if not m:
        return None

    name = m.group(1)
    args_str = (m.group(2) or "").strip()
    if not args_str:
        return (name, {})

    # try json
    if args_str.startswith("{"):
        try:
            args = json.loads(args_str)
            if isinstance(args, dict):
                return (name, args)
        except Exception:
            pass

    # key=value parser
    args: Dict[str, Any] = {}
    pairs = re.findall(r"(\w+)=('(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"|\[[^\]]*\]|[^\s]+)", args_str)
    if not pairs:
        # If we didn't find explicit key=val pairs, try to infer a tool from natural text.
        lower = args_str.lower()
        # weather detection: look for city names after keywords 'meteo', 'weather', 'che tempo'
        if any(k in lower for k in ("meteo", "weather", "che tempo", "tempo a", "temperature")):
            # try to pick the last word as city if present
            tokens = re.findall(r"[\w\-']+", args_str)
            city = tokens[-1] if tokens else ""
            return ("weather", {"city": city})
        # today / day_of_week detection
        if any(k in lower for k in ("oggi", "che giorno", "giorno della settimana", "day of week", "what day")):
            return ("day_of_week", {})
        if any(k in lower for k in ("data", "oggi è", "what is the date", "what's the date", "today")):
            return ("today_is", {})
        # arithmetic detection: look for simple expressions like '2+2', 'sum of 3 and 5', 'add 2 and 3'
        if re.search(r"\b(add|sum|plus|minus|subtract|multiply|times|divide|diviso)\b", lower) or re.search(r"\d+\s*[+\-*/]\s*\d+", args_str):
            # try to extract two numbers
            nums = re.findall(r"-?\d+\.?\d*", args_str)
            if len(nums) >= 2:
                a = int(float(nums[0]))
                b = int(float(nums[1]))
                # determine operation
                if re.search(r"\b(add|sum|plus)\b", lower) or re.search(r"\+", args_str):
                    return ("add", {"a": a, "b": b})
                if re.search(r"\b(subtract|minus)\b", lower) or re.search(r"-", args_str):
                    return ("subtract", {"a": a, "b": b})
                if re.search(r"\b(multiply|times)\b", lower) or re.search(r"\*", args_str):
                    return ("multiply", {"a": a, "b": b})
                if re.search(r"\b(divide|diviso)\b", lower) or re.search(r"/", args_str):
                    return ("divide", {"a": a, "b": b})
        # wikipedia detection
        if any(k in lower for k in ("wikipedia", "wiki", "chi è", "cosa è", "che cos'è", "informazioni su", "info su")):
            # try to extract a query portion after keywords
            # naive approach: remove common leading words
            query = re.sub(r"(?i)^(che cos'?è|chi è|cosa è|info su|informazioni su|wikipedia|wiki)\s+", "", args_str).strip()
            if not query:
                query = args_str
            return ("wikipedia_search", {"query": query})
        # fallback: treat whole string as a generic positional query
        return (name, {"q": args_str})

    for k, v in pairs:
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        elif v.lower() in ("null", "none"):
            v = None
        elif re.fullmatch(r"-?\d+", v):
            v = int(v)
        elif re.fullmatch(r"-?\d+\.\d+", v):
            try:
                v = float(v)
            except Exception:
                pass
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if inner:
                parts = [p.strip().strip('"\'') for p in inner.split(",")]
                new_parts = []
                for p in parts:
                    if re.fullmatch(r"-?\d+", p):
                        new_parts.append(int(p))
                    elif re.fullmatch(r"-?\d+\.\d+", p):
                        try:
                            new_parts.append(float(p))
                        except Exception:
                            new_parts.append(p)
                    else:
                        new_parts.append(p)
                v = new_parts
        args[k] = v

    return (name, args)


def call_tool(name: str, arguments: Dict[str, Any]) -> str:
    try:
        result = tools.run_callable(name, arguments)
        if isinstance(result, dict):
            if "error" in result:
                return f"Error calling {name}: {result['error']}"
            try:
                return json.dumps(result, ensure_ascii=False, indent=2)
            except Exception:
                return str(result)
        else:
            return str(result)
    except Exception as e:
        return f"Exception while calling tool {name}: {e}"


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
    

def bot_response(user_message: str, state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], ]:
    user_id = state.get("user_id")
    messages: List[Dict[str, Any]] = state.get("messages") or []
    messages_answer: List[Dict[str, Any]] = state.get("messages") or []

    messages.append({"role": "user", "content": user_message})
    save_history(user_id, messages)

    parsed = parse_tool_invocation(user_message)
    if parsed:
        name, args = parsed
        assistant_text = f"Eseguo lo strumento '/{name}' con argomenti: {args}"
        messages.append({"role": "assistant", "content": assistant_text})
        tool_output = call_tool(name, args)
        messages.append({"role": "assistant", "content": f"Risultato di /{name}: {tool_output}"})
        state["messages"] = messages
        save_history(user_id, messages)
        return messages, state

    answer = ""  # Ensure answer is always defined
    # use Ollama if available
    if OLLAMA_AVAILABLE:
        try:
            convo = [{"role": m["role"], "content": m["content"]} for m in messages[-12:]]
            system_msg = {"role": "system", "content": "Sei un assistente che può usare tool locali come /add, /weather, ecc."}
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
     
    # return messages_answer, state
    return messages, state


def reset_conversation(user_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = history_filepath(user_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    state = {"user_id": user_id, "messages": []}
    return [], state


def build_demo():
    with gr.Blocks(title="Agentic RAG - tools + Ollama") as demo:
        gr.Markdown("""
        ## Agentic RAG
        Invia messaggi all'agente. Per chiamare un tool usa:
        `/toolname key=val` oppure `/toolname {"key": "val"}`
        """)

        chatbot = gr.Chatbot(label="Conversazione", type="messages")
        state = gr.State({"user_id": None, "messages": []})

        user_id_inp = gr.Textbox(label="User ID (opzionale)", placeholder="Inserisci ID per persistere la conversazione")

        with gr.Row():
            txt = gr.Textbox(label="Messaggio", placeholder="Scrivi qui...\nUsa /add a=1 b=2 per chiamare un tool")
            send = gr.Button("Invia")

        with gr.Row():
            tool_dropdown = gr.Dropdown(choices=sorted(list(tools.registered_functions.keys())), label="Tool (opzionale)")
            tool_args = gr.Textbox(label="Argomenti JSON (opzionale)", placeholder='{"a":1, "b":2}')
            run_tool_btn = gr.Button("Esegui tool")
            reset_btn = gr.Button("Reset")

        def set_user_and_load(uid: str, current_state: Dict[str, Any]):
            if uid and uid.strip():
                messages = load_history(uid)
                return {"user_id": uid, "messages": messages}, messages
            return current_state, current_state.get("messages") or []

        user_id_inp.change(fn=set_user_and_load, inputs=[user_id_inp, state], outputs=[state, chatbot])

        send.click(fn=bot_response, inputs=[txt, state], outputs=[chatbot, state])

        def exec_selected_tool(tool_name: str, json_args: str, state: Dict[str, Any]):
            user_id = state.get("user_id")
            messages = state.get("messages") or []
            if not tool_name:
                messages.append({"role": "assistant", "content": "Nessun tool selezionato."})
                state["messages"] = messages
                save_history(user_id, messages)
                return messages, state
            try:
                args = json.loads(json_args) if json_args and json_args.strip() else {}
            except Exception as e:
                messages.append({"role": "assistant", "content": f"Errore nel parsing JSON degli argomenti: {e}"})
                state["messages"] = messages
                save_history(user_id, messages)
                return messages, state
            messages.append({"role": "user", "content": f"/ {tool_name} (invocazione manuale)"})
            messages.append({"role": "assistant", "content": f"Eseguo {tool_name} con {args}"})
            out = call_tool(tool_name, args)
            messages.append({"role": "assistant", "content": f"Risultato di {tool_name}: {out}"})
            state["messages"] = messages
            save_history(user_id, messages)
            return messages, state

        run_tool_btn.click(fn=exec_selected_tool, inputs=[tool_dropdown, tool_args, state], outputs=[chatbot, state])

        def reset_for_user(uid: str):
            _, state = reset_conversation(uid if uid and uid.strip() else None)
            return [], state

        reset_btn.click(fn=reset_for_user, inputs=[user_id_inp], outputs=[chatbot, state])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0")
