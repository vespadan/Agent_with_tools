"""
Agentic_RAG.py

Interfaccia Gradio minimale per un agente che può usare i tool definiti in `tools.py`.

Caratteristiche:
- Gestione della storia della conversazione per sessione (stato Gradio)
- Rilevamento semplice di invocazioni di tool usando la sintassi:
  /toolname arg1=val1 arg2=val2
  oppure: /toolname {"arg1": 1, "arg2": "val"}
- Dropdown per scegliere un tool e campo JSON per passare argomenti
- Pulsante "Reset" per iniziare una nuova conversazione

Nota: Questo agente non usa API esterne di LLM; mostra come integrare ed eseguire
i tool locali definiti in `tools.py`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import gradio as gr

import tools

# Optional Ollama integration (only used if OLLAMA_HOST/OLLAMA_MODEL are set and ollama is installed)
OLLAMA_AVAILABLE = False
OLLAMA_HOST = os.environ.get("OLLAMA_HOST")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
try:
    # prefer AsyncClient where available in repo examples
    from ollama import Client as OllamaClient
    OLLAMA_AVAILABLE = True
except Exception:
    try:
        from ollama import AsyncClient as OllamaAsyncClient  # type: ignore
        OLLAMA_AVAILABLE = True
    except Exception:
        OLLAMA_AVAILABLE = False


def ollama_chat_sync(messages: List[Dict[str, str]], model: str | None = None) -> str:
    """Call Ollama synchronously. messages is a list of dicts with role/content.

    This uses the blocking Client if available; otherwise uses AsyncClient via asyncio.run.
    Returns the assistant reply string.
    """
    model_to_use = model or OLLAMA_MODEL
    try:
        # use blocking client if present
        if 'OllamaClient' in globals():
            client = OllamaClient(host=OLLAMA_HOST) if OLLAMA_HOST else OllamaClient()
            resp = client.chat(model=model_to_use, messages=messages)
            return getattr(resp, 'content', str(resp)) or str(resp)
        else:
            # fallback to async client
            import asyncio

            async def _call():
                client = OllamaAsyncClient(host=OLLAMA_HOST) if OLLAMA_HOST else OllamaAsyncClient()
                resp = await client.chat(model=model_to_use, messages=messages)
                return getattr(resp, 'content', str(resp)) or str(resp)

            return asyncio.run(_call())
    except Exception as e:
        return f"(Errore Ollama: {e})"

# directory to persist conversation histories per-user
HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "agent_histories")
os.makedirs(HIST_DIR, exist_ok=True)


def parse_tool_invocation(text: str) -> Tuple[str, Dict[str, Any]] | None:
    """Parse a simple tool invocation command.

    Supported forms:
    - /toolname key=val key2=val2
    - /toolname {"key": "val", ...}
    Returns (tool_name, args_dict) or None if no invocation found.
    """
    text = text.strip()
    m = re.match(r"^/(\w+)(?:\s+(.*))?$", text)
    if not m:
        return None

    name = m.group(1)
    args_str = (m.group(2) or "").strip()

    if not args_str:
        return (name, {})

    # Try JSON first
    if args_str.startswith("{"):
        try:
            args = json.loads(args_str)
            if isinstance(args, dict):
                return (name, args)
        except Exception:
            pass

    # Parse key=value pairs separated by spaces; support quoted strings and basic lists (comma-separated)
    args: Dict[str, Any] = {}
    pairs = re.findall(r"(\w+)=('(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"|\[[^\]]*\]|[^\s]+)", args_str)
    if not pairs:
        # If nothing parsed, treat the whole remaining string as a single positional argument
        return (name, {"q": args_str})

    for k, v in pairs:
        # strip quotes
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        # try boolean/null
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        elif v.lower() in ("null", "none"):
            v = None
        # try int/float
        elif re.fullmatch(r"-?\d+", v):
            v = int(v)
        elif re.fullmatch(r"-?\d+\.\d+", v):
            try:
                v = float(v)
            except Exception:
                pass
        # list syntax [a,b,c]
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if inner:
                parts = [p.strip().strip('"\'') for p in inner.split(",")]
                # try convert numbers
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
    """Call a registered tool from tools.py and return a string representation of the result."""
    try:
        result = tools.run_callable(name, arguments)
        # tools.run_callable may return complex objects; stringify safely
        if isinstance(result, dict):
            # If there is an error key, surface it
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


def bot_response(user_message: str, state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process a user message and update chat history using 'messages' format for Gradio Chatbot.

    state is expected to be a dict with keys: user_id (optional) and messages (list of {"role":..., "content":...}).
    Returns (messages, state) as outputs.
    """
    user_id = state.get("user_id")
    messages: List[Dict[str, Any]] = state.get("messages") or []

    # append user message
    messages.append({"role": "user", "content": user_message})

    # persist raw history
    save_history(user_id, messages)

    # Check for explicit tool invocation
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

    # Use Ollama to produce a nicer reply if available; otherwise fallback to canned message
    if OLLAMA_AVAILABLE:
        try:
            # convert our messages to Ollama style if needed
            convo = [{"role": m["role"], "content": m["content"]} for m in messages[-12:]]
            system_msg = {"role": "system", "content": "Sei un assistente che può usare tool locali come /add, /weather, ecc."}
            ollama_msgs = [system_msg] + convo
            assistant_reply = ollama_chat_sync(ollama_msgs)
        except Exception as e:
            assistant_reply = f"(Errore Ollama: {e})\nSe vuoi usare un tool locale invia /toolname ...\nTool disponibili: {', '.join(sorted(list(tools.registered_functions.keys())))}"
    else:
        assistant_reply = (
            "Ho ricevuto il tuo messaggio. Se vuoi usare uno dei tool locali, invia un comando nel formato:\n"
            "/toolname arg1=val1 arg2=val2  oppure /toolname {\"arg1\": val}\n"
            "Tool disponibili: " + ", ".join(sorted(list(tools.registered_functions.keys())))
        )

    messages.append({"role": "assistant", "content": assistant_reply})
    state["messages"] = messages
    save_history(user_id, messages)
    return messages, state


def reset_conversation(user_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reset the conversation both in-memory and on-disk for the given user_id. Returns empty messages and state."""
    # remove persisted file
    path = history_filepath(user_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    state = {"user_id": user_id, "messages": []}
    return [], state


def build_demo():
    with gr.Blocks(title="Agentic RAG - demo tool locali") as demo:
        gr.Markdown("""
        ## Agentic RAG (demo)
        Usa la casella sotto per inviare messaggi all'agente.

        - Per chiamare un tool: `/toolname key=val` oppure `/toolname {"key": "val"}`
        - Seleziona un tool dal menu a tendina e passi argomenti in JSON per eseguirlo manualmente.
        - Premi `Reset` per azzerare la conversazione.
        """)

        # Use 'messages' format to avoid deprecation warnings; each message is a dict {role, content}
        chatbot = gr.Chatbot(label="Conversazione", type="messages")
        state = gr.State({"user_id": None, "messages": []})

        # load persisted history if user provides an id
        user_id_inp = gr.Textbox(label="User ID (opzionale)", placeholder="Inserisci ID per persistere la conversazione")

        with gr.Row():
            txt = gr.Textbox(label="Messaggio", placeholder="Scrivi qui...\nUsa /add a=1 b=2 per chiamare un tool")
            send = gr.Button("Invia")

        with gr.Row():
            tool_dropdown = gr.Dropdown(choices=sorted(list(tools.registered_functions.keys())), label="Tool (opzionale)")
            tool_args = gr.Textbox(label="Argomenti JSON (opzionale)", placeholder='{"a":1, "b":2}')
            run_tool_btn = gr.Button("Esegui tool")
            reset_btn = gr.Button("Reset")

        # Wiring: set user id -> load history
        def set_user_and_load(uid: str, current_state: Dict[str, Any]):
            if uid and uid.strip():
                messages = load_history(uid)
                # convert persisted messages (list of dicts) into state
                return {"user_id": uid, "messages": messages}, messages
            return current_state, current_state.get("messages") or []

        user_id_inp.change(fn=set_user_and_load, inputs=[user_id_inp, state], outputs=[state, chatbot])

        # Wiring: send message
        send.click(fn=bot_response, inputs=[txt, state], outputs=[chatbot, state])

        # Execute selected tool with JSON args
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
            # returns empty messages and state
            _, state = reset_conversation(uid if uid and uid.strip() else None)
            return [], state

        reset_btn.click(fn=reset_for_user, inputs=[user_id_inp], outputs=[chatbot, state])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0")
