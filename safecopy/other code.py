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



        # def exec_selected_tool(tool_name: str, json_args: str, state: Dict[str, Any]):
        #     user_id = state.get("user_id")
        #     messages = state.get("messages") or []
        #     if not tool_name:
        #         messages.append({"role": "assistant", "content": "Nessun tool selezionato."})
        #         state["messages"] = messages
        #         save_history(user_id, messages)
        #         return messages, state
        #     try:
        #         args = json.loads(json_args) if json_args and json_args.strip() else {}
        #     except Exception as e:
        #         messages.append({"role": "assistant", "content": f"Errore nel parsing JSON degli argomenti: {e}"})
        #         state["messages"] = messages
        #         save_history(user_id, messages)
        #         return messages, state
        #     messages.append({"role": "user", "content": f"/ {tool_name} (invocazione manuale)"})
        #     messages.append({"role": "assistant", "content": f"Eseguo {tool_name} con {args}"})
        #     out = call_tool(tool_name, args)
        #     messages.append({"role": "assistant", "content": f"Risultato di {tool_name}: {out}"})
        #     state["messages"] = messages
        #     save_history(user_id, messages)
        #     return messages, state

        # run_tool_btn.click(fn=exec_selected_tool, inputs=[tool_dropdown, tool_args, state], outputs=[chatbot, state])
        
        def index_conversation(uid: str):
            global vectorstore
            
            # gestire la indicizzazione per specifico utente
            
            # carica documenti e costruisci vectorstore
            documents = TextLoader("/home/vespa/VStudioProjects/Agent_with_tools/docs/trascrizione.txt").load()
            vectorstore = FAISS.from_documents(documents, OllamaEmbeddings(model="llama3.2:latest"))
            return 
        
        # indicizza la conversazione per RAG
        index_btn.click(fn=index_conversation, inputs=[user_id_inp], outputs=[])
        
            else:
        # Fallback: use Ollama directly or a helpful message
        if OLLAMA_AVAILABLE:
            try:
                convo = [{"role": m["role"], "content": m["content"]} for m in messages[-12:]]
                system_msg = {"role": "system", "content": CUSTOM_PROMPT}
                # Append verbosity instruction to the system prompt so fallback calls
                # to Ollama also try to produce larger, more detailed replies.
                system_msg_verbose = dict(system_msg)
                system_msg_verbose["content"] = f"{system_msg_verbose.get('content','')}\n{OLLAMA_VERBOSE_INSTRUCTION}"
                ollama_msgs = [system_msg_verbose] + convo
                resp = ollama_chat_sync(ollama_msgs, model=OLLAMA_MODEL)
                assistant_reply = str(resp)
            except Exception as e:
                assistant_reply = f"(Errore Ollama: {e})\nSe vuoi usare un tool locale invia /toolname ...\nTool disponibili: {', '.join(sorted(list(tools.registered_functions.keys()))) }"
        else:
            assistant_reply = (
                "Ho ricevuto il tuo messaggio. Se vuoi usare uno dei tool locali, invia un comando nel formato:\n"
                "/toolname arg1=val1 arg2=val2  oppure /toolname {\"arg1\": val}\n"
                "Tool disponibili: " + ", ".join(sorted(list(tools.registered_functions.keys())))
            )
            
            
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