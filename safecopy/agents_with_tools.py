
import os
import json
import ollama
from dotenv import load_dotenv
from typing import Any, List, Optional
from memory import ChatMessage, BaseMemory
from tools import run_callable, today_is_tool, weather_tool, day_of_week_tool, add_tool, multiply_tool, square_tool


from dotenv import load_dotenv
load_dotenv(interpolate=False)


class BaseChatModel:
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        keep_alive: int = -1,
        format: Optional[str] = None,
        max_stored_memory_messages: int = 50,
    ) -> None:

        self.client = client
        self.model = model
        self.format = format
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.memory = BaseMemory(max_size=max_stored_memory_messages)
        self.system = """Sei un assistente AI utile. Quando utilizzi gli strumenti, fornisci sempre una risposta naturale e completa utilizzando le informazioni raccolte. 
        Formula la tua risposta come una frase breve e coerente"""

    def message(self, human: str, ai: str) -> ChatMessage:
        return ChatMessage(message={"human": human, "ai": ai})


class OllamaChatModel(BaseChatModel):
    """http://10.100.14.73:11434 is the default Ollama port serving API."""

    def __init__(self, tools: list[dict], model: str = "llama3.2") -> None:
        self.model = model
        self.tools = tools
        # allow remote Ollama host via environment variable
        ollama_host = os.environ.get("OLLAMA_HOST", "http://10.100.14.73:11434")
        self.client = ollama.Client(host=ollama_host)
        super().__init__(client=self.client, model=self.model)

    def extract(self, tool_call) -> list:
        """Extract and execute tool call"""
        data = []

        if not isinstance(tool_call, list):
            tool_call = [tool_call]

        for tool in tool_call:
            func_name = tool.function.name
            if isinstance(tool.function.arguments, str):
                func_arguments = json.loads(tool.function.arguments)
            else:
                func_arguments = tool.function.arguments
            result = run_callable(func_name, func_arguments)
            data.append(result)
        return data

    def response(self, user_prompt: str, system_message: str = None) -> ollama.ChatResponse:
        messages = [
            {
                "role": "system",
                "content": system_message if system_message else self.system,
            }
        ]

        for msg in self.memory.get():
            if isinstance(msg, ChatMessage):
                messages.extend(
                    [
                        {"role": "user", "content": msg.human},
                        {"role": "assistant", "content": msg.ai},
                    ]
                )

        messages.append({"role": "user", "content": user_prompt})

        return self.client.chat(
            model=self.model,
            messages=messages,
            format=self.format,
            keep_alive=self.keep_alive,
            tools=self.tools,
        )

    def chat(self, system_message: str = None, save_chat: bool = False) -> None:
        system_message = system_message if system_message else self.system

        while True:
            user_prompt = input("Domanda: ")
            if user_prompt == "Ciao" or user_prompt == "Esci":
                self.memory.add(self.message(human=user_prompt, ai="Ciao"))
                if save_chat:
                    self.memory.save(model_name=str(self.model))
                self.memory.clear()
                print("AI: Ciao")
                break

            response = self.response(user_prompt, system_message)

            # If there are tool calls, process them and get final response
            if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                collected_data = {}

                for tool_call in response.message.tool_calls:
                    result = self.extract(tool_call)
                    collected_data[tool_call.function.name] = result

                final_prompt = (
                    f"Basandoti sulle informazioni seguenti:\n"
                    f"{collected_data}"
                    f"fornisci una risposta naturale alla domanda originale: '{user_prompt}'"
                )

                final_response = self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Riepiloga la tua risposta prima di inviarla. Rendila breve e concisa."
                            + " "
                            + system_message,
                        },
                        {"role": "user", "content": final_prompt},
                    ],
                )
                response_content = final_response.message.content
            else:
                # If no tool calls, use the original response
                response_content = response.message.content

            if response_content:
                print(f"AI: {response_content}", end="\n\n")
                self.memory.add(self.message(human=user_prompt, ai=response_content))



def load_model(model: str = None) -> OllamaChatModel:
    # caricamento modello con tools integrati
    tools = [today_is_tool, weather_tool, day_of_week_tool, add_tool, multiply_tool, square_tool]
    # allow overriding default model via OLLAMA_MODEL env var
    env_model = os.environ.get("OLLAMA_MODEL")
    if env_model:
        return OllamaChatModel(tools=tools, model=env_model)
    return OllamaChatModel(tools=tools, model=model)
    

def main():
    model = load_model("ollama")
    model.chat(save_chat=True)

if __name__ == "__main__":
    main()
