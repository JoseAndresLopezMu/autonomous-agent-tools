"""Agente autónomo con LangChain + Groq + tool calling."""
import os
from typing import Iterator, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from tools import ALL_TOOLS


SYSTEM_PROMPT = """Eres un agente autónomo que razona paso a paso para resolver tareas.

Tienes acceso a estas herramientas:
- web_search: busca información actualizada en internet
- python_repl: ejecuta código Python para cálculos y análisis
- read_file: lee archivos subidos (PDF, CSV, TXT)
- get_weather: obtiene el tiempo actual de una ciudad

REGLAS:
1. Usa SIEMPRE get_weather si preguntan por el tiempo o clima de cualquier ciudad — nunca respondas sin llamar a la herramienta.
2. Usa SIEMPRE web_search para precios actuales, noticias de hoy o cualquier dato que cambie con el tiempo.
3. Usa python_repl para cálculos numéricos, nunca calcules mentalmente.
4. Usa read_file solo cuando el usuario haya subido un archivo.
5. Para preguntas de conocimiento general que NO requieren datos actuales (historia, conceptos, fórmulas), responde directamente sin herramientas.
6. Responde siempre en el mismo idioma que el usuario.
7. Si piden tabla o comparativa, usa formato tabla markdown (| col | col |).

FORMATO DE CÓDIGO PYTHON (crítico):
- Usa SIEMPRE saltos de línea reales entre sentencias, NUNCA en una sola línea.
- Indentación de 4 espacios para bloques (for, if, def, while).
- NO uses punto y coma para separar sentencias."""


def build_agent(
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    verbose: bool = True,
    max_iterations: int = 10,
) -> AgentExecutor:
    """Construye el agente con las herramientas registradas."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY no configurada. Añádela en .env "
            "(consigue una gratis en https://console.groq.com)"
        )

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=groq_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return executor


def run_agent_stream(executor: AgentExecutor, user_input: str, chat_history=None, _retry: bool = True) -> Iterator[dict]:
    chat_history = chat_history or []
    try:
        for chunk in executor.stream({"input": user_input, "chat_history": chat_history}):
            if "actions" in chunk:
                for action in chunk["actions"]:
                    yield {"type": "tool_start", "tool": action.tool, "input": action.tool_input}
            if "steps" in chunk:
                for step in chunk["steps"]:
                    yield {"type": "tool_end", "tool": step.action.tool, "output": str(step.observation)[:2000]}
            if "output" in chunk:
                yield {"type": "final", "output": chunk["output"]}
    except Exception as e:
        err = str(e).lower()
        groq_tool_error = (
            "tool call validation failed" in err
            or "failed to call a function" in err
            or "failed_generation" in err
        )
        if _retry and groq_tool_error:
            # First retry: same query without retry flag (transient Groq failures)
            retry_events = list(run_agent_stream(executor, user_input, chat_history=chat_history, _retry=False))
            retry_failed = all(ev["type"] == "error" for ev in retry_events)
            if not retry_failed:
                yield from retry_events
            else:
                # Final fallback: answer directly without tools
                direct_prompt = (
                    f"{user_input}\n\n"
                    "(Responde directamente desde tu conocimiento sin usar herramientas externas.)"
                )
                try:
                    result = executor.invoke({"input": direct_prompt, "chat_history": chat_history})
                    yield {"type": "final", "output": result.get("output", str(result))}
                except Exception as e2:
                    yield {"type": "error", "output": f"Error: {e2}"}
        else:
            yield {"type": "error", "output": f"Error en el agente: {e}"}


def run_agent_sync(executor: AgentExecutor, user_input: str, chat_history=None) -> dict:
    """Ejecuta el agente sin streaming (útil para tests)."""
    chat_history = chat_history or []
    result = executor.invoke({"input": user_input, "chat_history": chat_history})
    return result
