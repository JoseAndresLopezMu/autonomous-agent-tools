import os
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

REACT_TEMPLATE = """Eres un agente que razona paso a paso. Tienes acceso a las siguientes herramientas:

{tools}

Formato OBLIGATORIO (responde siempre en español):

Question: la pregunta que debes responder
Thought: piensa qué herramienta necesitas usar
Action: la herramienta a usar, debe ser una de [{tool_names}]
Action Input: el input para la herramienta
Observation: el resultado de la herramienta
... (repite Thought/Action/Action Input/Observation hasta tener la respuesta)
Thought: Ya tengo la respuesta final
Final Answer: la respuesta final al usuario

¡Empieza!

Question: {input}
Thought:{agent_scratchpad}"""


class StepCollector(BaseCallbackHandler):
    def __init__(self):
        self.steps = []

    def on_agent_action(self, action, **kwargs):
        self.steps.append({
            "type": "action",
            "tool": action.tool,
            "input": str(action.tool_input),
            "log": action.log.strip(),
        })

    def on_tool_end(self, output, **kwargs):
        preview = str(output)
        if len(preview) > 800:
            preview = preview[:800] + "..."
        self.steps.append({"type": "observation", "output": preview})


def run_agent(query: str, uploaded_content: str = None) -> tuple[str, list]:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    tools = [
        TavilySearchResults(
            max_results=4,
            name="web_search",
            description="Busca información actualizada en internet. Úsala para noticias, precios, datos en tiempo real.",
        ),
        PythonREPLTool(
            description="Ejecuta código Python. Úsala para cálculos, análisis de datos, estadísticas o generar texto estructurado."
        ),
    ]

    if uploaded_content:
        from langchain.tools import tool as make_tool

        content_snapshot = uploaded_content

        @make_tool
        def read_uploaded_file(query: str) -> str:  # noqa: ARG001
            """Lee el archivo subido por el usuario (CSV, PDF o texto). Úsala cuando el usuario pida analizar su archivo."""
            return content_snapshot

        tools.append(read_uploaded_file)

    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)

    collector = StepCollector()
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=10,
        callbacks=[collector],
        handle_parsing_errors=True,
    )

    result = executor.invoke({"input": query})
    return result["output"], collector.steps
