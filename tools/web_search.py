"""Web search tool usando Tavily API."""
import os
from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def web_search(query: str) -> str:
    """Busca información en internet en tiempo real. Úsalo para noticias, precios o datos actuales."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY no está configurada en .env"

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

        output = []
        if response.get("answer"):
            output.append(f"Resumen: {response['answer']}\n")

        output.append("Resultados:")
        for i, result in enumerate(response.get("results", []), 1):
            output.append(
                f"\n{i}. {result.get('title', 'Sin título')}"
                f"\n   URL: {result.get('url', '')}"
                f"\n   {result.get('content', '')[:300]}..."
            )

        return "\n".join(output) if output else "No se encontraron resultados."
    except Exception as e:
        return f"Error en búsqueda web: {str(e)}"
