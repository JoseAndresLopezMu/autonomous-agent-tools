---
title: Agente Autónomo IA
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: true
license: mit
---

# 🤖 Agente Autónomo IA

Agente de inteligencia artificial que razona paso a paso para resolver tareas complejas, eligiendo y encadenando herramientas de forma autónoma.

## Herramientas

| Herramienta | Descripción |
|---|---|
| 🌐 `web_search` | Búsqueda en internet en tiempo real (Tavily) |
| 🐍 `python_repl` | Ejecuta código Python para cálculos y análisis |
| 📄 `read_file` | Lee archivos subidos: PDF, CSV, TXT |
| 🌤️ `get_weather` | Tiempo actual y pronóstico (Open-Meteo, sin API key) |

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| UI | Streamlit |
| LLM | Groq · LLaMA 3.1 8B / LLaMA 3.3 70B |
| Framework agente | LangChain (tool calling) |
| Búsqueda web | Tavily API |
| Tiempo | Open-Meteo (gratuito, sin key) |

## Configuración

Necesitas dos claves de API en un archivo `.env`:

```
GROQ_API_KEY=gsk_...       # gratis en https://console.groq.com
TAVILY_API_KEY=tvly_...    # gratis en https://tavily.com (1000 búsquedas/mes)
```

En Hugging Face Spaces añádelas como **Secrets** en la configuración del Space.

## Instalación local

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # rellena tus keys
streamlit run app.py
```

## Ejemplos de uso

- `"¿Cuál es el precio actual del Bitcoin?"`
- `"¿Qué tiempo hace en Bilbao esta semana?"`
- `"Calcula la suma de los primeros 100 números primos"`
- `"Busca las últimas noticias sobre IA y hazme un resumen"`
- `"Analiza este CSV y dime qué columna tiene mayor variación"` (sube el archivo primero)

## Arquitectura

```
Usuario
  │
  ▼
LLM (LLaMA 3.3 70B via Groq)
  │
  ├── ¿información actualizada? ──► web_search  ──► resultado
  ├── ¿cálculo / análisis?      ──► python_repl ──► resultado
  ├── ¿archivo subido?          ──► read_file   ──► contenido
  └── ¿pregunta por el tiempo?  ──► get_weather ──► clima
        │
        ▼
  Respuesta final al usuario
```

El bucle se repite hasta 10 iteraciones.

## Estructura del proyecto

```
├── app.py              # Interfaz Streamlit
├── agent.py            # Lógica del agente LangChain
├── tools/
│   ├── web_search.py   # Búsqueda con Tavily
│   ├── python_repl.py  # Ejecución de Python
│   ├── read_file.py    # Lectura de archivos
│   └── get_weather.py  # Consulta del tiempo
├── tests/              # Suite de tests
├── requirements.txt
└── .env.example
```

## Añadir una herramienta nueva

1. Crea `tools/mi_tool.py`:

```python
from langchain_core.tools import tool

@tool
def mi_tool(param: str) -> str:
    """Descripción clara de qué hace y cuándo usarla."""
    return "resultado"
```

2. Regístrala en `tools/__init__.py` añadiéndola a `ALL_TOOLS`.

3. El agente la descubrirá automáticamente.
