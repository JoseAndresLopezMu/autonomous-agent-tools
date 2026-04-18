import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import run_agent

load_dotenv()

st.set_page_config(page_title="Agente Autónomo", page_icon="🤖", layout="wide")

st.title("🤖 Agente Autónomo con Herramientas")
st.caption("Powered by **Groq (LLaMA 3.1)** · **LangChain** · **Tavily Search**")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🛠️ Herramientas activas")
    st.markdown(
        """
| Herramienta | Descripción |
|---|---|
| 🌐 `web_search` | Busca en internet en tiempo real |
| 🐍 `python_repl` | Ejecuta código Python |
| 📄 `read_file` | Lee el archivo que subas |
"""
    )

    st.divider()
    st.header("📁 Subir archivo")
    uploaded_file = st.file_uploader("CSV, PDF o TXT", type=["csv", "pdf", "txt"])

    uploaded_content: str | None = None
    if uploaded_file:
        if uploaded_file.type == "text/csv" or uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            uploaded_content = (
                f"Archivo CSV — {len(df)} filas, {len(df.columns)} columnas.\n"
                f"Columnas: {list(df.columns)}\n\n"
                f"Primeras filas:\n{df.head(10).to_string()}\n\n"
                f"Estadísticas:\n{df.describe().to_string()}"
            )
            st.success(f"✅ CSV cargado — {len(df)} filas")

        elif uploaded_file.type == "text/plain" or uploaded_file.name.endswith(".txt"):
            uploaded_content = uploaded_file.read().decode("utf-8", errors="replace")
            st.success("✅ Texto cargado")

        else:
            try:
                import pymupdf4llm

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                uploaded_content = pymupdf4llm.to_markdown(tmp_path)
                os.unlink(tmp_path)
                st.success("✅ PDF cargado")
            except Exception as e:
                st.error(f"Error leyendo PDF: {e}")

    st.divider()
    st.header("💡 Ejemplos")
    examples = [
        "¿Cuál es el precio actual del Bitcoin?",
        "Busca las últimas noticias sobre inteligencia artificial",
        "Calcula los primeros 15 números de Fibonacci con Python",
        "¿Qué empresas del IBEX 35 subieron más en 2025?",
        "Analiza el archivo subido y dime qué columna tiene mayor variación",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"btn_{ex[:20]}"):
            st.session_state["pending_query"] = ex
            st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander("🧠 Razonamiento del agente", expanded=False):
                for step in msg["steps"]:
                    if step["type"] == "action":
                        st.markdown(f"**🔧 Herramienta:** `{step['tool']}`")
                        if step["log"]:
                            st.markdown(f"**💭 Pensamiento:** {step['log']}")
                        st.markdown(f"**📥 Input:** `{step['input']}`")
                    else:
                        st.markdown(f"**📊 Observación:**\n```\n{step['output']}\n```")
                    st.divider()
        st.write(msg["content"])

# ── Input (chat_input o botón ejemplo) ───────────────────────────────────────
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Escribe tu objetivo o pregunta…")
query = pending or user_input

if query:
    # Validate keys
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ Falta `GROQ_API_KEY` en el archivo `.env`")
        st.stop()
    if not os.getenv("TAVILY_API_KEY"):
        st.warning("⚠️ Sin `TAVILY_API_KEY` la búsqueda web no funcionará. Obtén una gratis en tavily.com")

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Run agent
    with st.chat_message("assistant"):
        steps_slot = st.empty()
        answer_slot = st.empty()

        with st.spinner("🤖 El agente está razonando…"):
            try:
                answer, steps = run_agent(query, uploaded_content)
            except Exception as e:
                st.error(f"Error del agente: {e}")
                st.stop()

        # Show reasoning steps
        if steps:
            with steps_slot.expander("🧠 Razonamiento del agente", expanded=True):
                for step in steps:
                    if step["type"] == "action":
                        st.markdown(f"**🔧 Herramienta:** `{step['tool']}`")
                        if step["log"]:
                            st.markdown(f"**💭 Pensamiento:** {step['log']}")
                        st.markdown(f"**📥 Input:** `{step['input']}`")
                    else:
                        st.markdown(f"**📊 Observación:**\n```\n{step['output']}\n```")
                    st.divider()

        answer_slot.write(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "steps": steps}
        )
