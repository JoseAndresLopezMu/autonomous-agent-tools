import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

UPLOAD_DIR = Path("/tmp/agent_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.environ["AGENT_UPLOAD_DIR"] = str(UPLOAD_DIR)

from agent import build_agent, run_agent_stream  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agente Autónomo IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #0e1117; color: #e6eaf0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b2e 0%, #0e1117 100%);
        border-right: 1px solid #1f2b45;
    }

    /* Brand header */
    .brand-title {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .brand-sub {
        font-size: 0.75rem;
        color: #6b7280;
        margin-bottom: 0;
    }

    /* Tool cards */
    .tool-card {
        background: #161b2e;
        border: 1px solid #1f2b45;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .tool-card b { color: #a78bfa; }

    /* Status badges */
    .badge-ok {
        background: #064e3b; color: #6ee7b7;
        padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
    }
    .badge-err {
        background: #7f1d1d; color: #fca5a5;
        padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background: #161b2e;
        border: 1px solid #1f2b45;
        border-radius: 14px;
        margin-bottom: 10px;
        padding: 4px;
    }

    /* Expanders — forzar dark en todos los niveles */
    div[data-testid="stExpander"],
    div[data-testid="stExpander"] *,
    div[data-testid="stExpander"] details,
    div[data-testid="stExpander"] details > div,
    div[data-testid="stExpander"] details summary,
    .streamlit-expanderContent,
    .streamlit-expanderContent > div,
    [data-baseweb="block"] {
        background-color: #111827 !important;
        color: #e6eaf0 !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #1f2b45 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        background-color: #1a2035 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.85; }
    .stButton > button[kind="secondary"] {
        background: #161b2e;
        border: 1px solid #2d3748;
        color: #a0aec0;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #818cf8;
        color: #818cf8;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background: #161b2e !important;
        border-color: #1f2b45 !important;
    }

    /* Divider */
    hr { border-color: #1f2b45; }

    /* Example buttons row */
    .examples-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    /* Code blocks — wrap text, no horizontal scroll */
    pre, code {
        background: #0a0e1a !important;
        border-radius: 8px !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
        overflow-x: hidden !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

GROQ_MODELS = {
    "LLaMA 3.3 70B (recomendado)": "llama-3.3-70b-versatile",
    "LLaMA 3.1 8B  (rápido)": "llama-3.1-8b-instant",
}

EXAMPLES = [
    "Precio del oro hoy en dólares",
    "¿Qué tiempo hace en Madrid ahora mismo?",
    "Calcula los primeros 20 números de Fibonacci",
    "Busca las últimas noticias sobre inteligencia artificial",
    "¿Cuánto es 15% de propina sobre 47.80 €?",
    "Haz una tabla markdown comparando GPT-4o, Claude 3.5 y Gemini 2.0",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="brand-title">🤖 Agente Autónomo IA</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-sub">Groq · LangChain · Tavily · Open-Meteo</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Model selector
    st.markdown("**Modelo**")
    model_label = st.selectbox(
        "modelo",
        list(GROQ_MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected_model = GROQ_MODELS[model_label]

    st.divider()

    # API status
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    st.markdown("**Estado de APIs**")
    st.markdown(
        f'Groq &nbsp; <span class="{"badge-ok" if groq_ok else "badge-err"}">{"activa" if groq_ok else "falta key"}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'Tavily &nbsp; <span class="{"badge-ok" if tavily_ok else "badge-err"}">{"activa" if tavily_ok else "falta key"}</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    # File uploader
    st.markdown("**Subir archivos**")
    uploaded = st.file_uploader(
        "PDF, CSV, TXT o MD",
        type=["pdf", "csv", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for f in uploaded:
            (UPLOAD_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"{len(uploaded)} archivo(s) listos para el agente")

    existing = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    if existing:
        st.markdown("**Archivos disponibles:**")
        for name in existing:
            st.code(name, language=None)

    st.divider()

    # Tools
    st.markdown("**Herramientas activas**")
    tools_info = [
        ("🌐", "web_search", "Búsqueda en tiempo real"),
        ("🐍", "python_repl", "Ejecuta código Python"),
        ("📄", "read_file", "Lee PDF / CSV / TXT"),
        ("🌤️", "get_weather", "Tiempo en cualquier ciudad"),
    ]
    for icon, name, desc in tools_info:
        st.markdown(
            f'<div class="tool-card">{icon} <b>{name}</b><br><span style="color:#9ca3af">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        if "agent" in st.session_state:
            del st.session_state["agent"]
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## Agente Autónomo con Herramientas")
st.caption(
    "Describe tu objetivo y el agente razonará paso a paso, usando las herramientas necesarias para resolverlo."
)

# Example buttons
st.markdown('<p class="examples-label">Prueba un ejemplo</p>', unsafe_allow_html=True)
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, use_container_width=True, key=f"ex_{i}"):
        st.session_state["pending_query"] = ex
        st.rerun()

st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Rebuild agent if model changed
if st.session_state.get("current_model") != selected_model:
    st.session_state.pop("agent", None)
    st.session_state["current_model"] = selected_model

if "agent" not in st.session_state:
    if not groq_ok:
        st.warning("Configura `GROQ_API_KEY` en `.env` para empezar.", icon="⚠️")
        st.stop()
    try:
        st.session_state.agent = build_agent(model=selected_model, verbose=False)
    except Exception as e:
        st.error(f"Error inicializando el agente: {e}")
        st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("steps"):
            with st.expander(f"🔍 Razonamiento — {len(msg['steps'])} paso(s)", expanded=False):
                for step in msg["steps"]:
                    st.markdown(f"**🔧 `{step['tool']}`**")
                    if step.get("input"):
                        raw = step["input"]
                        display = next(iter(raw.values())) if isinstance(raw, dict) and len(raw) == 1 else str(raw)
                        st.code(display, language="python" if step["tool"] == "python_repl" else "text")
                    st.markdown("*Resultado:*")
                    st.code(step["output"][:1500], language="text")
                    st.divider()
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Escribe tu pregunta u objetivo…")
query = pending or user_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Last 6 messages as history
    chat_history = []
    for m in st.session_state.messages[:-1][-6:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        chat_history.append(cls(content=m["content"]))

    with st.chat_message("assistant"):
        reasoning_box = st.expander("🔍 Razonamiento en vivo", expanded=True)
        final_placeholder = st.empty()

        steps: list[dict] = []
        final_output = ""
        live_spinner = None

        try:
            for event in run_agent_stream(st.session_state.agent, query, chat_history=chat_history):
                if event["type"] == "tool_start":
                    with reasoning_box:
                        st.markdown(f"**🔧 Usando `{event['tool']}`**")
                        # Show the input value cleanly
                        raw_input = event["input"]
                        if isinstance(raw_input, dict):
                            display = next(iter(raw_input.values())) if len(raw_input) == 1 else str(raw_input)
                        else:
                            display = str(raw_input)
                        st.code(display, language="python" if event["tool"] == "python_repl" else "text")
                        live_spinner = st.empty()
                        live_spinner.info("⏳ ejecutando…")

                elif event["type"] == "tool_end":
                    if live_spinner:
                        live_spinner.empty()
                        live_spinner = None
                    with reasoning_box:
                        st.markdown("*Resultado:*")
                        st.code(event["output"][:1500], language="text")
                        st.divider()
                    steps.append(
                        {"tool": event["tool"], "input": "", "output": event["output"]}
                    )

                elif event["type"] == "final":
                    final_output = event["output"]
                    final_placeholder.markdown(final_output)

                elif event["type"] == "error":
                    final_output = f"❌ {event['output']}"
                    final_placeholder.error(final_output)

        except Exception as e:
            final_output = f"❌ Error inesperado: {e}"
            final_placeholder.error(final_output)

        st.session_state.messages.append(
            {"role": "assistant", "content": final_output, "steps": steps}
        )
