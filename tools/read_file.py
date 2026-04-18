"""Tool para leer archivos subidos por el usuario (PDF, CSV, TXT)."""
import os
from pathlib import Path
from langchain_core.tools import tool


UPLOAD_DIR = Path(os.getenv("AGENT_UPLOAD_DIR", "/tmp/agent_uploads"))


@tool
def read_file(filename: str) -> str:
    """Lee un archivo subido por el usuario (PDF, CSV o TXT). Úsalo cuando el usuario pida analizar su archivo."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filepath = UPLOAD_DIR / filename

    if not filepath.exists():
        available = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
        if not available:
            return f"Error: No hay archivos subidos. El usuario necesita subir '{filename}' primero."
        return f"Error: '{filename}' no encontrado. Archivos disponibles: {', '.join(available)}"

    ext = filepath.suffix.lower()

    try:
        if ext == ".pdf":
            return _read_pdf(filepath)
        elif ext == ".csv":
            return _read_csv(filepath)
        elif ext in (".txt", ".md"):
            return filepath.read_text(encoding="utf-8")[:10000]
        else:
            return f"Error: Extensión '{ext}' no soportada. Usa PDF, CSV o TXT."
    except Exception as e:
        return f"Error leyendo '{filename}': {str(e)}"


def _read_pdf(filepath: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(filepath))
    pages_text = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages_text.append(f"--- Página {i} ---\n{text}")
    full = "\n\n".join(pages_text)
    # Truncar a 15k chars para no reventar el contexto
    if len(full) > 15000:
        full = full[:15000] + "\n\n[...texto truncado, PDF demasiado largo...]"
    return full


def _read_csv(filepath: Path) -> str:
    import pandas as pd
    df = pd.read_csv(filepath)
    summary = []
    summary.append(f"CSV: {filepath.name}")
    summary.append(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
    summary.append(f"Columnas: {list(df.columns)}")
    summary.append(f"\nTipos de datos:\n{df.dtypes.to_string()}")
    summary.append(f"\nPrimeras 10 filas:\n{df.head(10).to_string()}")
    try:
        summary.append(f"\nEstadísticas numéricas:\n{df.describe().to_string()}")
    except Exception:
        pass
    return "\n".join(summary)
