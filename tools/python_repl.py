"""Python REPL tool para ejecutar cálculos y análisis de datos."""
import io
import contextlib
import traceback
from langchain_core.tools import tool


# Estado persistente entre llamadas (mismo namespace en la sesión)
_repl_globals: dict = {}


@tool
def python_repl(code: str) -> str:
    """Ejecuta código Python. Usa print() para ver resultados. Disponible: pandas, numpy, math, json."""
    # Preparar librerías comunes en el namespace si es la primera vez
    if not _repl_globals:
        try:
            import math, json, datetime
            import pandas as pd
            _repl_globals.update({
                "math": math,
                "json": json,
                "datetime": datetime,
                "pd": pd,
                "__builtins__": __builtins__,
            })
            try:
                import numpy as np
                _repl_globals["np"] = np
            except ImportError:
                pass
        except Exception as e:
            return f"Error inicializando REPL: {e}"

    import re as _re
    # Strip markdown code fences the LLM sometimes adds
    code = _re.sub(r"^```(?:python)?\n?", "", code.strip())
    code = _re.sub(r"\n?```$", "", code.strip())

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), \
             contextlib.redirect_stderr(stderr_buffer):
            exec(code, _repl_globals)

        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()

        if stderr:
            return f"STDERR:\n{stderr}\n\nSTDOUT:\n{stdout}" if stdout else f"STDERR:\n{stderr}"

        return stdout if stdout else "(ejecutado sin output — usa print() para ver resultados)"
    except Exception:
        return f"Error ejecutando código:\n{traceback.format_exc()}"


def reset_repl():
    """Limpia el estado del REPL (útil para tests)."""
    _repl_globals.clear()
