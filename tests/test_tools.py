"""Tests unitarios para cada herramienta del agente."""
import os
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Configurar directorio de uploads antes de importar tools
TEST_UPLOAD_DIR = Path("/tmp/agent_test_uploads")
TEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.environ["AGENT_UPLOAD_DIR"] = str(TEST_UPLOAD_DIR)

from tools.web_search import web_search
from tools.python_repl import python_repl, reset_repl
from tools.read_file import read_file
from tools.get_weather import get_weather


# ---------- python_repl ----------

class TestPythonRepl:
    def setup_method(self):
        reset_repl()

    def test_simple_arithmetic(self):
        result = python_repl.invoke({"code": "print(2 + 2)"})
        assert "4" in result

    def test_math_library_available(self):
        result = python_repl.invoke({"code": "print(math.sqrt(16))"})
        assert "4.0" in result

    def test_pandas_available(self):
        code = "df = pd.DataFrame({'a':[1,2,3]}); print(df['a'].sum())"
        result = python_repl.invoke({"code": code})
        assert "6" in result

    def test_variables_persist_across_calls(self):
        python_repl.invoke({"code": "x = 42"})
        result = python_repl.invoke({"code": "print(x * 2)"})
        assert "84" in result

    def test_syntax_error_is_caught(self):
        result = python_repl.invoke({"code": "print(1 +)"})
        assert "Error" in result or "SyntaxError" in result

    def test_runtime_error_is_caught(self):
        result = python_repl.invoke({"code": "print(1/0)"})
        assert "Error" in result or "ZeroDivisionError" in result

    def test_no_output_returns_message(self):
        result = python_repl.invoke({"code": "y = 5"})
        assert "sin output" in result.lower() or result == ""

    def test_list_comprehension(self):
        result = python_repl.invoke({"code": "print(sum([i**2 for i in range(10)]))"})
        assert "285" in result


# ---------- read_file ----------

class TestReadFile:
    def setup_method(self):
        """Limpia y prepara el directorio de test."""
        for f in TEST_UPLOAD_DIR.iterdir():
            if f.is_file():
                f.unlink()

    def test_file_not_found_no_files(self):
        result = read_file.invoke({"filename": "noexiste.pdf"})
        assert "Error" in result
        assert "no hay archivos" in result.lower() or "no encontrado" in result.lower()

    def test_file_not_found_with_other_files(self):
        (TEST_UPLOAD_DIR / "otro.txt").write_text("contenido")
        result = read_file.invoke({"filename": "noexiste.pdf"})
        assert "Error" in result
        assert "otro.txt" in result

    def test_read_txt(self):
        (TEST_UPLOAD_DIR / "test.txt").write_text("Hola mundo\nSegunda línea")
        result = read_file.invoke({"filename": "test.txt"})
        assert "Hola mundo" in result
        assert "Segunda línea" in result

    def test_read_csv(self):
        csv_content = "nombre,edad,ciudad\nAna,30,Madrid\nLuis,25,Barcelona\nMaria,35,Bilbao"
        (TEST_UPLOAD_DIR / "datos.csv").write_text(csv_content)
        result = read_file.invoke({"filename": "datos.csv"})
        assert "Filas: 3" in result
        assert "nombre" in result
        assert "Ana" in result

    def test_unsupported_extension(self):
        (TEST_UPLOAD_DIR / "archivo.xyz").write_text("datos")
        result = read_file.invoke({"filename": "archivo.xyz"})
        assert "Error" in result
        assert ".xyz" in result


# ---------- get_weather ----------

class TestGetWeather:
    def test_geocoding_not_found(self):
        with patch("tools.get_weather.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"results": []}
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_weather.invoke({"city": "CiudadInventada123"})
            assert "no se encontró" in result.lower()

    def test_successful_weather_fetch(self):
        with patch("tools.get_weather.requests.get") as mock_get:
            # Primer call: geocoding
            geo_resp = MagicMock()
            geo_resp.json.return_value = {
                "results": [{
                    "latitude": 43.26,
                    "longitude": -2.93,
                    "name": "Bilbao",
                    "country": "España",
                }]
            }
            geo_resp.raise_for_status = MagicMock()

            # Segundo call: weather
            weather_resp = MagicMock()
            weather_resp.json.return_value = {
                "current": {
                    "temperature_2m": 18.5,
                    "relative_humidity_2m": 70,
                    "weather_code": 1,
                    "wind_speed_10m": 12.3,
                },
                "daily": {
                    "time": ["2026-04-18", "2026-04-19", "2026-04-20"],
                    "temperature_2m_max": [20, 22, 19],
                    "temperature_2m_min": [12, 13, 11],
                    "precipitation_sum": [0, 2.5, 0],
                },
            }
            weather_resp.raise_for_status = MagicMock()

            mock_get.side_effect = [geo_resp, weather_resp]

            result = get_weather.invoke({"city": "Bilbao"})
            assert "Bilbao" in result
            assert "18.5" in result
            assert "Mayormente despejado" in result
            assert "2026-04-18" in result

    def test_network_error_handled(self):
        import requests
        with patch("tools.get_weather.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection error")
            result = get_weather.invoke({"city": "Madrid"})
            assert "Error" in result


# ---------- web_search ----------

class TestWebSearch:
    def test_no_api_key(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
            # Forzar el unset
            os.environ.pop("TAVILY_API_KEY", None)
            result = web_search.invoke({"query": "test"})
            assert "TAVILY_API_KEY" in result

    def test_successful_search(self):
        os.environ["TAVILY_API_KEY"] = "fake_key_for_test"
        with patch("tools.web_search.TavilyClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.search.return_value = {
                "answer": "Python es un lenguaje de programación.",
                "results": [
                    {
                        "title": "Python.org",
                        "url": "https://python.org",
                        "content": "Sitio oficial de Python. " * 20,
                    },
                    {
                        "title": "Wikipedia",
                        "url": "https://wikipedia.org",
                        "content": "Python es un lenguaje...",
                    },
                ],
            }
            MockClient.return_value = mock_instance

            result = web_search.invoke({"query": "qué es Python"})
            assert "Python" in result
            assert "python.org" in result.lower()
            assert "Resumen" in result

    def test_api_error_handled(self):
        os.environ["TAVILY_API_KEY"] = "fake_key"
        with patch("tools.web_search.TavilyClient") as MockClient:
            MockClient.side_effect = Exception("API limit reached")
            result = web_search.invoke({"query": "test"})
            assert "Error" in result


# ---------- Integración de tools en el registry ----------

class TestToolsRegistry:
    def test_all_tools_exported(self):
        from tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 4
        names = {t.name for t in ALL_TOOLS}
        assert names == {"web_search", "python_repl", "read_file", "get_weather"}

    def test_all_tools_have_descriptions(self):
        from tools import ALL_TOOLS
        for t in ALL_TOOLS:
            assert t.description
            assert len(t.description) > 20, f"Tool {t.name} needs better description"

    def test_all_tools_have_args_schema(self):
        from tools import ALL_TOOLS
        for t in ALL_TOOLS:
            assert t.args_schema is not None, f"Tool {t.name} missing args schema"
