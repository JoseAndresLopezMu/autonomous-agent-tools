"""Runner de tests minimalista — ejecuta los tests de tools sin pytest/langchain.
Se usa SOLO para verificar la lógica de las tools en este sandbox offline.
En producción usa: pytest tests/ -v
"""
import sys
import os
import traceback
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Cargar stubs de langchain_core y tavily ANTES de importar tools
from _stubs import _install_langchain_stub
_install_langchain_stub()

# Directorio de uploads para tests
TEST_UPLOAD_DIR = Path("/tmp/agent_test_uploads")
TEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.environ["AGENT_UPLOAD_DIR"] = str(TEST_UPLOAD_DIR)


# Fake pytest para compatibilidad mínima
class _FakePytest:
    class raises:
        def __init__(self, exc_type, match=None):
            self.exc_type = exc_type
            self.match = match
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exc_type.__name__} but nothing raised")
            if not issubclass(exc_type, self.exc_type):
                return False
            if self.match and self.match not in str(exc_val):
                raise AssertionError(f"Message '{exc_val}' does not match '{self.match}'")
            return True

pytest_module = type(sys)("pytest")
pytest_module.raises = _FakePytest.raises
sys.modules["pytest"] = pytest_module


# unittest.mock está en stdlib, perfecto
from unittest.mock import patch, MagicMock  # noqa


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_class(self, cls):
        name = cls.__name__
        print(f"\n  📦 {name}")
        instance = cls()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in test_methods:
            if hasattr(instance, "setup_method"):
                instance.setup_method()
            try:
                getattr(instance, method_name)()
                print(f"    ✅ {method_name}")
                self.passed += 1
            except Exception as e:
                print(f"    ❌ {method_name}: {e}")
                self.errors.append((f"{name}.{method_name}", traceback.format_exc()))
                self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        if self.failed == 0:
            print(f"✅ {self.passed}/{total} tests pasaron")
        else:
            print(f"❌ {self.failed}/{total} tests fallaron")
            print("\nDetalles de fallos:")
            for name, tb in self.errors:
                print(f"\n--- {name} ---")
                print(tb)
        return self.failed == 0


if __name__ == "__main__":
    runner = TestRunner()

    print("🧪 Ejecutando tests de herramientas (modo offline, stubs de LangChain)\n")

    # Importar tests
    from test_tools import (
        TestPythonRepl,
        TestReadFile,
        TestGetWeather,
        TestWebSearch,
        TestToolsRegistry,
    )

    for cls in [TestPythonRepl, TestReadFile, TestGetWeather, TestWebSearch, TestToolsRegistry]:
        runner.run_class(cls)

    ok = runner.summary()
    sys.exit(0 if ok else 1)
