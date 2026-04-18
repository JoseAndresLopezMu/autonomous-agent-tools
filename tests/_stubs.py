"""Stub mínimo de langchain_core para poder ejecutar tests offline.
Solo se usa si langchain_core no está instalado realmente."""
import sys
import types
import functools


def _install_langchain_stub():
    """Instala stubs mínimos de langchain_core.tools y tavily si no están."""
    try:
        import langchain_core.tools  # noqa
        return  # Ya instalado, no stubeamos
    except ImportError:
        pass

    # Crear módulo fake langchain_core.tools
    lc_core = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")

    class FakeTool:
        def __init__(self, func):
            self.func = func
            self.name = func.__name__
            self.description = (func.__doc__ or "").strip()
            self.args_schema = _make_args_schema(func)
            functools.update_wrapper(self, func)

        def invoke(self, input_dict):
            if isinstance(input_dict, dict):
                return self.func(**input_dict)
            return self.func(input_dict)

        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    def _make_args_schema(func):
        """Fake args_schema — un objeto con algo de info."""
        class _Schema:
            fields = list(func.__code__.co_varnames[:func.__code__.co_argcount])
        return _Schema

    def tool_decorator(func):
        return FakeTool(func)

    lc_tools.tool = tool_decorator
    lc_core.tools = lc_tools
    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.tools"] = lc_tools

    # Stub de tavily
    try:
        import tavily  # noqa
    except ImportError:
        tavily_mod = types.ModuleType("tavily")

        class TavilyClient:
            def __init__(self, api_key): self.api_key = api_key
            def search(self, **kwargs): return {"results": [], "answer": ""}

        tavily_mod.TavilyClient = TavilyClient
        sys.modules["tavily"] = tavily_mod


_install_langchain_stub()
