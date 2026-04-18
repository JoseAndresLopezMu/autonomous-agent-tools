"""Tests para la construcción del agente y el flujo de streaming."""
import os
import pytest
from unittest.mock import patch, MagicMock


class TestAgentBuild:
    def test_build_fails_without_groq_key(self):
        from agent import build_agent
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                build_agent()

    def test_build_succeeds_with_key(self):
        """Verifica que el agente se puede construir con una key (aunque sea falsa)."""
        os.environ["GROQ_API_KEY"] = "gsk_fake_for_test"
        from agent import build_agent
        # No debe lanzar excepción; la construcción no contacta Groq
        agent = build_agent(verbose=False)
        assert agent is not None
        assert len(agent.tools) == 4

    def test_agent_has_all_tools_registered(self):
        os.environ["GROQ_API_KEY"] = "gsk_fake_for_test"
        from agent import build_agent
        agent = build_agent(verbose=False)
        tool_names = {t.name for t in agent.tools}
        assert "web_search" in tool_names
        assert "python_repl" in tool_names
        assert "read_file" in tool_names
        assert "get_weather" in tool_names

    def test_agent_max_iterations_configurable(self):
        os.environ["GROQ_API_KEY"] = "gsk_fake_for_test"
        from agent import build_agent
        agent = build_agent(max_iterations=5, verbose=False)
        assert agent.max_iterations == 5


class TestAgentStreaming:
    def test_stream_handles_tool_events(self):
        """Simula eventos del executor y verifica que run_agent_stream los mapea bien."""
        from agent import run_agent_stream

        # Mock action y step
        mock_action = MagicMock()
        mock_action.tool = "python_repl"
        mock_action.tool_input = {"code": "print(1+1)"}

        mock_step = MagicMock()
        mock_step.action = mock_action
        mock_step.observation = "2"

        mock_executor = MagicMock()
        mock_executor.stream.return_value = iter([
            {"actions": [mock_action]},
            {"steps": [mock_step]},
            {"output": "La respuesta es 2"},
        ])

        events = list(run_agent_stream(mock_executor, "suma 1+1"))

        types = [e["type"] for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        assert "final" in types

        tool_start = next(e for e in events if e["type"] == "tool_start")
        assert tool_start["tool"] == "python_repl"

        final = next(e for e in events if e["type"] == "final")
        assert final["output"] == "La respuesta es 2"

    def test_stream_handles_exceptions(self):
        from agent import run_agent_stream
        mock_executor = MagicMock()
        mock_executor.stream.side_effect = Exception("fallo interno")

        events = list(run_agent_stream(mock_executor, "test"))
        assert any(e["type"] == "error" for e in events)


class TestAgentEndToEnd:
    """Test de integración que simula una invocación completa del agente
    mockeando la respuesta del LLM."""

    def test_sync_invocation_returns_output(self):
        from agent import run_agent_sync

        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "Resultado final",
            "intermediate_steps": [],
        }

        result = run_agent_sync(mock_executor, "pregunta cualquiera")
        assert result["output"] == "Resultado final"
        mock_executor.invoke.assert_called_once()
