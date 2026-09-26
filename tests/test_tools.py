import pytest
from unittest.mock import MagicMock, patch

from e2b_code_interpreter import Execution, ExecutionError, Logs, SandboxException

from app.agent.tools import documentation_search, python_repl, web_search


@patch("app.agent.tools.build_rag_chain")
def test_documentation_search_returns_answer_and_sources(mock_build):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Ridge is L2 regularization."
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        MagicMock(metadata={"source": "https://scikit-learn.org/ridge.html"})
    ]
    mock_build.return_value = (mock_chain, mock_retriever)

    import app.agent.tools as tools_module
    tools_module._chain = None
    tools_module._retriever = None

    result = documentation_search.invoke({"query": "What is Ridge?"})

    assert "Ridge" in result
    assert "Sources:" in result
    assert "scikit-learn.org" in result


def _mock_sandbox(mock_sandbox_cls, execution):
    sandbox = mock_sandbox_cls.create.return_value.__enter__.return_value
    sandbox.run_code.return_value = execution
    return sandbox


@pytest.fixture
def e2b_key(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.e2b_api_key", "test-key")


@patch("app.agent.tools.Sandbox")
def test_python_repl_executes_arithmetic(mock_sandbox_cls, e2b_key):
    sandbox = _mock_sandbox(
        mock_sandbox_cls, Execution(logs=Logs(stdout=["42\n"]))
    )

    result = python_repl.invoke({"code": "print(7 * 6)"})

    assert "42" in result
    sandbox.run_code.assert_called_once()
    assert mock_sandbox_cls.create.call_args.kwargs["allow_internet_access"] is False


@patch("app.agent.tools.Sandbox")
def test_python_repl_handles_syntax_error_gracefully(mock_sandbox_cls, e2b_key):
    _mock_sandbox(
        mock_sandbox_cls,
        Execution(error=ExecutionError("SyntaxError", "incomplete input", "")),
    )

    result = python_repl.invoke({"code": "print(1 +"})

    assert "SyntaxError" in result


@patch("app.agent.tools.Sandbox")
def test_python_repl_returns_sandbox_failure_as_text(mock_sandbox_cls, e2b_key):
    mock_sandbox_cls.create.side_effect = SandboxException("quota exceeded")

    result = python_repl.invoke({"code": "print(1)"})

    assert "Sandbox error" in result


@patch("app.agent.tools.Sandbox")
def test_python_repl_truncates_long_output(mock_sandbox_cls, e2b_key, monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.agent_max_output_chars", 10)
    _mock_sandbox(mock_sandbox_cls, Execution(logs=Logs(stdout=["x" * 100])))

    result = python_repl.invoke({"code": "print('x' * 100)"})

    assert result.startswith("x" * 10)
    assert "truncated" in result


@patch("app.agent.tools.Sandbox")
def test_python_repl_disabled_without_api_key(mock_sandbox_cls, monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.e2b_api_key", None)

    result = python_repl.invoke({"code": "print(1)"})

    assert "disabled" in result.lower()
    mock_sandbox_cls.create.assert_not_called()


def test_web_search_respects_disable_flag(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_web_search", False)
    result = web_search.invoke({"query": "anything"})
    assert "disabled" in result.lower()