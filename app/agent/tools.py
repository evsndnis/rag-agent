from langchain_core.tools import tool

from e2b_code_interpreter import Sandbox, SandboxException
from langchain_community.tools import DuckDuckGoSearchRun

from app.rag.chain import build_rag_chain
from app.config import settings

_chain = None
_retriever = None
_ddg_search = DuckDuckGoSearchRun()


def _ensure_rag_ready():
    global _chain, _retriever
    if _chain is None:
        _chain, _retriever = build_rag_chain()


@tool
def documentation_search(query: str) -> str:
    """Search the scikit-learn documentation corpus for a given query.

    Use this tool when the user asks about scikit-learn classes, methods,
    parameters, or general ML concepts (Ridge, Lasso, decision trees, metrics).

    Args:
        query: natural-language question or keyword search.

    Returns:
        Answer text followed by a list of source URLs.
    """
    _ensure_rag_ready()
    docs = _retriever.invoke(query)
    answer = _chain.invoke(query)
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    sources_block = "\n".join(f"- {url}" for url in sources)
    return f"{answer}\n\nSources:\n{sources_block}"

def _truncate(text: str) -> str:
    limit = settings.agent_max_output_chars
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


# Код пишет LLM, а значит его может подсунуть пользователь или веб-страница
# (prompt injection). Поэтому исполняем его не в нашем процессе, а в E2B —
# отдельной одноразовой microVM без доступа к нашим файлам, env и сети.
@tool
def python_repl(code: str) -> str:
    """Execute Python code in an isolated sandbox and return its output.

    Use this tool for arithmetic, computing formulas, or transforming data.
    Each call starts a fresh environment with no internet access and no state
    from previous calls. Use `print()` to surface results.

    Args:
        code: Python source code to execute.

    Returns:
        Stdout of the executed code, or error message.
    """
    if not settings.e2b_api_key:
        return "Python execution is disabled: E2B_API_KEY is not set."

    try:
        with Sandbox.create(
            api_key=settings.e2b_api_key,
            timeout=settings.sandbox_timeout + 30,
            allow_internet_access=False,
        ) as sandbox:
            execution = sandbox.run_code(code, timeout=settings.sandbox_timeout)
    except SandboxException as e:
        return f"Sandbox error: {e}"

    output = "".join(execution.logs.stdout)
    if execution.text:
        output += execution.text
    if execution.logs.stderr:
        output += "\n[stderr]\n" + "".join(execution.logs.stderr)
    if execution.error:
        output += f"\n{execution.error.name}: {execution.error.value}"
    return _truncate(output.strip() or "(no output — use print())")


@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo for recent or general-knowledge info.

    Use this tool when the question requires fresh data (release notes,
    latest versions, news) that is NOT in the documentation corpus.

    Args:
        query: free-text search query.

    Returns:
        Top-3 search results as title + snippet text.
    """
    if not settings.enable_web_search:
        return "Web search is disabled in this environment."
    return _ddg_search.run(query)


TOOLS = [documentation_search, python_repl, web_search]