from langchain_core.tools import tool

from app.rag.chain import build_rag_chain

_chain = None
_retriever = None


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