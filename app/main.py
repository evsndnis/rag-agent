import logging
import time
import uuid
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.rag.chain import build_rag_chain
from app.agent.graph import build_agent_graph
from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.agent.guardrails import GuardrailError, check_input, check_output
from app.schemas.agent import AgentRequest, AgentResponse, Source as AgentSource, TraceStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_chain = None
_retriever = None
_agent_graph = None
_agent_checkpointer = None

# LaTeX delimiters для Gradio Chatbot. LLM-ответы про Ridge, Lasso, метрики
# содержат формулы $$..$$ / \[..\] / $..$ — без этого блока они отрисуются
# как сырые строки `$\ell_1$`.
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain, _retriever, _agent_graph, _agent_checkpointer
    _chain, _retriever = build_rag_chain()
    _agent_checkpointer = MemorySaver()
    _agent_graph = build_agent_graph(checkpointer=_agent_checkpointer)
    print("RAG chain + agent graph ready")
    yield
    _chain = None
    _retriever = None
    _agent_graph = None
    _agent_checkpointer = None


app = FastAPI(title="RAG service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    docs = _retriever.invoke(payload.question)
    try:
        answer = _chain.invoke(payload.question)
    except Exception as exc:
        logger.exception("LLM call failed in /chat: %r", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM provider temporarily unavailable. "
                f"Try again in 30-60 seconds. Raw: {type(exc).__name__}"
            ),
        ) from exc
    sources = [
        Source(
            url=doc.metadata.get("source", "unknown"),
            snippet=doc.page_content[:200].strip(),
            full_context=doc.page_content.strip(),
        )
        for doc in docs
    ]
    return ChatResponse(answer=answer, sources=sources)


def _extract_trace(messages: list, sources: list) -> tuple[list[TraceStep], list[str]]:
    steps: list[TraceStep] = []
    tools_used: list[str] = []
    step_num = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                step_num += 1
                tools_used.append(tc["name"])
                steps.append(TraceStep(
                    step=step_num,
                    node="agent",
                    tool=tc["name"],
                    input=tc.get("args", {}),
                    output="(tool requested)",
                    latency_ms=0,
                ))
        elif isinstance(msg, ToolMessage):
            if steps and steps[-1].output == "(tool requested)":
                steps[-1].output = msg.content[:500]
    return steps, tools_used


@app.post("/agent", response_model=AgentResponse)
def agent_chat(payload: AgentRequest) -> AgentResponse:
    try:
        check_input(payload.question)
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=f"Input rejected: {e}") from e

    thread_id = payload.thread_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    result = _agent_graph.invoke(
        {"messages": [HumanMessage(content=payload.question)], "iteration_count": 0},
        config={"configurable": {"thread_id": thread_id}},
    )
    total_ms = int((time.perf_counter() - t0) * 1000)

    raw_answer = result["messages"][-1].content
    trace_steps, tools_used = _extract_trace(result["messages"], sources=[])
    if trace_steps:
        trace_steps[-1].latency_ms = total_ms

    safe_answer, guardrail_reason = check_output(raw_answer, tools_used)

    sources: list[AgentSource] = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and "Sources:" in msg.content:
            body, _, sources_block = msg.content.partition("Sources:")
            tool_text = body.strip()  # текст, который агент реально получил от инструмента
            for line in sources_block.strip().splitlines():
                url = line.strip().lstrip("- ").strip()
                if url:
                    sources.append(AgentSource(url=url, snippet=tool_text[:200], full_context=tool_text))

    return AgentResponse(
        answer=safe_answer,
        trace=trace_steps,
        sources=sources,
        guardrail_triggered=guardrail_reason,
        iterations=result.get("iteration_count", 0),
    )


def _format_timings(retrieval_ms: float, llm_ms: float | None, llm_error: str | None) -> str:
    lines = [
        "### ⏱ Тайминги последнего запроса",
        "",
        f"- 🔍 **Retrieval (embed + Qdrant):** {retrieval_ms:.0f} ms",
    ]
    if llm_ms is not None:
        lines.append(f"- 🤖 **LLM call:** {llm_ms:.0f} ms")
        lines.append(f"- 📊 **Total:** {retrieval_ms + llm_ms:.0f} ms")
    else:
        lines.append(f"- 🤖 **LLM call:** ❌ {llm_error}")
    return "\n".join(lines)


def _format_sources(docs: list) -> str:
    if not docs:
        return "### 📚 Источники\n\n_Ничего не найдено_"
    lines = ["### 📚 Источники", ""]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:140].strip().replace("\n", " ")
        lines.append(f"**[{i}]** `{source}`")
        lines.append(f"> {snippet}…")
        lines.append("")
    return "\n".join(lines)


def respond(message: str, history: list):
    """Streaming Gradio handler — generator that yields on every chunk."""
    if not message or not message.strip():
        yield history, "", "### ⏱ Тайминги\n\n_Пустой запрос_", "### 📚 Источники\n\n_—_"
        return

    history = history + [{"role": "user", "content": message}]

    t0 = time.perf_counter()
    docs = _retriever.invoke(message)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    sources_panel = _format_sources(docs)

    # Yield #1: sources уже на экране, LLM ещё не начал писать.
    history.append({"role": "assistant", "content": ""})
    yield (
        history, "",
        "### ⏱ Тайминги\n\n"
        f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
        "- 🤖 **LLM:** _streaming…_",
        sources_panel,
    )

    t1 = time.perf_counter()
    ttft_ms: float | None = None
    accumulated = ""
    try:
        for chunk in _chain.stream(message):
            if not chunk:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t1) * 1000
            accumulated += chunk
            history[-1]["content"] = accumulated
            yield (
                history, "",
                f"### ⏱ Тайминги\n\n"
                f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
                f"- ⚡ **TTFT (1st token):** {ttft_ms:.0f} ms\n"
                f"- 🤖 **LLM:** _streaming… {len(accumulated)} chars_",
                sources_panel,
            )

        llm_total_ms = (time.perf_counter() - t1) * 1000
        yield (
            history, "",
            "### ⏱ Тайминги последнего запроса\n\n"
            f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
            f"- ⚡ **TTFT:** {ttft_ms:.0f} ms\n"
            f"- 🤖 **LLM stream (full):** {llm_total_ms:.0f} ms\n"
            f"- 📊 **Total:** {retrieval_ms + llm_total_ms:.0f} ms",
            sources_panel,
        )
    except Exception as exc:
        logger.exception("LLM call failed in UI stream: %r", exc)
        history[-1]["content"] = (
            f"⚠️ LLM-провайдер сейчас недоступен ({type(exc).__name__}). "
            f"Попробуй через 30-60 секунд."
        )
        yield (
            history, "",
            _format_timings(retrieval_ms, None, type(exc).__name__),
            sources_panel,
        )


# CSS делает три вещи: 1) распахивает контейнер на всю ширину,
# 2) фиксирует высоту чата и боковой панели на calc(100vh - 220px) —
#    минус headers — чтобы при наборе сообщения чат НЕ сжимался,
# 3) добавляет видимую границу между чатом и боковой панелью.
CSS = """
.gradio-container { max-width: 100% !important; padding: 1rem !important; }
#chatbot { height: calc(100vh - 220px) !important; min-height: 500px !important; }
#side-panel { height: calc(100vh - 220px) !important; overflow-y: auto !important;
              padding: 1rem !important; border-left: 1px solid #ddd !important; }
"""

with gr.Blocks(
    title="scikit-learn docs RAG",
    css=CSS,
    fill_height=True,
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 📖 scikit-learn docs RAG assistant\n"
        "_Спрашивай про Linear models, Decision trees, Metrics — на русском или английском._"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                latex_delimiters=LATEX_DELIMITERS,
                show_copy_button=True,
                avatar_images=(None, None),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Например: «Покажи формулу Ridge» или «Чем precision отличается от recall»",
                    scale=8,
                    container=False,
                    autofocus=True,
                )
                send = gr.Button("Отправить", scale=1, variant="primary")
            gr.Examples(
                examples=[
                    "How does Ridge regression work?",
                    "Что ты умеешь?",
                    "Объясни разницу между precision и recall с формулами",
                    "When does a decision tree overfit?",
                ],
                inputs=msg,
            )
        with gr.Column(scale=1, elem_id="side-panel"):
            timings_md = gr.Markdown(
                "### ⏱ Тайминги последнего запроса\n\n_Задайте вопрос, чтобы увидеть тайминги._"
            )
            sources_md = gr.Markdown("### 📚 Источники\n\n_—_")

    msg.submit(respond, [msg, chatbot], [chatbot, msg, timings_md, sources_md])
    send.click(respond, [msg, chatbot], [chatbot, msg, timings_md, sources_md])


app = gr.mount_gradio_app(app, demo, path="/")