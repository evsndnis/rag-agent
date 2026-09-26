import logging
import time
import uuid
from contextlib import asynccontextmanager

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

import gradio as gr
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.rag.chain import build_rag_chain
from app.agent.graph import build_agent_graph
from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.agent.guardrails import GuardrailError, check_input, check_output
from app.schemas.agent import AgentRequest, AgentResponse, Source as AgentSource, TraceStep

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

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


class DisableProxyBufferingMiddleware(BaseHTTPMiddleware):
    """Tell nginx (X-Accel-Buffering) and others not to buffer responses on
    Gradio's SSE-streaming endpoints. Без этого live-обновления статуса
    приходят одним пакетом в конце."""

    SSE_PATHS = ("/queue/data", "/queue/join")

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if any(request.url.path.startswith(p) for p in self.SSE_PATHS):
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["Cache-Control"] = "no-cache"
        return response


app.add_middleware(DisableProxyBufferingMiddleware)

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


def respond_agent(message: str, history: list, thread_id_state: str):
    """Streaming handler for agent mode.

    Yields на каждом событии графа: смена статуса, токены LLM, финал.
    Gradio дорисовывает UI на каждый yield.
    """
    if not message or not message.strip():
        yield history, "", "_Пустой запрос_", "_—_", "_—_", "💤 Готов к работе", thread_id_state
        return

    try:
        check_input(message)
    except GuardrailError as e:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ Запрос отклонён guardrails: {e}"},
        ]
        yield history, "", "_Отклонён guardrails_", "_—_", "_—_", "🛡 Guardrail отклонил запрос", thread_id_state
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    thread_id = thread_id_state or str(uuid.uuid4())

    t0 = time.perf_counter()
    yield history, "", "_Старт…_", "_—_", "_—_", "🤔 LLM решает, какой инструмент вызвать…", thread_id

    accumulated = ""
    final_messages: list = []
    tool_calls_count = 0
    iterations = 0

    for stream_mode, payload in _agent_graph.stream(
        {"messages": [HumanMessage(content=message)], "iteration_count": 0},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["updates", "messages"],
    ):
        if stream_mode == "updates":
            # payload = {"agent": {...}} или {"tool_executor": {...}}
            node, state_update = next(iter(payload.items()))
            node_msgs = state_update.get("messages", []) or []
            final_messages = (final_messages + node_msgs) if node_msgs else final_messages

            if node == "agent":
                iterations += 1
                last = node_msgs[-1] if node_msgs else None
                tool_calls = getattr(last, "tool_calls", None) or []
                if tool_calls:
                    tc = tool_calls[0]
                    name = tc["name"]
                    tool_calls_count += 1
                    status = {
                        "documentation_search": "📚 Ищу в документации scikit-learn…",
                        "python_repl": "🧮 Считаю в Python REPL…",
                        "web_search": "🌐 Ищу в интернете…",
                    }.get(name, f"🛠 Вызываю {name}…")
                    yield history, "", "_В процессе…_", "_—_", "_—_", status, thread_id
            elif node == "tool_executor":
                tool_msg = node_msgs[-1] if node_msgs else None
                tool_name = getattr(tool_msg, "name", "tool")
                length = len(getattr(tool_msg, "content", "") or "")
                status = f"🔍 Анализирую результат {tool_name} ({length} симв.), решаю что делать дальше…"
                yield history, "", "_В процессе…_", "_—_", "_—_", status, thread_id

        elif stream_mode == "messages":
            # payload = (AIMessageChunk, metadata)
            msg_chunk, meta = payload
            chunk_text = getattr(msg_chunk, "content", "") or ""
            # печатаем только финальный ответ агента, не reasoning-промежутки tool-вызовов
            if chunk_text and meta.get("langgraph_node") == "agent" and not getattr(msg_chunk, "tool_call_chunks", None):
                accumulated += chunk_text
                history[-1]["content"] = accumulated
                yield history, "", "_…_", "_—_", "_—_", f"✍️ LLM пишет ответ потоком… {len(accumulated)} симв.", thread_id

    total_ms = int((time.perf_counter() - t0) * 1000)

    trace_steps, tools_used = _extract_trace(final_messages, sources=[])
    safe_answer, _ = check_output(accumulated, tools_used)
    history[-1]["content"] = safe_answer

    trace_md_lines = ["### Шаги агента", ""]
    for s in trace_steps:
        trace_md_lines.append(
            f"**{s.step}.** `{s.node}` → tool `{s.tool}` · "
            f"args `{s.input}` · output `{s.output[:120]}…`"
        )
    trace_md = "\n\n".join(trace_md_lines) if trace_steps else "_Без tool-вызовов_"

    timings = (
        f"### ⏱ Тайминги\n\n"
        f"- 🤖 **Total:** {total_ms} ms\n"
        f"- 🔄 **Итераций:** {iterations}\n"
        f"- 🛠 **Tool-вызовов:** {tool_calls_count}"
    )
    sources_panel = "### 📚 Источники\n\n_см. блок Шаги агента_"
    final_status = f"✅ Готово за {total_ms / 1000:.1f} сек · {tool_calls_count} tool-вызовов · {iterations} итераций"

    yield history, "", timings, sources_panel, trace_md, final_status, thread_id


def _route_respond(message, history, mode, thread_id_state):
    if mode == "Агент (/agent)":
        yield from respond_agent(message, history, thread_id_state)
    else:
        # старый /chat handler — добавляем пустой trace и нейтральный статус
        for out in respond(message, history):
            yield (*out, "_(режим Быстрый — trace не используется)_", "✅ Готово", thread_id_state)


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
    status_line = gr.Markdown(
        "💤 Готов к работе",
        elem_id="status-line",
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
            mode_radio = gr.Radio(
                choices=["Быстрый (/chat)", "Агент (/agent)"],
                value="Быстрый (/chat)",
                label="Режим",
            )
            timings_md = gr.Markdown(
                "### ⏱ Тайминги последнего запроса\n\n_Задайте вопрос, чтобы увидеть тайминги._"
            )
            sources_md = gr.Markdown("### 📚 Источники\n\n_—_")
            with gr.Accordion("Что сделал агент", open=False):
                trace_md = gr.Markdown("_Включите режим «Агент», чтобы увидеть шаги._")

    # thread_id агента живёт в сессии браузера: один диалог — один thread в checkpointer
    thread_id_state = gr.State("")

    inputs = [msg, chatbot, mode_radio, thread_id_state]
    outputs = [chatbot, msg, timings_md, sources_md, trace_md, status_line, thread_id_state]

    msg.submit(_route_respond, inputs, outputs)
    send.click(_route_respond, inputs, outputs)


app = gr.mount_gradio_app(app, demo, path="/")