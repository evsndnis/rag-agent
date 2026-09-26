import re

from app.config import settings

_PROMPT_INJECTION_PATTERNS = [
    r"ignore (previous|all) instructions",
    r"you are (now|a)",
    r"system prompt",
    r"<\|im_(start|end)\|>",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PROMPT_INJECTION_PATTERNS]

_ALLOWED_CHARS = re.compile(
    r"^[A-Za-zА-Яа-я0-9\s.,?!()\-—:;'\"$%/+=\[\]*^<>_&|@#~]+$"
)

_DECLARATIVE_MARKERS = (
    "according to",
    "as stated",
    "the documentation says",
    "согласно",
    "по документации",
)


class GuardrailError(ValueError):
    pass


def check_input(question: str) -> None:
    if not (1 <= len(question) <= 500):
        raise GuardrailError(f"length out of range: {len(question)}")
    if not _ALLOWED_CHARS.match(question):
        raise GuardrailError("forbidden characters in question")
    for pattern in _COMPILED:
        if pattern.search(question):
            raise GuardrailError(f"prompt-injection pattern: {pattern.pattern}")


def check_output(answer: str, trace_tools: list[str]) -> tuple[str, str | None]:
    """Return (safe_answer, triggered_reason_or_None)."""
    if len(answer) > settings.agent_max_output_chars:
        return (
            answer[: settings.agent_max_output_chars] + "\n…(truncated)",
            "output_too_long",
        )

    lowered = answer.lower()
    has_declarative = any(marker in lowered for marker in _DECLARATIVE_MARKERS)
    used_doc_search = "documentation_search" in trace_tools

    if has_declarative and not used_doc_search:
        return (
            "I cannot make a documentation-grounded claim without consulting "
            "the corpus first. Please rephrase or ask a more specific question.",
            "declarative_without_source",
        )

    return answer, None