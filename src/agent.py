import logging
import time
from typing import Any

from src.config import LOG_MAX_TOOL_RESULT_CHARS
from src.llm import get_agent
from src.security import security

logger = logging.getLogger(__name__)


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _truncate(text: str, max_chars: int = LOG_MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated, {len(text)} chars total]"


def _trace_event(trace: list[dict[str, Any]] | None, event: str, **fields: Any) -> None:
    if trace is not None:
        trace.append({"event": event, **fields})


def _log_messages(thread_id: str, messages: list, trace: list[dict[str, Any]] | None) -> None:
    logger.info("AGENT MESSAGE TRACE | thread=%s | total_messages=%d", thread_id, len(messages))

    for index, message in enumerate(messages):
        msg_type = getattr(message, "type", type(message).__name__)
        msg_name = getattr(message, "name", None)
        content = _truncate(_message_text(message))

        logger.debug(
            "AGENT MESSAGE | thread=%s | index=%d | type=%s | name=%s | content=%r",
            thread_id,
            index,
            msg_type,
            msg_name,
            content,
        )

        _trace_event(
            trace,
            "message",
            index=index,
            type=msg_type,
            name=msg_name,
            content=content,
        )

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args")
                logger.info(
                    "AGENT TOOL CALL | thread=%s | tool=%s | args=%s",
                    thread_id,
                    tool_name,
                    tool_args,
                )
                _trace_event(
                    trace,
                    "tool_call",
                    tool=tool_name,
                    args=tool_args,
                )

        if getattr(message, "type", None) == "tool":
            tool_name = getattr(message, "name", "unknown")
            result_text = _truncate(_message_text(message))
            logger.info(
                "AGENT TOOL RESULT | thread=%s | tool=%s | result=%r",
                thread_id,
                tool_name,
                result_text,
            )
            _trace_event(
                trace,
                "tool_result",
                tool=tool_name,
                result=result_text,
            )


def run_agent(
    user_input: str,
    thread_id: str = "user-1",
    include_trace: bool = False,
) -> str | tuple[str, list[dict[str, Any]]]:
    trace: list[dict[str, Any]] | None = [] if include_trace else None
    started_at = time.perf_counter()

    logger.info(
        "AGENT RUN START | thread=%s | raw_input=%r",
        thread_id,
        user_input,
    )
    _trace_event(trace, "run_start", thread_id=thread_id, raw_input=user_input)

    allowed, cleaned_input, notes = security.check_input(user_input)

    logger.info(
        "AGENT INPUT | thread=%s | allowed=%s | input=%r",
        thread_id,
        allowed,
        cleaned_input if allowed else "[BLOCKED]",
    )
    _trace_event(
        trace,
        "input",
        allowed=allowed,
        cleaned_input=cleaned_input if allowed else None,
        notes=notes,
    )

    if notes:
        logger.warning(
            "INPUT SECURITY | thread=%s | notes=%s",
            thread_id,
            notes,
        )

    if not allowed:
        logger.warning("AGENT BLOCKED | thread=%s", thread_id)
        blocked_message = notes[0]
        _trace_event(trace, "blocked", reason=blocked_message)
        if include_trace:
            return blocked_message, trace
        return blocked_message

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    agent = get_agent()
    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": cleaned_input,
                    }
                ]
            },
            config=config,
        )
    except Exception:
        logger.exception("AGENT INVOKE FAILED | thread=%s", thread_id)
        raise

    messages = result.get("messages", [])
    _log_messages(thread_id, messages, trace)

    if not messages:
        logger.error("AGENT EMPTY RESPONSE | thread=%s", thread_id)
        empty_message = "I could not generate a response. Please contact support."
        _trace_event(trace, "empty_response")
        if include_trace:
            return empty_message, trace
        return empty_message

    response = _message_text(messages[-1])
    safe_response, output_notes = security.check_output(response)

    if output_notes:
        logger.warning(
            "OUTPUT SECURITY | thread=%s | notes=%s",
            thread_id,
            output_notes,
        )
        _trace_event(trace, "output_security", notes=output_notes)

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "AGENT OUTPUT | thread=%s | elapsed_ms=%s | response=%r",
        thread_id,
        elapsed_ms,
        safe_response,
    )
    _trace_event(
        trace,
        "run_complete",
        elapsed_ms=elapsed_ms,
        response=safe_response,
    )
    logger.info("AGENT RUN END | thread=%s | elapsed_ms=%s", thread_id, elapsed_ms)

    if include_trace:
        return safe_response, trace
    return safe_response
