import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import run_agent
from src.config import API_HOST, API_PORT, STREAMLIT_ORIGINS
from src.llm import get_agent
from src.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aster & Row Support Agent",
    description="RAG-powered customer support API with multi-turn conversation history.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=STREAMLIT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID for multi-turn history. Omit to start a new session.",
    )
    include_trace: bool = Field(
        default=False,
        description="Return structured agent trace (tool calls, results, timing) in the response.",
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    trace: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str


@app.on_event("startup")
def startup():
    logger.info("API startup | loading agent | host=%s port=%s", API_HOST, API_PORT)
    get_agent()
    logger.info("API startup complete")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    logger.info(
        "API CHAT REQUEST | thread=%s | include_trace=%s | message=%r",
        thread_id,
        request.include_trace,
        request.message,
    )

    try:
        if request.include_trace:
            response, trace = run_agent(
                request.message,
                thread_id=thread_id,
                include_trace=True,
            )
            return ChatResponse(response=response, thread_id=thread_id, trace=trace)

        response = run_agent(request.message, thread_id=thread_id)
        return ChatResponse(response=response, thread_id=thread_id)
    except Exception as exc:
        logger.exception("API CHAT FAILED | thread=%s", thread_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_config=None,
    )
