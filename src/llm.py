import logging

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from src.config import CHAT_MODEL
from src.tools import make_order_lookup_tool
from src.prompts import SYSTEM_PROMPT
from src.retriever import build_hybrid_retriever
from src.security import wrap_tools

load_dotenv()

logger = logging.getLogger(__name__)

_memory = MemorySaver()
_agent = None
_hybrid_retriever = None


def get_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = build_hybrid_retriever()
        logger.info("Hybrid retriever ready")
    return _hybrid_retriever


def get_agent():
    global _agent
    if _agent is None:
        logger.info("AGENT INIT START | model=%s", CHAT_MODEL)

        get_retriever()
        logger.info("AGENT INIT | hybrid retriever ready")

        order_tool = make_order_lookup_tool()
        tools = wrap_tools(order_tool)
        logger.info("AGENT INIT | tools ready | count=%d", len(tools))

        llm = init_chat_model(CHAT_MODEL)
        logger.info("AGENT INIT | chat model loaded")

        _agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=_memory,
        )
        logger.info("AGENT INIT COMPLETE")
    return _agent


def get_memory():
    return _memory
