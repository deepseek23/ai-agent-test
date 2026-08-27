from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "notebook" / "chroma_langchain_db"
KB_DIR = PROJECT_ROOT / "knowledge-base"
ORDERS_PATH = PROJECT_ROOT / "data" / "orders.json"
EVAL_FILE = PROJECT_ROOT / "evaluation" / "visible-cases.json"

COLLECTION_NAME = "example_collection"
EMBEDDING_MODEL = "huggingface:BAAI/bge-small-en-v1.5"
CHAT_MODEL = "google_genai:gemini-3.1-flash-lite"

CANCELLATION_WINDOW_MINUTES = 30
TERMINAL_NON_DELIVERY_STATUSES = {"cancelled", "returned"}

LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = LOG_DIR / "agent.log"
LOG_MAX_TOOL_RESULT_CHARS = 2000

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

STREAMLIT_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
