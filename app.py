import json
import requests
from datetime import datetime
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL   = "http://localhost:8000"
CHAT_ENDPOINT  = f"{API_BASE_URL}/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

st.set_page_config(page_title="Customer Support", page_icon="💬", layout="centered")

# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("thread_id", None),
    ("api_ok", None),
    ("show_trace", False),
    ("pending_suggestion", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_health() -> bool:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=4)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False

def send_message(message: str) -> dict:
    payload = {
        "message": message,
        "thread_id": st.session_state.thread_id,
        "include_trace": st.session_state.show_trace,
    }
    r = requests.post(CHAT_ENDPOINT, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

if st.session_state.api_ok is None:
    st.session_state.api_ok = check_health()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💬 Support Chat")
    st.caption("Aster & Row")
    
    if st.button("✏️ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()

    st.divider()
    st.session_state.show_trace = st.toggle("Show agent trace", value=st.session_state.show_trace)
    
    if st.session_state.thread_id:
        st.divider()
        st.caption("Session ID")
        st.code(st.session_state.thread_id)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Customer Support")

if not st.session_state.api_ok:
    st.error("⚠️ The support service is offline. Please ensure the API is running.", icon="🚨")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("trace") and msg["role"] == "assistant":
            with st.expander("🔍 Agent trace"):
                st.json(msg["trace"])

# Suggestions (Only display if no messages exist)
if not st.session_state.messages:
    st.info("👋 How can we help you? Ask us about orders, returns, product care, or anything else.")
    
    suggestions = ["Track my order", "Start a return", "Product care", "Store hours", "Talk to a person"]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion):
            st.session_state.pending_suggestion = suggestion
            st.rerun()

# Input handling (from input box or queued suggestion)
prompt = st.chat_input("Message support...", disabled=not st.session_state.api_ok)

if st.session_state.pending_suggestion:
    prompt = st.session_state.pending_suggestion
    st.session_state.pending_suggestion = None

if prompt:
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt, "ts": datetime.now()})
    with st.chat_message("user"):
        st.write(prompt)
        
    # Process and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                data = send_message(prompt)
                st.session_state.thread_id = data["thread_id"]
                
                msg_data = {
                    "role": "assistant",
                    "content": data["response"],
                    "ts": datetime.now(),
                    "trace": data.get("trace")
                }
                st.write(data["response"])
                
                if msg_data.get("trace"):
                    with st.expander("🔍 Agent trace"):
                        st.json(msg_data["trace"])
                
                st.session_state.messages.append(msg_data)
                
            except requests.exceptions.ConnectionError:
                err = "⚠️ Couldn't reach the support service. Please try again in a moment."
                st.write(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "ts": datetime.now()})
            except Exception as e:
                err = f"⚠️ Something went wrong: {e}"
                st.write(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "ts": datetime.now()})