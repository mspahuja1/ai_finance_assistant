import streamlit as st
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from langchain_core.messages import HumanMessage, AIMessage
from app import app
import uuid

# --- Page Title ---
st.set_page_config(page_title="AI Finance Assistant - Mandeep Pahuja", layout="wide")
st.title("💼 AI Finance Assistant")

# --- Sidebar: Capabilities Overview ---
st.sidebar.header("Designed and developed by Mandeep Pahuja")
st.sidebar.header("🤖 What I Can Help With")

st.sidebar.markdown("""
### 🧠 Finance Q&A  
General financial education, concepts, and explanations.

### 📊 Portfolio Analysis  
Upload your portfolio or describe your holdings for insights.

### 📈 Market Analysis  
Real‑time market trends, sector movements, and macro signals.

### 🎯 Goal Planning  
Retirement, savings, budgeting, and long‑term planning.

### 📰 News Synthesis  
Summaries and insights from financial news.

### 🧾 Tax Education  
Tax concepts, account types, and optimization strategies.
""")

# Optional: File upload for portfolio analysis
uploaded_file = st.sidebar.file_uploader(
    "Upload Portfolio Data (Work in Progress)", 
    type=["csv", "xlsx", "pdf"]
)

# --- Sidebar Controls ---
st.sidebar.header("Session Controls")

# Clear UI chat only
if st.sidebar.button("Clear Chat"):
    st.session_state.pop("last_result", None)

# Start a brand new conversation (new LangGraph thread)
if st.sidebar.button("Start New Chat"):
    st.session_state["thread_id"] = f"thread-{uuid.uuid4()}"
    st.session_state.pop("last_result", None)

# ✅ Cache Statistics
st.sidebar.divider()
st.sidebar.header("💾 Cache Statistics")

try:
    from tools.cache import market_cache
    cache_stats = market_cache.get_stats()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📦 Cached", cache_stats['valid_entries'])
        st.metric("⏰ TTL", f"{cache_stats['ttl_seconds']}s")
    
    with col2:
        st.metric("🗑️ Expired", cache_stats['expired_entries'])
        st.metric("📊 Total", cache_stats['total_entries'])
    
    if st.sidebar.button("🗑️ Clear Cache"):
        market_cache.clear()
        st.sidebar.success("Cache cleared!")
except Exception as e:
    st.sidebar.error(f"Cache stats unavailable: {e}")

# ✅ Generate a unique thread ID per user session
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"thread-{uuid.uuid4()}"

# --- Chat Input ---
user_input = st.chat_input("Ask me anything about your finances...")

# ✅ When the user submits a message
if user_input:
    payload = {"messages": [HumanMessage(content=user_input)]}

    with st.spinner("🤔 Thinking..."):
        result = app.invoke(
            payload,
            config={"configurable": {"thread_id": st.session_state["thread_id"]}}
        )

    # ✅ Store the latest result so we can display it
    st.session_state["last_result"] = result

# ✅ Display conversation from LangGraph memory
if "last_result" in st.session_state:
    for msg in st.session_state["last_result"]["messages"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                # Handle tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    st.write("🔧 Fetching data...")
                else:
                    st.write(msg.content)

# ---------------------------------------------------------
#  Log file viewer 
# ---------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(current_dir), "logs")

agent_logs = {
    "🔄 MCP Transactions": "mcp_transactions.log",
    "Router": "router.log",
    "Finance Agent": "finance_agent.log",
    "Portfolio Agent": "portfolio_agent.log",
    "Market Agent": "market_agent.log",
    "Goal Agent": "goal_agent.log",
    "News Agent": "news_agent.log",
    "Tax Agent": "tax_agent.log",
}

st.subheader("Agent Logs")

for agent_name, filename in agent_logs.items():
    log_path = os.path.join(LOG_DIR, filename)
    with st.expander(agent_name):
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                st.code(f.read(), language="text")
        else:
            st.text("No logs yet.")

# ---------------------------------------------------------

# --- Optional Visualization Section ---
st.divider()
st.subheader("📊 Analysis & Visualizations")

if uploaded_file:
    st.write("Portfolio file uploaded:", uploaded_file.name)
    st.info("Portfolio analysis results will appear here.")