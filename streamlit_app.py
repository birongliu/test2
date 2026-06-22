"""
DataPilot Streamlit Web Interface
==================================
A web UI for the DataPilot Databricks SQL copilot, powered by LangGraph and OpenAI.
"""

import logging
import os
import streamlit as st
from uuid_utils import uuid4

from main import build_agent
from knowledge import get_schema_text

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="DataPilot",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .response-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sql-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
    }
    .interpretation-box {
        background-color: black;
        padding: 1rem;
        border-left: 4px solid #0077be;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧭 DataPilot")
st.markdown("*A Databricks SQL Copilot for Analysts*")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **DataPilot** helps you write correct, efficient SQL for Databricks.
    
    - 📊 Query builder with schema awareness
    - 🔍 Smart table and column lookup
    - 🚀 Fast, read-only SQL generation
    """)
    
    st.divider()
    
    st.header("Settings")
    show_schema = st.checkbox("Show schema context", value=False)
    show_retrieval = st.checkbox("Show tool calls", value=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "agent" not in st.session_state:
    with st.spinner("Loading schema and initializing agent..."):
        try:
            schema_context = get_schema_text()
            st.session_state.agent = build_agent(schema_context)
            st.session_state.schema_context = schema_context
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()

# Display schema context if requested
if show_schema:
    with st.expander("📋 Schema Context", expanded=False):
        st.markdown("```")
        st.write(st.session_state.schema_context)
        st.markdown("```")

# Main chat interface
st.divider()
st.subheader("Ask a Question")

# Chat history display
if st.session_state.messages:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                # Parse the response to extract sections
                content = msg["content"]
                
                # Try to break down the response into logical sections
                if "Interpretation:" in content:
                    parts = content.split("\n\n")
                    for part in parts:
                        if part.strip().startswith("Interpretation:"):
                            st.markdown(f'<div class="interpretation-box">{part.replace("Interpretation:", "**Interpretation:**")}</div>', unsafe_allow_html=True)
                        elif part.strip().startswith("```sql"):
                            st.markdown(f'<div class="sql-box">{part}</div>', unsafe_allow_html=True)
                        elif part.strip():
                            st.markdown(part)
                else:
                    st.write(content)
        elif msg["role"] == "tool_call" and show_retrieval:
            with st.chat_message("assistant"):
                st.caption(f"🔧 Tool call: {msg['tool_name']}")
                st.code(msg['content'], language="json")

# User input
user_input = st.chat_input("Ask about your Databricks tables...", key="user_input")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Invoke agent
    with st.spinner("Generating response..."):
        try:
            result = st.session_state.agent.invoke(
                {"question": user_input},
                {"configurable": {"thread_id": uuid4()}},
            )
            
            response = result.get("response", "No response generated.")
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to display the new message
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            log.error(f"Agent invocation failed: {e}")

# Footer
st.divider()
st.markdown("""
---
*DataPilot* | Powered by [LangGraph](https://langchain-ai.github.io/langgraph/) + [OpenAI](https://openai.com/) + [Databricks](https://databricks.com/)

**Safety**: All queries are read-only. Write operations are refused.
""")
