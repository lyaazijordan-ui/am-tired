import streamlit as st
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai_engine import query_ai, auto_graph

st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Analyst")

if 'uploaded_data' not in st.session_state:
    st.warning("Upload data first")
    st.stop()

df = st.session_state['uploaded_data']

# Chat memory
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display chat
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
prompt = st.chat_input("Ask about your data...")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = query_ai(prompt, df=df)
        st.write(response)

    st.session_state.chat.append({"role": "assistant", "content": response})

# AUTO GRAPH MAGIC
st.subheader("📊 Auto Visualization")
fig = auto_graph(df)

if fig:
    st.plotly_chart(fig, use_container_width=True)
