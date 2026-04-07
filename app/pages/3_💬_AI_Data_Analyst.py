# data_analyst_page.py

import streamlit as st
import sys, os
import pandas as pd

# Path fix for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import AI engine functions
from ai_engine import query_ai, auto_graph, explain_data, suggest_graph

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")
st.title("🤖 AI Data Analyst")

# -----------------------------
# UPLOAD DATA
# -----------------------------
if 'uploaded_data' not in st.session_state:
    uploaded_file = st.file_uploader("Upload your CSV or Excel dataset", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state['uploaded_data'] = df
        st.success(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    else:
        st.warning("Upload a dataset to start analyzing.")
        st.stop()

df = st.session_state['uploaded_data']

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# DISPLAY EXISTING CHAT
# -----------------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
prompt = st.chat_input("Ask about your data...")

if prompt:
    # Save user message
    st.session_state.chat.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Build dataset + chat context
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat])
    context = f"""
Dataset info:
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Column Names: {list(df.columns)}
Sample Data: {df.head(3).to_dict()}
"""
    full_context = f"{chat_history}\n\n{context}"

    # AI RESPONSE
    with st.chat_message("assistant"):
        with st.spinner("AI analyzing your data..."):
            try:
                response = query_ai(prompt, context=full_context, df=df)
            except Exception as e:
                response = f"AI failed: {str(e)}"
            st.write(response)

    # Save AI response
    st.session_state.chat.append({"role": "assistant", "content": response})

# -----------------------------
# AUTO GRAPH MAGIC
# -----------------------------
st.subheader("📊 Auto Visualization")

# AI GRAPH SUGGESTION
with st.spinner("AI choosing best visualizations..."):
    try:
        suggestion = suggest_graph(df)
    except Exception as e:
        suggestion = f"Graph suggestion failed: {str(e)}"
    st.info(suggestion)

# GENERATE CHARTS
charts = auto_graph(df)
if charts:
    st.subheader("📊 Smart Data Dashboard")
    for chart in charts:
        st.plotly_chart(chart, use_container_width=True)

# AI INSIGHTS
st.subheader("🧠 AI Insights")
with st.spinner("Analyzing dataset insights..."):
    try:
        insight = explain_data(df)
    except Exception as e:
        insight = f"Insight generation failed: {str(e)}"
    st.success(insight)
