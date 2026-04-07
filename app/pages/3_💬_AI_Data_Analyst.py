import streamlit as st
import sys, os

# Path fix
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Correct imports (cleaned)
from ai_engine import query_ai, auto_graph, explain_data, suggest_graph

# Page config
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Analyst")

# Check dataset
if 'uploaded_data' not in st.session_state:
    st.warning("Upload data first")
    st.stop()

df = st.session_state['uploaded_data']

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# DISPLAY CHAT
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

    # Build context
    context = f"""
    Dataset info:
    Rows: {df.shape[0]}
    Columns: {df.shape[1]}
    Column Names: {list(df.columns)}
    Sample Data: {df.head(3).to_dict()}
    """

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("AI analyzing your data..."):
            response = query_ai(
    prompt=prompt,
    context=full_context,
    df=df,
    api_key=OPENROUTER_API_KEY   # <-- pass the Streamlit secret here
            )
            st.write(response)

    # Save response
    st.session_state.chat.append({"role": "assistant", "content": response})

# -----------------------------
# AUTO GRAPH MAGIC
# -----------------------------
st.subheader("📊 Auto Visualization")

# -----------------------------
# AI GRAPH SUGGESTION
# -----------------------------
with st.spinner("AI choosing best visualizations..."):
    suggestion = suggest_graph(df, api_key=OPENROUTER_API_KEY)
    st.info(suggestion)

# -----------------------------
# GENERATE GRAPHS
# -----------------------------
charts = auto_graph(df)

if charts:
    st.subheader("📊 Smart Data Dashboard")

    for chart in charts:
        st.plotly_chart(chart, use_container_width=True)

    # -----------------------------
    # AI EXPLANATION
    # -----------------------------
    st.subheader("🧠 AI Insights")

    with st.spinner("Analyzing dataset insights..."):
        insight = explain_data(df)
        st.success(insight)

    # -----------------------------
    # AI EXPLANATION
    # -----------------------------
    st.subheader("🧠 AI Insights")

    with st.spinner("Analyzing dataset insights..."):
        insight = explain_data(df)
        st.success(insight)
