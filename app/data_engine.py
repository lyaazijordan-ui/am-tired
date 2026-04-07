import streamlit as st
import pandas as pd
import numpy as np
import requests
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
import os

# ------------------------------
# --- AI Engine Configuration ---
# ------------------------------
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# quick test
st.write("connected:", bool(API_KEY))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free"
]

def query_ai(prompt, data_context=""):
    """Query OpenRouter API with fallback models."""
    if not API_KEY:
        return "⚠️ API Key missing. AI features are disabled."

    full_prompt = f"Context: {data_context}\n\nQuestion: {prompt}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-data-lab.app",
        "X-Title": "Intellectual Data Lab"
    }

    for model in FALLBACK_MODELS:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert data analyst providing insights on datasets."},
                    {"role": "user", "content": full_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }

            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error with {model}: {e}")
            continue

    return "AI Error: All model endpoints are currently unreachable."

def generate_insight(df, column):
    """Generates a short AI insight for a column."""
    summary = f"Column '{column}' - Mean: {df[column].mean():.2f}, Max: {df[column].max()}."
    return query_ai(f"Give a short expert insight on these stats: {summary}")

# ------------------------------
# --- Prediction & Anomaly ---
# ------------------------------
def predict_future(df, column, steps=5):
    if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
        return np.array([])
    y = df[column].values
    if len(y) < 2:
        return np.array([])
    X = np.arange(len(y)).reshape(-1,1)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(y), len(y)+steps).reshape(-1,1)
    return model.predict(future_X)

def detect_anomalies(df, column, threshold=2):
    if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
        return pd.DataFrame()
    data = df[column]
    mean = data.mean()
    std = data.std()
    anomalies = df[(data > mean + threshold*std) | (data < mean - threshold*std)]
    return anomalies

# ------------------------------
# --- PDF Report Generator ---
# ------------------------------
def generate_pdf(filename, insights, predictions, anomalies):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 15, "Intellectual Data Lab - Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # AI Insights
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0,0,0)
    pdf.cell(0, 10, "1. AI Insights", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, insights)

    # Predictions
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0,10,"2. Predicted Future Values", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0,8,predictions)

    # Anomalies
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0,10,"3. Anomalies Detected", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0,8,str(anomalies))

    pdf.output(filename)

# ------------------------------
# --- Streamlit App Layout ---
# ------------------------------
st.set_page_config(page_title="Intellectual Data Lab", page_icon="🧠", layout="wide")
st.title("🧠 Intellectual Data Lab")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv","xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.session_state['uploaded_data'] = df
    st.success(f"Loaded {uploaded_file.name} ({df.shape[0]} rows x {df.shape[1]} columns)")

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # Select column for analysis
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        column = st.selectbox("Select numeric column for predictions & anomalies", numeric_cols)

        # Generate AI insight
        with st.spinner("Generating AI insights..."):
            insight_text = generate_insight(df, column)
        st.markdown(f"**AI Insight:** {insight_text}")

        # Predict future values
        future_steps = st.slider("How many future steps to predict?", 1, 20, 5)
        predictions = predict_future(df, column, steps=future_steps)
        st.markdown(f"**Future Predictions ({future_steps} steps):** {predictions}")

        # Detect anomalies
        anomalies = detect_anomalies(df, column)
        st.markdown(f"**Anomalies Detected:** {len(anomalies)} rows")
        if not anomalies.empty:
            st.dataframe(anomalies)

        # PDF Report
        if st.button("Generate PDF Report"):
            filename = f"report_{column}.pdf"
            generate_pdf(filename, insight_text, predictions, anomalies)
            st.success(f"PDF report generated: {filename}")
            st.download_button("Download Report", filename, file_name=filename)

    else:
        st.info("No numeric columns available for prediction or anomaly detection.")
