import os
from dotenv import load_dotenv
import requests
import streamlit as st
from fpdf import FPDF
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------
load_dotenv()  # Load .env file
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    st.warning("⚠️ OpenRouter API key not found. AI features will be disabled.")
else:
    st.success("✅ OpenRouter API key loaded successfully")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free"
]

# -----------------------------
# AI QUERY FUNCTION
# -----------------------------
def query_ai(prompt, data_context=""):
    """Query OpenRouter API with fallback models."""
    if not API_KEY:
        return "Error: OpenRouter API Key missing. Please set OPENROUTER_API_KEY in your environment."

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

            # DEBUG: log response for troubleshooting
            print(f"[DEBUG] Model: {model}, Status: {response.status_code}, Response: {response.text[:200]}")

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

            print(f"[WARNING] Model {model} returned status {response.status_code}. Trying next...")

        except Exception as e:
            print(f"[ERROR] Model {model} failed: {e}")
            continue

    return "AI Error: All model endpoints are currently unreachable. Please check your internet or API key."

# -----------------------------
# INSIGHTS AND ANALYSIS
# -----------------------------
def generate_insight(df, column):
    """Generates an automated summary insight for a column."""
    summary = (
        f"Column '{column}' - Mean: {df[column].mean():.2f}, "
        f"Max: {df[column].max()}, Min: {df[column].min()}."
    )
    return query_ai(f"Give a short expert insight on these stats: {summary}")

def medical_analysis(df):
    """Specialized medical data check."""
    cols = list(df.columns)
    prompt = (
        f"Identify potential health-related risks or trends in these data headers: {cols}. "
        "Disclaimer: This is not medical advice."
    )
    return query_ai(prompt)

# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------
def predict_future(df, column, steps=5):
    """Predicts future values of a numeric column using linear regression."""
    y = df[column].values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
    predictions = model.predict(future_X)

    return predictions

def detect_anomalies(df, column):
    """Detects anomalies that lie beyond 2 standard deviations from the mean."""
    data = df[column]
    mean = data.mean()
    std = data.std()

    anomalies = df[(data > mean + 2*std) | (data < mean - 2*std)]
    return anomalies

# -----------------------------
# PDF REPORT GENERATOR
# -----------------------------
def generate_pdf(filename, insights, preds, anomalies):
    """Creates a branded PDF report for Intellectual Data Lab."""
    pdf = FPDF()
    pdf.add_page()

    # Header Branding
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 15, "Intellectual Data Lab - Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Section 1: AI Insights
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. AI Insights", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, str(insights))

    # Section 2: Predictions & Anomalies
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Statistical Findings", ln=True)
    pdf.set_font("Arial", size=11)
    content = f"Predictions:\n{preds}\n\nAnomalies Detected:\n{anomalies}"
    pdf.multi_cell(0, 8, content)

    pdf.output(filename)
