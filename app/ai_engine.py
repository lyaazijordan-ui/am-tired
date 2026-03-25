# ai_engine.py
import os
import requests
import streamlit as st
from fpdf import FPDF
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# -----------------------------
# API CONFIGURATION
# -----------------------------
API_KEY = os.getenv("OPENROUTER_API_KEY")  # Picked from Streamlit Secrets
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free"
]

if not API_KEY:
    st.warning("⚠️ No OpenRouter API key found. AI features will be limited.")

# -----------------------------
# LOCAL FALLBACK AI
# -----------------------------
def local_ai_answer(prompt, df=None):
    """Simple local AI fallback for basic dataset analysis."""
    prompt_lower = prompt.lower()
    
    if df is None:
        return "Local AI: No dataset available for analysis."

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    try:
        if "summary" in prompt_lower:
            summary_text = "Local AI: Dataset Summary:\n"
            for col in numeric_cols:
                summary_text += f"- {col}: mean={df[col].mean():.2f}, min={df[col].min()}, max={df[col].max()}, std={df[col].std():.2f}\n"
            return summary_text

        elif "correlation" in prompt_lower:
            corr = df[numeric_cols].corr()
            return f"Local AI: Correlation Matrix:\n{corr.round(2)}"

        elif "trend" in prompt_lower:
            trends = ""
            for col in numeric_cols:
                if len(df[col]) > 1:
                    if df[col].iloc[-1] > df[col].iloc[0]:
                        trends += f"- {col} appears to be increasing.\n"
                    elif df[col].iloc[-1] < df[col].iloc[0]:
                        trends += f"- {col} appears to be decreasing.\n"
                    else:
                        trends += f"- {col} is stable.\n"
            return f"Local AI: Trends Observed:\n{trends}"

        elif "anomaly" in prompt_lower:
            anomalies_text = ""
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                anomalies = df[(df[col] > mean + 2*std) | (df[col] < mean - 2*std)]
                anomalies_text += f"- {col}: {len(anomalies)} anomalies detected\n"
            return f"Local AI: Anomaly Detection Results:\n{anomalies_text}"

        elif "medical" in prompt_lower:
            return "Local AI: Basic medical data check is limited. Check column distributions and outliers."

        else:
            return "Local AI: I can summarize, find correlations, trends, anomalies, or do basic medical checks."
    except Exception as e:
        return f"Local AI Error: {e}"

# -----------------------------
# AI QUERY FUNCTION
# -----------------------------
def query_ai(prompt, data_context="", df=None):
    """Query OpenRouter API with fallback to local AI."""
    if not API_KEY:
        # If no API key, always use local AI
        return local_ai_answer(prompt, df)

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
                try:
                    return result["choices"][0]["message"]["content"].strip()
                except KeyError:
                    continue  # If API response is weird, try next model

            print(f"Model {model} returned status {response.status_code}. Trying next...")

        except Exception as e:
            print(f"Error with {model}: {e}")
            continue

    # If all models fail, fallback to local AI
    return local_ai_answer(prompt, df)

# -----------------------------
# INSIGHTS AND ANALYSIS
# -----------------------------
def generate_insight(df, column):
    summary = (
        f"Column '{column}' - Mean: {df[column].mean():.2f}, "
        f"Max: {df[column].max()}, Min: {df[column].min()}."
    )
    return query_ai(f"Give a short expert insight on these stats: {summary}", df=df)

def medical_analysis(df):
    cols = list(df.columns)
    prompt = (
        f"Identify potential health-related risks or trends in these data headers: {cols}. "
        "Disclaimer: This is not medical advice."
    )
    return query_ai(prompt, df=df)

# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------
def predict_future(df, column, steps=5):
    y = df[column].values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
    predictions = model.predict(future_X)

    return predictions

def detect_anomalies(df, column):
    data = df[column]
    mean = data.mean()
    std = data.std()

    anomalies = df[(data > mean + 2*std) | (data < mean - 2*std)]
    return anomalies

# -----------------------------
# PDF REPORT GENERATOR
# -----------------------------
def generate_pdf(filename, insights, preds, anomalies):
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
