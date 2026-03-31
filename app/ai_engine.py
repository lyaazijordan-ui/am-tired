# ai_engine.py
import os
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fpdf import FPDF

# -----------------------------
# API CONFIG
# -----------------------------
API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free"
]

# -----------------------------
# LOCAL AI (NO INTERNET)
# -----------------------------
def local_ai(prompt, df=None):
    prompt = prompt.lower()

    if df is not None:
        numeric_cols = df.select_dtypes(include=['number']).columns

        if "mean" in prompt:
            return df[numeric_cols].mean().to_string()

        if "max" in prompt:
            return df[numeric_cols].max().to_string()

        if "min" in prompt:
            return df[numeric_cols].min().to_string()

        if "summary" in prompt:
            return df.describe().to_string()

        if "correlation" in prompt:
            return df.corr(numeric_only=True).to_string()

        if "columns" in prompt:
            return str(list(df.columns))

        if "rows" in prompt:
            return f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"

    return "Local AI: I can summarize data, calculate stats, and detect trends."

# -----------------------------
# MAIN AI FUNCTION
# -----------------------------
def query_ai(prompt, data_context="", df=None):
    if not API_KEY:
        return local_ai(prompt, df)

    full_prompt = f"Context: {data_context}\n\nQuestion: {prompt}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    for model in FALLBACK_MODELS:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a professional data analyst."},
                    {"role": "user", "content": full_prompt}
                ]
            }

            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=15)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except:
            continue

    # fallback if API fails
    return local_ai(prompt, df)

# -----------------------------
# SMART GRAPH GENERATOR
# -----------------------------
def auto_graph(df):
    import plotly.express as px

    numeric = df.select_dtypes(include=['number']).columns
    categorical = df.select_dtypes(exclude=['number']).columns

    if len(numeric) >= 2:
        return px.scatter(df, x=numeric[0], y=numeric[1], title="Auto Scatter Plot")

    if len(numeric) == 1:
        return px.histogram(df, x=numeric[0], title="Distribution")

    if len(categorical) >= 1:
        return px.bar(df[categorical[0]].value_counts().reset_index(),
                      x='index', y=categorical[0],
                      title="Category Distribution")

    return None

# -----------------------------
# PREDICTIONS
# -----------------------------
def predict_future(df, column, steps=5):
    y = df[column].values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(y), len(y)+steps).reshape(-1, 1)
    return model.predict(future_X)

def detect_anomalies(df, column):
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] > mean + 2*std) | (df[column] < mean - 2*std)]

# -----------------------------
# PDF
# -----------------------------
def generate_pdf(filename, insights):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Report", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, insights)

    pdf.output(filename)
