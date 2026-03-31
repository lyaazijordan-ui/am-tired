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
    import plotly.figure_factory as ff

    charts = []

    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(exclude=['number']).columns.tolist()

    # -----------------------------
    # 1. TIME SERIES (Line Chart)
    # -----------------------------
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            if len(numeric) >= 1:
                fig = px.line(df, x=col, y=numeric[0],
                              title=f"📈 Trend of {numeric[0]} over {col}")
                charts.append(fig)
                break

    # -----------------------------
    # 2. SCATTER (Relationships)
    # -----------------------------
    if len(numeric) >= 2:
        fig = px.scatter(df, x=numeric[0], y=numeric[1],
                         title=f"🔗 Relationship: {numeric[0]} vs {numeric[1]}")
        charts.append(fig)

    # -----------------------------
    # 3. HISTOGRAM (Distribution)
    # -----------------------------
    if len(numeric) >= 1:
        fig = px.histogram(df, x=numeric[0],
                           title=f"📊 Distribution of {numeric[0]}")
        charts.append(fig)

    # -----------------------------
    # 4. BOXPLOT (Outliers)
    # -----------------------------
    if len(numeric) >= 1:
        fig = px.box(df, y=numeric[0],
                     title=f"📦 Outliers in {numeric[0]}")
        charts.append(fig)

    # -----------------------------
    # 5. BAR (Categorical)
    # -----------------------------
    if len(categorical) >= 1:
        counts = df[categorical[0]].value_counts().reset_index()
        counts.columns = ["Category", "Count"]

        fig = px.bar(counts, x="Category", y="Count",
                     title=f"📊 Distribution of {categorical[0]}")
        charts.append(fig)

    # -----------------------------
    # 6. HEATMAP (Correlation)
    # -----------------------------
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=round(corr.values, 2),
            showscale=True
        )
        fig.update_layout(title="🔥 Correlation Heatmap")
        charts.append(fig)

    return charts
    
def explain_data(df):
    numeric = df.select_dtypes(include=['number']).columns.tolist()

    summary = df.describe().to_string()

    prompt = f"""
    You are a data analyst.

    Explain the dataset in simple terms:
    - Key trends
    - Any unusual patterns
    - Important insights

    Data Summary:
    {summary}
    """

    return query_ai(prompt, df=df)
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
