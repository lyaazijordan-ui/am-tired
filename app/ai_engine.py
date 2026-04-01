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
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free"
]

# -----------------------------
# LOCAL AI (NO INTERNET)
# -----------------------------
def local_ai(prompt, df=None):
    prompt = prompt.lower()

    if df is not None:
        numeric_cols = df.select_dtypes(include=['number']).columns

        try:
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

        except Exception as e:
            return f"Local AI Error: {str(e)}"

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
                ],
                "max_tokens": 500
            }

            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"API error with {model}: {e}")
            continue

    return local_ai(prompt, df)

# -----------------------------
# AUTO GRAPH (FIXED)
# -----------------------------
def auto_graph(df):
    import plotly.express as px
    import plotly.figure_factory as ff

    charts = []

    try:
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        categorical = df.select_dtypes(exclude=['number']).columns.tolist()

        # LINE (time series)
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                if numeric:
                    fig = px.line(df, x=col, y=numeric[0],
                                  title=f"📈 Trend of {numeric[0]} over {col}")
                    charts.append(fig)
                    break

        # SCATTER
        if len(numeric) >= 2:
            charts.append(px.scatter(df, x=numeric[0], y=numeric[1],
                                     title=f"🔗 {numeric[0]} vs {numeric[1]}"))

        # HISTOGRAM
        if numeric:
            charts.append(px.histogram(df, x=numeric[0],
                                       title=f"📊 Distribution of {numeric[0]}"))

        # BOXPLOT
        if numeric:
            charts.append(px.box(df, y=numeric[0],
                                 title=f"📦 Outliers in {numeric[0]}"))

        # BAR
        if categorical:
            counts = df[categorical[0]].value_counts().reset_index()
            counts.columns = ["Category", "Count"]
            charts.append(px.bar(counts, x="Category", y="Count",
                                 title=f"📊 {categorical[0]} Distribution"))

        # HEATMAP (🔥 FIXED PROPERLY)
        if len(numeric) >= 2:
            corr = df[numeric].corr()
            rounded = np.round(corr.values, 2)

            fig = ff.create_annotated_heatmap(
                z=rounded,
                x=list(corr.columns),
                y=list(corr.index),
                annotation_text=rounded.astype(str),
                showscale=True
            )

            fig.update_layout(title="🔥 Correlation Heatmap")
            charts.append(fig)

    except Exception as e:
        print("Auto graph error:", e)

    return charts

# -----------------------------
# AI EXPLANATION (FIXED)
# -----------------------------
def explain_data(df):
    try:
        summary = df.describe().to_string()

        prompt = f"""
        Explain this dataset in simple terms:
        - Key trends
        - Patterns
        - Insights

        Data:
        {summary}
        """

        return query_ai(prompt, df=df)

    except Exception as e:
        return f"Insight Error: {str(e)}"

# -----------------------------
# SIMPLE INSIGHT FUNCTION (FIXED)
# -----------------------------
def generate_insight(df, column):
    try:
        if column not in df.columns:
            return f"Column '{column}' not found."

        if not np.issubdtype(df[column].dtype, np.number):
            return f"Column '{column}' is not numeric."

        return f"""
        📊 Insights for '{column}':
        Mean: {df[column].mean():.2f}
        Max: {df[column].max()}
        Min: {df[column].min()}
        """

    except Exception as e:
        return f"Insight error: {str(e)}"

# -----------------------------
# PREDICTIONS
# -----------------------------
def predict_future(df, column, steps=5):
    try:
        y = df[column].values
        X = np.arange(len(y)).reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
        return model.predict(future_X)

    except Exception as e:
        return f"Prediction error: {str(e)}"

def detect_anomalies(df, column):
    try:
        mean = df[column].mean()
        std = df[column].std()
        return df[(df[column] > mean + 2*std) | (df[column] < mean - 2*std)]

    except Exception as e:
        return f"Anomaly error: {str(e)}"

# -----------------------------
# PDF
# -----------------------------
def generate_pdf(filename, insights):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Report", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, str(insights))

    pdf.output(filename)
