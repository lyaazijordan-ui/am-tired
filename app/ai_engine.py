# ai_engine.py (FINAL GOD MODE)

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
# 🧠 LOCAL AI (GOD MODE)
# -----------------------------
def local_ai(prompt, df=None):
    prompt = prompt.lower()

    if df is None:
        return "Upload a dataset first."

    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(exclude=['number']).columns.tolist()

    try:
        # INTENT DETECTION
        if any(x in prompt for x in ["summary", "overview", "describe"]):
            return f"📊 Summary:\n{df.describe().to_string()}"

        if any(x in prompt for x in ["mean", "average"]):
            return "📊 Averages:\n" + df[numeric].mean().to_string()

        if "max" in prompt:
            return "📈 Max values:\n" + df[numeric].max().to_string()

        if "min" in prompt:
            return "📉 Min values:\n" + df[numeric].min().to_string()

        if any(x in prompt for x in ["correlation", "relationship"]):
            corr = df[numeric].corr()
            return f"🔗 Correlation:\n{corr.to_string()}"

        if "trend" in prompt:
            result = []
            for col in numeric:
                if df[col].iloc[-1] > df[col].iloc[0]:
                    result.append(f"{col}: Increasing 📈")
                else:
                    result.append(f"{col}: Decreasing 📉")
            return "\n".join(result)

        if any(x in prompt for x in ["outlier", "anomaly"]):
            report = []
            for col in numeric:
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[(df[col] > mean + 2*std) | (df[col] < mean - 2*std)]
                report.append(f"{col}: {len(outliers)} anomalies")
            return "\n".join(report)

        if "important" in prompt:
            variances = df[numeric].var()
            return f"⭐ Most important column: {variances.idxmax()}"

        return (
            "🤖 I didn’t fully get that.\n\nTry:\n"
            "- summary\n- trends\n- correlation\n- anomalies"
        )

    except Exception as e:
        return f"Local AI Error: {str(e)}"


# -----------------------------
# 🌐 HYBRID AI (API + LOCAL)
# -----------------------------
def query_ai(prompt, context="", df=None):
    if not API_KEY:
        return local_ai(prompt, df)

    full_prompt = f"{context}\n\nUser Question: {prompt}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    for model in FALLBACK_MODELS:
        try:
            res = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a smart data analyst."},
                        {"role": "user", "content": full_prompt}
                    ]
                },
                timeout=15
            )

            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]

        except:
            continue

    return local_ai(prompt, df)


# -----------------------------
# 📊 AUTO GRAPH ENGINE
# -----------------------------
def auto_graph(df):
    import plotly.express as px
    import plotly.figure_factory as ff

    charts = []
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(exclude=['number']).columns.tolist()

    try:
        # Line (time)
        for col in df.columns:
            if "date" in col.lower():
                charts.append(px.line(df, x=col, y=numeric[0]))

        # Scatter
        if len(numeric) >= 2:
            charts.append(px.scatter(df, x=numeric[0], y=numeric[1]))

        # Histogram
        if numeric:
            charts.append(px.histogram(df, x=numeric[0]))

        # Box
        if numeric:
            charts.append(px.box(df, y=numeric[0]))

        # Bar
        if categorical:
            counts = df[categorical[0]].value_counts().reset_index()
            counts.columns = ["Category", "Count"]
            charts.append(px.bar(counts, x="Category", y="Count"))

        # Heatmap FIXED
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
            charts.append(fig)

    except Exception as e:
        print("Graph error:", e)

    return charts

def suggest_graph(df):
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(exclude=['number']).columns.tolist()

    suggestions = []

    try:
        # TIME SERIES
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                if numeric:
                    suggestions.append(
                        f"📈 Line Chart → Best for trends over time using '{col}'"
                    )

        # RELATIONSHIPS
        if len(numeric) >= 2:
            suggestions.append(
                f"🔗 Scatter Plot → Shows relationship between '{numeric[0]}' and '{numeric[1]}'"
            )

        # DISTRIBUTION
        if numeric:
            suggestions.append(
                f"📊 Histogram → Understand distribution of '{numeric[0]}'"
            )

        # OUTLIERS
        if numeric:
            suggestions.append(
                f"📦 Box Plot → Detect outliers in '{numeric[0]}'"
            )

        # CATEGORY
        if categorical:
            suggestions.append(
                f"📊 Bar Chart → Compare categories in '{categorical[0]}'"
            )

        # CORRELATION
        if len(numeric) >= 2:
            suggestions.append(
                "🔥 Heatmap → Visualize correlations between numeric features"
            )

        if suggestions:
            return "🧠 AI Graph Suggestions:\n\n" + "\n".join(suggestions)

        return "No strong visualization suggestion found."

    except Exception as e:
        return f"Suggestion error: {str(e)}"


# -----------------------------
# 🧠 AUTO EXPLANATION
# -----------------------------
def explain_data(df):
    summary = df.describe().to_string()
    return query_ai("Explain key insights from this data:\n" + summary, df=df)


# -----------------------------
# 📈 PREDICTIONS
# -----------------------------
def predict_future(df, column, steps=5):
    y = df[column].values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future = np.arange(len(y), len(y)+steps).reshape(-1, 1)
    return model.predict(future)


def detect_anomalies(df, column):
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] > mean + 2*std) | (df[column] < mean - 2*std)]


# -----------------------------
# 📄 PDF
# -----------------------------
def generate_pdf(filename, text):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Report", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, str(text))

    pdf.output(filename)
