import json
import re
import os
import traceback
import requests
import pandas as pd
from dotenv import load_dotenv
from tools import set_dataframe, load_file, csv_summary_tool, correlation_tool, top_values_tool, chart_generator_tool

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL = "gpt-oss-120b"  # Cerebras ka correct model name

def call_cerebras(messages: list, max_tokens: int = 2048) -> str:
    """Direct Cerebras API call."""
    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    
    # Debug: print response if error
    if not resp.ok:
        print(f"Cerebras error {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def run_analyst(df: pd.DataFrame, file_path: str, user_question: str = "") -> str:
    summary = csv_summary_tool._run()
    correlation = correlation_tool._run()

    extra = f'\nAlso answer this specific question: "{user_question}"' if user_question.strip() else ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a world-class senior data analyst. "
                "Analyze the provided dataset summary and produce a detailed markdown EDA report. "
                "Always cite specific numbers, column names, and percentages. "
                "Structure your report with these exact sections:\n"
                "## 📊 Dataset Overview\n"
                "## 📈 Key Statistics & Distributions\n"
                "## 🔗 Correlations & Relationships\n"
                "## 🔍 Trends & Patterns\n"
                "## ⚠️ Anomalies & Data Quality Issues\n"
                "## 💡 Business Insights & Recommendations"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Here is the dataset summary:\n\n{summary}\n\n"
                f"Correlation matrix:\n{correlation}\n\n"
                f"Please write a comprehensive EDA report.{extra}"
            ),
        },
    ]
    return call_cerebras(messages, max_tokens=3000)


def run_visualizer(df: pd.DataFrame, analysis_report: str) -> list:
    summary = csv_summary_tool._run()
    summary_data = json.loads(summary)
    numeric_cols = summary_data.get("numeric_columns", [])
    categorical_cols = summary_data.get("categorical_columns", [])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data visualization expert. "
                "Based on the analysis report and available columns, decide which 3-4 charts to create. "
                "Respond ONLY with a valid JSON array. No explanation, no markdown, just raw JSON.\n"
                "Format:\n"
                '[\n'
                '  {"chart_type": "histogram", "x_column": "col_name", "y_column": null, '
                '"title": "Distribution of col_name", "filename": "col_name_distribution"},\n'
                '  {"chart_type": "heatmap", "x_column": null, "y_column": null, '
                '"title": "Correlation Heatmap", "filename": "correlation_heatmap"}\n'
                ']\n'
                "chart_type options: histogram, bar, line, scatter, heatmap, boxplot, pie\n"
                "Use null for columns not needed by that chart type."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis report:\n{analysis_report[:1500]}\n\n"
                f"Numeric columns: {numeric_cols}\n"
                f"Categorical columns: {categorical_cols}\n\n"
                "Return JSON array of 3-4 charts to create."
            ),
        },
    ]

    chart_plan_raw = call_cerebras(messages, max_tokens=800)

    chart_plan = []
    try:
        clean = re.sub(r'```json|```', '', chart_plan_raw).strip()
        chart_plan = json.loads(clean)
    except Exception:
        match = re.search(r'\[.*\]', chart_plan_raw, re.DOTALL)
        if match:
            try:
                chart_plan = json.loads(match.group())
            except Exception:
                chart_plan = []

    created_charts = []
    for chart in chart_plan:
        try:
            result = chart_generator_tool._run(
                chart_type=chart.get("chart_type", "histogram"),
                title=chart.get("title", "Chart"),
                filename=chart.get("filename", "chart"),
                x_column=chart.get("x_column"),
                y_column=chart.get("y_column"),
            )
            if "SUCCESS" in result:
                path = result.replace("SUCCESS: Chart saved to '", "").replace("'", "").strip()
                created_charts.append({
                    "path": path,
                    "title": chart.get("title", "Chart"),
                    "description": f"{chart.get('chart_type', '').title()} chart: {chart.get('title', '')}",
                })
        except Exception as e:
            print(f"Chart error: {e}")
            continue

    return created_charts


def run_analysis(
    file_path: str,
    user_question: str = "",
    progress_callback=None,
) -> dict:
    result = {"analysis_report": "", "charts": [], "error": None}

    try:
        if progress_callback:
            progress_callback("📂 Loading dataset...", 10)
        df = load_file(file_path)
        set_dataframe(df, file_path)

        if progress_callback:
            progress_callback("🧠 Analyst Agent reading your data...", 30)
        analysis_report = run_analyst(df, file_path, user_question)
        result["analysis_report"] = analysis_report

        if progress_callback:
            progress_callback("📊 Visualizer Agent creating charts...", 70)
        charts = run_visualizer(df, analysis_report)
        result["charts"] = charts

        if progress_callback:
            progress_callback("✅ Analysis complete!", 100)

    except Exception as e:
        result["error"] = f"{str(e)}\n\nFull traceback:\n{traceback.format_exc()}"

    return result