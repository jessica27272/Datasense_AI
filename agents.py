import os
from crewai import Agent, LLM
from tools import csv_summary_tool, correlation_tool, chart_generator_tool, top_values_tool


def get_llm():
    return LLM(
        model="cerebras/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        temperature=0.3,
    )


def create_analyst_agent() -> Agent:
    return Agent(
        role="Senior Data Analyst",
        goal=(
            "Perform a thorough exploratory data analysis on the uploaded dataset. "
            "Identify key trends, distributions, correlations, outliers, and business insights. "
            "Produce a well-structured markdown report with sections: "
            "Dataset Overview, Key Statistics, Trends & Patterns, Correlations, Anomalies, and Recommendations."
        ),
        backstory=(
            "You are a world-class senior data analyst with 15 years of experience at top-tier "
            "consultancies. You have an exceptional eye for patterns in raw data and a talent "
            "for explaining complex findings in clear, actionable language that non-technical "
            "stakeholders can understand. You are meticulous, evidence-driven, and always "
            "ground your insights in the actual numbers."
        ),
        tools=[csv_summary_tool, correlation_tool, top_values_tool],
        llm=get_llm(),
        verbose=True,
        max_iter=8,
        allow_delegation=False,
    )


def create_visualizer_agent() -> Agent:
    return Agent(
        role="Data Visualization Specialist",
        goal=(
            "Based on the analysis report, create 3-4 professional, insightful charts "
            "that best communicate the key findings. Choose chart types strategically: "
            "use histograms for distributions, bar charts for comparisons, heatmaps for "
            "correlations, line charts for trends, and scatter plots for relationships. "
            "Return a JSON list of created chart paths and their descriptions."
        ),
        backstory=(
            "You are an expert data visualization engineer who has designed dashboards "
            "for Fortune 500 companies. You believe that the right chart can make data "
            "speak for itself. You always choose the most appropriate visualization for "
            "each insight, ensuring charts are clear, professional, and immediately "
            "understandable to any audience."
        ),
        tools=[csv_summary_tool, chart_generator_tool],
        llm=get_llm(),
        verbose=True,
        max_iter=10,
        allow_delegation=False,
    )