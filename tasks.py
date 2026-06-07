from crewai import Task
from agents import create_analyst_agent, create_visualizer_agent


def create_analysis_task(analyst: object, filename: str, user_question: str = "") -> Task:
    """
    Task 1: EDA — Exploratory Data Analysis.
    The analyst reads the CSV, computes stats, and writes a full markdown report.
    """
    extra_instruction = (
        f'\n\nUser also asked: "{user_question}"  — address this specifically in your report.'
        if user_question.strip()
        else ""
    )

    return Task(
        description=(
            f"You have been given a dataset file: '{filename}'.\n\n"
            "Follow these steps IN ORDER:\n"
            "1. Call csv_summary_tool to get the full dataset overview.\n"
            "2. Call correlation_tool to compute correlations between numeric columns.\n"
            "3. For each important categorical column (if any), call top_values_tool.\n"
            "4. Synthesize ALL findings into a detailed markdown report.\n\n"
            "Your report MUST include these sections:\n"
            "## 📊 Dataset Overview\n"
            "## 📈 Key Statistics & Distributions\n"
            "## 🔗 Correlations & Relationships\n"
            "## 🔍 Trends & Patterns\n"
            "## ⚠️ Anomalies & Data Quality Issues\n"
            "## 💡 Business Insights & Recommendations\n\n"
            "Be specific — cite actual numbers, column names, and percentages."
            + extra_instruction
        ),
        expected_output=(
            "A comprehensive markdown EDA report with 6 clearly labeled sections. "
            "Each section must contain specific findings with actual numbers and column names. "
            "End with 3-5 concrete, actionable business recommendations."
        ),
        agent=analyst,
    )


def create_visualization_task(visualizer: object, analyst_task: Task) -> Task:
    """
    Task 2: Chart Creation.
    The visualizer reads the analyst's report and creates 3-4 relevant charts.
    """
    return Task(
        description=(
            "You have just received a detailed data analysis report from the Senior Data Analyst.\n\n"
            "Your job is to create 3-4 professional charts that best visualize the KEY findings.\n\n"
            "RULES:\n"
            "- Always call csv_summary_tool FIRST to know the available columns.\n"
            "- Choose chart types strategically based on data types:\n"
            "  * Numeric distribution → histogram\n"
            "  * Category comparison → bar\n"
            "  * Two numeric variables → scatter\n"
            "  * Correlation matrix → heatmap\n"
            "  * Category proportions → pie\n"
            "  * Time series → line\n"
            "  * Spread & outliers → boxplot\n"
            "- Use descriptive filenames (e.g., 'revenue_by_region', 'age_distribution').\n"
            "- Each chart must have a clear, professional title.\n\n"
            "After creating all charts, return a JSON array like this:\n"
            '[\n'
            '  {"path": "output/chart1.png", "title": "...", "description": "..."},\n'
            '  {"path": "output/chart2.png", "title": "...", "description": "..."}\n'
            ']\n'
        ),
        expected_output=(
            "A valid JSON array (and nothing else after it) listing each created chart with: "
            "path (string), title (string), description (string explaining what the chart shows). "
            "Minimum 3 charts, maximum 5 charts."
        ),
        agent=visualizer,
        context=[analyst_task],
    )