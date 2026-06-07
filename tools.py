import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Type
from pydantic import BaseModel, Field

# ── Shared state ──────────────────────────────────────────────────────────────
_df: Optional[pd.DataFrame] = None
_csv_path: Optional[str] = None


def set_dataframe(df: pd.DataFrame, path: str):
    global _df, _csv_path
    _df = df
    _csv_path = path


def get_dataframe() -> Optional[pd.DataFrame]:
    return _df


def load_file(file_path: str) -> pd.DataFrame:
    ext = file_path.lower().split(".")[-1]
    if ext == "csv":
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV file.")
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


# ── Simple tool classes (no CrewAI dependency) ────────────────────────────────

class CSVSummaryTool:
    def _run(self, dummy: str = "run") -> str:
        df = get_dataframe()
        if df is None:
            return "ERROR: No CSV loaded."
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        summary = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().mean() * 100).round(2).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "descriptive_stats": df.describe(include="all").fillna("N/A").to_dict(),
            "first_5_rows": df.head(5).fillna("N/A").to_dict(orient="records"),
        }
        return json.dumps(summary, default=str, indent=2)


class CorrelationTool:
    def _run(self, dummy: str = "run") -> str:
        df = get_dataframe()
        if df is None:
            return "ERROR: No CSV loaded."
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return "No numeric columns found."
        return numeric_df.corr().round(3).to_json()


class TopValuesTool:
    def _run(self, column: str, n: int = 10) -> str:
        df = get_dataframe()
        if df is None:
            return "ERROR: No CSV loaded."
        if column not in df.columns:
            return f"ERROR: Column '{column}' not found."
        return df[column].value_counts().head(n).to_json()


class ChartGeneratorTool:
    def _run(
        self,
        chart_type: str,
        title: str,
        filename: str,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
    ) -> str:
        df = get_dataframe()
        if df is None:
            return "ERROR: No CSV loaded."

        os.makedirs("output", exist_ok=True)
        out_path = f"output/{filename}.png"

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("mako", 10)

        try:
            chart_type = chart_type.lower().strip()

            if chart_type == "histogram":
                col = x_column or df.select_dtypes(include="number").columns[0]
                ax.hist(df[col].dropna(), bins=30, color=palette[3], edgecolor="white")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")

            elif chart_type == "bar":
                if x_column and y_column and x_column in df.columns and y_column in df.columns:
                    grouped = df.groupby(x_column)[y_column].mean().sort_values(ascending=False).head(15)
                    grouped.plot(kind="bar", ax=ax, color=palette[2], edgecolor="white")
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(f"Mean {y_column}")
                    plt.xticks(rotation=45, ha="right")
                elif x_column and x_column in df.columns:
                    df[x_column].value_counts().head(15).plot(kind="bar", ax=ax, color=palette[2])
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45, ha="right")
                else:
                    col = df.select_dtypes(include=["object", "category"]).columns[0]
                    df[col].value_counts().head(15).plot(kind="bar", ax=ax, color=palette[2])
                    plt.xticks(rotation=45, ha="right")

            elif chart_type == "line":
                if x_column and y_column and x_column in df.columns and y_column in df.columns:
                    df_sorted = df.sort_values(x_column)
                    ax.plot(df_sorted[x_column], df_sorted[y_column], color=palette[4], linewidth=2)
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                else:
                    num_cols = df.select_dtypes(include="number").columns[:4]
                    df[num_cols].plot(ax=ax, linewidth=2)

            elif chart_type == "scatter":
                num_cols = df.select_dtypes(include="number").columns
                xc = x_column if x_column and x_column in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
                yc = y_column if y_column and y_column in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
                if xc and yc:
                    ax.scatter(df[xc], df[yc], alpha=0.6, color=palette[5], edgecolors="white", s=60)
                    ax.set_xlabel(xc)
                    ax.set_ylabel(yc)

            elif chart_type == "heatmap":
                numeric_df = df.select_dtypes(include="number")
                corr = numeric_df.corr()
                sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
                            cmap="mako", center=0, square=True, linewidths=0.5)

            elif chart_type == "boxplot":
                num_cols = df.select_dtypes(include="number").columns.tolist()
                cols = [x_column] if x_column and x_column in num_cols else num_cols[:6]
                df[cols].plot(kind="box", ax=ax, patch_artist=True)
                plt.xticks(rotation=30, ha="right")

            elif chart_type == "pie":
                col = x_column if x_column and x_column in df.columns else df.select_dtypes(include=["object", "category"]).columns[0]
                counts = df[col].value_counts().head(8)
                ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                       colors=palette[:len(counts)], startangle=140)
                ax.axis("equal")

            else:
                plt.close(fig)
                return f"ERROR: Unknown chart type '{chart_type}'"

            ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
            fig.tight_layout()
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return f"SUCCESS: Chart saved to '{out_path}'"

        except Exception as e:
            plt.close(fig)
            return f"ERROR creating chart: {str(e)}"


# ── Exported instances ────────────────────────────────────────────────────────
csv_summary_tool = CSVSummaryTool()
correlation_tool = CorrelationTool()
top_values_tool = TopValuesTool()
chart_generator_tool = ChartGeneratorTool()