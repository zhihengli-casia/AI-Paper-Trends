from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.utils import DECISION_TYPES


def create_analysis_dataframe(df_with_topics: pd.DataFrame, topic_labels: Dict[int, str]) -> pd.DataFrame:
    """
    Processes a DataFrame with topic IDs, calculates statistics, and adds readable topic names.
    This serves as the basis for all subsequent analysis and visualization.
    """
    print("--- Preprocessing data and calculating statistics ---")

    df = df_with_topics.copy()
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    if "decision" not in df.columns:
        df["decision"] = "N/A"
    if "avg_rating" not in df.columns:
        df["avg_rating"] = pd.NA

    df["Topic"] = pd.to_numeric(df["Topic"], errors="coerce").fillna(-1).astype(int)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")

    df = df[df["Topic"] != -1].copy()
    print(f"Removed {len(df_with_topics) - len(df)} outlier papers. {len(df)} papers remain for analysis.")

    if df.empty:
        print("No data available for analysis.")
        return pd.DataFrame()

    def clean_decision(decision_str: Any) -> str:
        normalized = str(decision_str).lower()
        if "oral" in normalized:
            return "Oral"
        if "spotlight" in normalized:
            return "Spotlight"
        if "poster" in normalized:
            return "Poster"
        if "reject" in normalized:
            return "Reject"
        return "N/A"

    df["clean_decision"] = df["decision"].apply(clean_decision)

    topic_stats = (
        df.groupby("Topic")
        .agg(paper_count=("id", "count"), avg_rating=("avg_rating", "mean"))
        .reset_index()
    )
    decision_counts = df.groupby(["Topic", "clean_decision"])["id"].count().unstack(fill_value=0)
    analysis_df = pd.merge(topic_stats, decision_counts, on="Topic", how="left")

    for decision_type in DECISION_TYPES:
        if decision_type not in analysis_df.columns:
            analysis_df[decision_type] = 0
        analysis_df[decision_type] = analysis_df[decision_type].fillna(0).astype(int)

    accepted_papers = analysis_df[["Oral", "Spotlight", "Poster"]].sum(axis=1)
    total_papers_in_scope = accepted_papers + analysis_df["Reject"] + analysis_df["N/A"]
    denominator = total_papers_in_scope.where(total_papers_in_scope > 0, other=1)
    analysis_df["acceptance_rate"] = accepted_papers / denominator

    analysis_df["Topic_Name"] = analysis_df["Topic"].map(topic_labels)
    analysis_df["Topic_Name"] = analysis_df["Topic_Name"].fillna(
        analysis_df["Topic"].map(lambda topic_id: f"Topic {topic_id}")
    )

    final_columns = [
        "Topic",
        "Topic_Name",
        "paper_count",
        "avg_rating",
        "acceptance_rate",
        "Oral",
        "Spotlight",
        "Poster",
        "Reject",
        "N/A",
    ]

    print("Data preprocessing and statistical analysis complete.")
    return (
        analysis_df[final_columns]
        .sort_values(by=["paper_count", "acceptance_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )


def plot_topic_ranking(
    df: pd.DataFrame,
    metric: str,
    conference: str,
    output_path: Path,
    top_n: int = 65,
) -> bool:
    """Generates a horizontal bar chart for the top N topics based on a specified metric."""
    top_df = df.sort_values(by=metric, ascending=False).head(top_n).dropna(subset=[metric])
    if top_df.empty:
        print(f"Skipping plot for '{metric}' because there is no usable data.")
        return False

    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"--- Generating Top {top_n} topics ranking plot (by: {metric}) ---")

    height_per_item, base_height, min_height = 0.4, 3.0, 8.0
    figure_height = max(min_height, (len(top_df) * height_per_item) + base_height)
    plt.figure(figsize=(14, figure_height))

    palette = "viridis" if metric == "paper_count" else "plasma"
    sns.barplot(
        x=metric,
        y="Topic_Name",
        data=top_df,
        orient="h",
        hue="Topic_Name",
        palette=palette,
        legend=False,
    )

    metric_title = metric.replace("_", " ").title()
    plt.title(f"Top {top_n} Topics by {metric_title} at {conference}", fontsize=20)
    plt.xlabel(metric_title, fontsize=14)
    plt.ylabel("Topic Name", fontsize=14)
    plt.yticks(fontsize=10 if len(top_df) > 30 else 12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to: {output_path}")
    return True


def plot_decision_breakdown(df: pd.DataFrame, conference: str, output_path: Path, top_n: int = 65) -> bool:
    """Generates a normalized stacked bar chart of decision breakdowns for the top N topics."""
    top_df = df.sort_values(by=["acceptance_rate", "paper_count"], ascending=[False, False]).head(top_n)
    if top_df.empty:
        print("Skipping decision breakdown plot because there is no usable data.")
        return False

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    print(f"--- Generating Top {top_n} topics decision breakdown plot ---")

    plot_data = top_df.set_index("Topic_Name")[DECISION_TYPES]
    row_totals = plot_data.sum(axis=1).replace(0, 1)
    plot_data_normalized = plot_data.div(row_totals, axis=0)

    height_per_item, base_height, min_height = 0.5, 4.0, 10.0
    figure_height = max(min_height, (len(plot_data_normalized) * height_per_item) + base_height)
    plt.rcParams["figure.figsize"] = (18, figure_height)

    ax = plot_data_normalized.plot(kind="barh", stacked=True, colormap="viridis", width=0.8)

    count_map = top_df.set_index("Topic_Name")["paper_count"]
    for index, topic_name in enumerate(plot_data_normalized.index):
        total_papers = count_map[topic_name]
        ax.text(1.01, index, f"n={total_papers}", va="center", fontsize=10, color="black", fontweight="bold")

    plt.title(f"Top {top_n} Topics: Decision Breakdown at {conference}", fontsize=20, pad=40)
    plt.xlabel("Proportion of Papers within Each Topic", fontsize=14)
    plt.ylabel("Topic Name (Sorted by Acceptance Rate)", fontsize=14)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(0, 1)
    plt.yticks(fontsize=10 if len(plot_data_normalized) > 30 else 12)
    plt.gca().invert_yaxis()
    plt.legend(
        title="Decision Type",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        frameon=False,
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to: {output_path}")
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    return True


def save_summary_table(df: pd.DataFrame, output_path: Path, top_n: int = 65) -> None:
    """Saves the final summary statistics table in HTML and CSV formats."""
    print(f"--- Generating Top {top_n} topics summary table ---")

    display_cols = [
        "Topic_Name",
        "paper_count",
        "avg_rating",
        "acceptance_rate",
        "Oral",
        "Spotlight",
        "Poster",
        "Reject",
        "N/A",
    ]
    final_table = (
        df[display_cols]
        .sort_values(by=["acceptance_rate", "paper_count"], ascending=[False, False])
        .head(top_n)
        .copy()
    )

    integer_columns = ["paper_count", *DECISION_TYPES]
    for column in integer_columns:
        final_table[column] = final_table[column].fillna(0).astype(int)

    csv_path = output_path.with_suffix(".csv")
    final_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV table saved to: {csv_path}")

    html_path = output_path.with_suffix(".html")
    formatters = {
        "avg_rating": lambda value: f"{value:.2f}" if pd.notna(value) else "N/A",
        "acceptance_rate": lambda value: f"{value:.2%}" if pd.notna(value) else "N/A",
        "paper_count": "{:d}",
        "Oral": "{:d}",
        "Spotlight": "{:d}",
        "Poster": "{:d}",
        "Reject": "{:d}",
        "N/A": "{:d}",
    }
    styler = final_table.style.format(formatters)
    paper_count_series = final_table["paper_count"]
    if paper_count_series.min() != paper_count_series.max():
        styler = styler.bar(subset=["paper_count"], color="#5fba7d", align="left")
    avg_rating_series = final_table["avg_rating"].dropna()
    if not avg_rating_series.empty and avg_rating_series.min() != avg_rating_series.max():
        styler = styler.bar(
            subset=["avg_rating"],
            color="#d65f5f",
            align="left",
            vmin=avg_rating_series.min(),
            vmax=avg_rating_series.max(),
        )
    styler = styler.set_caption(f"Top {top_n} Topics Analysis Summary (Sorted by Acceptance Rate)")

    with html_path.open("w", encoding="utf-8") as file:
        file.write(styler.to_html())
    print(f"HTML table saved to: {html_path}")


def main(config: Dict[str, Any], input_path: Path, output_dir: Path) -> None:
    """Main entry point for the analysis module, called by main.py."""
    try:
        df_with_topics = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Analysis failed: Input file '{input_path}' not found.")
        return

    if "Topic" not in df_with_topics.columns:
        print("Analysis failed: Input data does not include a 'Topic' column.")
        return

    labels_path = output_dir / "topic_labels.yaml"
    topic_labels: Dict[int, str] = {}
    try:
        with labels_path.open("r", encoding="utf-8") as file:
            loaded_labels = yaml.safe_load(file) or {}
        topic_labels = {int(topic_id): label for topic_id, label in loaded_labels.items()}
        print(f"Successfully loaded topic names from '{labels_path}'.")
    except FileNotFoundError:
        print(f"Topic name file not found at: {labels_path}. Default names will be used.")

    analysis_df = create_analysis_dataframe(df_with_topics, topic_labels)
    if analysis_df.empty:
        print("Analysis failed: Could not generate valid analysis data.")
        return

    analysis_config = config.get("analysis", {})
    tasks = analysis_config.get("tasks", [])
    top_n = analysis_config.get("top_n", 65)
    conference = config.get("conference_id", "").split("/")[0] or "Conference"

    if "plot_paper_count" in tasks:
        plot_topic_ranking(analysis_df, "paper_count", conference, output_dir / "top_topics_by_count.png", top_n)

    if "plot_avg_rating" in tasks:
        plot_topic_ranking(analysis_df, "avg_rating", conference, output_dir / "top_topics_by_rating.png", top_n)

    if "plot_decision_breakdown" in tasks:
        plot_decision_breakdown(analysis_df, conference, output_dir / "decision_breakdown.png", top_n)

    if "generate_summary_table" in tasks:
        save_summary_table(analysis_df, output_dir / "summary_table", top_n)
