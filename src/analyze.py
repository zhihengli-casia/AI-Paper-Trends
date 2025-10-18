import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

def create_analysis_dataframe(df_with_topics: pd.DataFrame, topic_labels: dict) -> pd.DataFrame:
    """
    Processes a DataFrame with topic IDs, calculates statistics, and adds readable topic names.
    This serves as the basis for all subsequent analysis and visualization.
    """
    print("--- Preprocessing data and calculating statistics ---")
    
    df = df_with_topics[df_with_topics['Topic'] != -1].copy()
    print(f"Removed {len(df_with_topics) - len(df)} outlier papers. {len(df)} papers remain for analysis.")

    if df.empty:
        print("No data available for analysis.")
        return pd.DataFrame()

    def clean_decision(decision_str):
        decision_str = str(decision_str).lower()
        if 'oral' in decision_str: return 'Oral'
        if 'spotlight' in decision_str: return 'Spotlight'
        if 'poster' in decision_str: return 'Poster'
        if 'reject' in decision_str: return 'Reject'
        return 'N/A'
    df['clean_decision'] = df['decision'].apply(clean_decision)

    topic_stats = df.groupby('Topic').agg(paper_count=('id', 'count'), avg_rating=('avg_rating', 'mean')).reset_index()
    decision_counts = df.groupby(['Topic', 'clean_decision'])['id'].count().unstack(fill_value=0)
    analysis_df = pd.merge(topic_stats, decision_counts, on='Topic', how='left').fillna(0)

    decision_types = ['Oral', 'Spotlight', 'Poster', 'Reject', 'N/A']
    for dtype in decision_types:
        if dtype not in analysis_df.columns:
            analysis_df[dtype] = 0
            
    accepted_papers = analysis_df['Oral'] + analysis_df['Spotlight'] + analysis_df['Poster']
    total_papers_in_scope = accepted_papers + analysis_df['Reject'] + analysis_df['N/A']
    analysis_df['acceptance_rate'] = (accepted_papers / total_papers_in_scope).fillna(0)
    
    analysis_df['Topic_Name'] = analysis_df['Topic'].map(topic_labels)
    analysis_df['Topic_Name'].fillna(analysis_df['Topic'].apply(lambda x: f"Topic {x}"), inplace=True)
    
    final_columns = ['Topic', 'Topic_Name', 'paper_count', 'avg_rating', 'acceptance_rate', 'Oral', 'Spotlight', 'Poster', 'Reject', 'N/A']
    final_columns = [col for col in final_columns if col in analysis_df.columns]
    
    print("Data preprocessing and statistical analysis complete.")
    return analysis_df[final_columns].sort_values(by='paper_count', ascending=False).reset_index(drop=True)

def plot_topic_ranking(df: pd.DataFrame, metric: str, conference: str, output_path: Path, top_n: int = 65):
    """Generates a horizontal bar chart for the top N topics based on a specified metric."""
    print(f"--- Generating Top {top_n} topics ranking plot (by: {metric}) ---")
    
    top_df = df.sort_values(by=metric, ascending=False).head(top_n)
    
    HEIGHT_PER_ITEM, BASE_HEIGHT, MIN_HEIGHT = 0.4, 3.0, 8.0
    figure_height = max(MIN_HEIGHT, (len(top_df) * HEIGHT_PER_ITEM) + BASE_HEIGHT)
    plt.figure(figsize=(14, figure_height))
    
    palette = 'viridis' if metric == 'paper_count' else 'plasma'
    sns.barplot(x=metric, y='Topic_Name', data=top_df, orient='h', palette=palette)
    
    metric_title = metric.replace('_', ' ').title()
    plt.title(f'Top {top_n} Topics by {metric_title} at {conference}', fontsize=20)
    plt.xlabel(metric_title, fontsize=14)
    plt.ylabel('Topic Name', fontsize=14)
    
    if len(top_df) > 30: plt.yticks(fontsize=10)
    else: plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {output_path}")

def plot_decision_breakdown(df: pd.DataFrame, conference: str, output_path: Path, top_n: int = 65):
    """Generates a normalized stacked bar chart of decision breakdowns for the top N topics."""
    print(f"--- Generating Top {top_n} topics decision breakdown plot ---")
    
    top_df = df.sort_values(by='acceptance_rate', ascending=False).head(top_n)
    decision_cols = ['Oral', 'Spotlight', 'Poster', 'Reject', 'N/A']
    plot_data = top_df.set_index('Topic_Name')[decision_cols]
    plot_data_normalized = plot_data.div(plot_data.sum(axis=1), axis=0)
    
    HEIGHT_PER_ITEM, BASE_HEIGHT, MIN_HEIGHT = 0.5, 4.0, 10.0
    figure_height = max(MIN_HEIGHT, (len(plot_data_normalized) * HEIGHT_PER_ITEM) + BASE_HEIGHT)
    plt.rcParams['figure.figsize'] = (18, figure_height) 
    
    ax = plot_data_normalized.plot(kind='barh', stacked=True, colormap='viridis', width=0.8) 
    
    count_map = top_df.set_index('Topic_Name')['paper_count']
    for i, topic_name in enumerate(plot_data_normalized.index):
        total_papers = count_map[topic_name]
        ax.text(1.01, i, f'n={total_papers}', va='center', fontsize=10, color='black', fontweight='bold')
    
    plt.title(f'Top {top_n} Topics: Decision Breakdown at {conference}', fontsize=20, pad=40) 
    plt.xlabel('Proportion of Papers within Each Topic', fontsize=14)
    plt.ylabel('Topic Name (Sorted by Acceptance Rate)', fontsize=14)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(0, 1) 
    if len(plot_data_normalized) > 30: plt.yticks(fontsize=10) 
    else: plt.yticks(fontsize=12) 
    plt.gca().invert_yaxis() 
    plt.legend(title='Decision Type', loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=5, frameon=False, fontsize=12) 
    
    plt.tight_layout(rect=[0, 0, 0.92, 1]) 
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {output_path}")
    plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

def save_summary_table(df: pd.DataFrame, output_path: Path, top_n: int = 65):
    """Saves the final summary statistics table in HTML and CSV formats."""
    print(f"--- Generating Top {top_n} topics summary table ---")
    
    display_cols = ['Topic_Name', 'paper_count', 'avg_rating', 'acceptance_rate', 'Oral', 'Spotlight', 'Poster', 'Reject', 'N/A']
    final_table = df[display_cols].sort_values(by='acceptance_rate', ascending=False).head(top_n).copy()
    
    csv_path = output_path.with_suffix('.csv')
    final_table.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV table saved to: {csv_path}")
    
    html_path = output_path.with_suffix('.html')
    styler = final_table.style.format({
        'avg_rating': '{:.2f}', 'acceptance_rate': '{:.2%}',
        'Oral': '{:d}', 'Spotlight': '{:d}', 'Poster': '{:d}', 'Reject': '{:d}', 'N/A': '{:d}'
    }).bar(subset=['paper_count'], color='#5fba7d', align='left') \
      .bar(subset=['avg_rating'], color='#d65f5f', align='left', vmin=df['avg_rating'].min()) \
      .set_caption(f"Top {top_n} Topics Analysis Summary (Sorted by Acceptance Rate)")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(styler.to_html())
    print(f"HTML table saved to: {html_path}")

def main(config: Dict[str, Any], input_path: Path, output_dir: Path):
    """Main entry point for the analysis module, called by main.py."""
    sns.set_theme(style="whitegrid", context="talk")
    
    try:
        df_with_topics = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Analysis failed: Input file '{input_path}' not found.")
        return
        
    labels_path = output_dir / 'topic_labels.yaml'
    topic_labels = {}
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            topic_labels = yaml.safe_load(f)
        print(f"Successfully loaded topic names from '{labels_path}'.")
    except FileNotFoundError:
        print(f"Topic name file not found at: {labels_path}. Default names will be used.")

    analysis_df = create_analysis_dataframe(df_with_topics, topic_labels)
    if analysis_df.empty:
        print("Analysis failed: Could not generate valid analysis data.")
        return

    analysis_config = config.get('analysis', {})
    tasks = analysis_config.get('tasks', [])
    conference = config.get('conference_id', '').split('/')[0]
    
    if 'plot_paper_count' in tasks:
        plot_topic_ranking(analysis_df, 'paper_count', conference, output_dir / "top_topics_by_count.png")
        
    if 'plot_avg_rating' in tasks:
        plot_topic_ranking(analysis_df, 'avg_rating', conference, output_dir / "top_topics_by_rating.png")
        
    if 'plot_decision_breakdown' in tasks:
        plot_decision_breakdown(analysis_df, conference, output_dir / "decision_breakdown.png")
    
    if 'generate_summary_table' in tasks:
        save_summary_table(analysis_df, output_dir / "summary_table")