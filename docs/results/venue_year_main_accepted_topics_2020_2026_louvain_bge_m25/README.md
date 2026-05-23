# Venue-Year Topic Analysis, 2020-2026

This directory contains lightweight artifacts from the accepted-paper, venue-year topic analysis.

Included files:

- `REPORT_CN.md`: Chinese report with venue-year topic rankings and high-level trend notes.
- `run_summary_by_venue_year.csv`: paper counts, topic counts, and outlier statistics for each venue-year run.
- `top10_topics_by_venue_year.csv`: top 10 topics for every venue-year group.
- `topic_summary_by_venue_year.csv`: full topic-level summary table.
- `label_trend_by_venue_year.csv`: topic-label trend table across years and venues.

Large generated artifacts are intentionally excluded from Git:

- raw crawled metadata under `data/`
- per-paper topic assignments
- cached embeddings
- local model files under `models/`
