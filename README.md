<div align="right"><strong>English</strong> | <a href="README_cn.md">中文</a></div>

# AI Paper Trends

Tools for crawling public AI conference paper metadata, normalizing main-conference accepted papers, clustering research topics, and tracking topic trends over time.

The project started as an ICLR 2025 OpenReview analysis pipeline. It has now been expanded into a multi-conference trend-analysis workflow that clusters papers independently for each venue-year, so topic distributions are comparable across conferences and years.

This is not just a paper list. The goal is to provide a reproducible **topic composition atlas** for major AI conferences: what each venue-year is made of, and how those research themes evolve over time.

## Current Analysis

The latest run covers main-conference accepted papers from 2020 to 2026:

- 15 venues
- 84 venue-year groups
- 117,100 papers used for clustering
- 763 venue-year topics
- 0 final outliers after graph-community clustering and small-community merging

Method:

`BGE embeddings -> cosine kNN graph -> Louvain community detection -> small-community merge -> deterministic Chinese topic naming`

Scope:

- Accepted main-conference papers only.
- NLP venues exclude Findings, Industry, SRW, and other non-main tracks.
- 2026 currently includes legally confirmed `ICLR 2026 accepted` and `AAAI 2026 Technical Tracks accepted`.

Lightweight result artifacts are committed under:

- Full Chinese report: [docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/REPORT_CN.md](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/REPORT_CN.md)
- Top 10 topics per venue-year: [top10_topics_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/top10_topics_by_venue_year.csv)
- Full topic summary: [topic_summary_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/topic_summary_by_venue_year.csv)
- Run summary: [run_summary_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/run_summary_by_venue_year.csv)
- Label trend table: [label_trend_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/label_trend_by_venue_year.csv)

Large raw crawls, per-paper topic assignments, embedding files, and model caches are intentionally not committed.

## Topic Composition Atlas

The repository includes lightweight composition visuals for reading and sharing:

- Atlas cover: [docs/visuals/xhs_composition_atlas_2020_2026/00_composition_atlas_cover.png](docs/visuals/xhs_composition_atlas_2020_2026/00_composition_atlas_cover.png)
- Year-by-year venue composition: [docs/visuals/xhs_composition_atlas_2020_2026/by_year/](docs/visuals/xhs_composition_atlas_2020_2026/by_year/)
- Venue-by-venue yearly composition: [docs/visuals/xhs_composition_atlas_2020_2026/by_venue/](docs/visuals/xhs_composition_atlas_2020_2026/by_venue/)
- Venue-year family composition table: [venue_year_family_composition.csv](docs/visuals/xhs_composition_atlas_2020_2026/venue_year_family_composition.csv)
- Full venue-year topic composition table: [venue_year_topic_composition_full.csv](docs/visuals/xhs_composition_atlas_2020_2026/venue_year_topic_composition_full.csv)

Each horizontal composition bar represents all accepted papers from one venue-year. Colors encode broad topic families and segment lengths encode shares. The right-side labels show the leading fine-grained topics; the full fine-grained composition is available in CSV.

## Data Scale

| Year | Venue-Year Groups | Papers | Topics |
|---:|---:|---:|---:|
| 2020 | 13 | 11,555 | 106 |
| 2021 | 14 | 13,334 | 117 |
| 2022 | 14 | 14,333 | 117 |
| 2023 | 13 | 17,424 | 120 |
| 2024 | 14 | 22,334 | 132 |
| 2025 | 14 | 28,619 | 144 |
| 2026 | 2 | 9,501 | 27 |

## Venues

| Area | Venues |
|---|---|
| Computer Vision | CVPR, ICCV, ECCV |
| Machine Learning | ICLR, ICML, NeurIPS |
| NLP | ACL, EMNLP, NAACL |
| General AI | AAAI, IJCAI |
| Multimedia | ACM MM / ACMMM |
| Data Mining / IR / Web | KDD, SIGIR, WWW |

## Recent Topic Examples

### ICLR 2026

| Rank | Topic | Papers | Share |
|---:|---|---:|---:|
| 1 | Video diffusion generation and editing | 752 | 14.05% |
| 2 | Efficient LLM inference, compression, and resource optimization | 691 | 12.91% |
| 3 | RL-driven LLM reasoning and reward learning | 655 | 12.24% |
| 4 | Stochastic/non-convex optimization and convergence | 645 | 12.05% |
| 5 | Multimodal/VLM understanding and reasoning | 625 | 11.68% |

### AAAI 2026

| Rank | Topic | Papers | Share |
|---:|---|---:|---:|
| 1 | 3D vision: Gaussian Splatting and view synthesis | 610 | 14.70% |
| 2 | LLM reasoning: QA, commonsense, and chain-of-thought | 596 | 14.36% |
| 3 | Reinforcement learning: policy optimization and reward learning | 463 | 11.16% |
| 4 | Graph anomaly detection, clustering, and structural representation | 391 | 9.42% |
| 5 | Efficient LLM inference, compression, and resource optimization | 344 | 8.29% |

### NeurIPS 2025

| Rank | Topic | Papers | Share |
|---:|---|---:|---:|
| 1 | RL-driven LLM reasoning and reward learning | 781 | 14.77% |
| 2 | Video diffusion generation and editing | 583 | 11.03% |
| 3 | Multimodal/VLM understanding and reasoning | 575 | 10.88% |
| 4 | Bandits, regret bounds, and online decision making | 570 | 10.78% |
| 5 | Stochastic/non-convex optimization and convergence | 548 | 10.37% |

### CVPR 2025

| Rank | Topic | Papers | Share |
|---:|---|---:|---:|
| 1 | Text-to-image diffusion, sampling, and image editing | 458 | 15.95% |
| 2 | 3D vision: point clouds, depth estimation, and camera pose | 413 | 14.39% |
| 3 | Multimodal/VLM understanding and cross-modal reasoning | 349 | 12.16% |
| 4 | Video diffusion generation and editing | 276 | 9.61% |
| 5 | Embodied AI: robot manipulation, navigation, and VLA | 253 | 8.81% |

## Quick Start

```bash
git clone https://github.com/zhihengli-casia/AI-Paper-Trends.git
cd AI-Paper-Trends
conda create -n ai-paper-trends python=3.10
conda activate ai-paper-trends
pip install -r requirements.txt
```

### Legacy Single-Venue OpenReview Analysis

The original ICLR/OpenReview workflow is still available for analyzing submissions, reviews, decisions, and reviewer scores where OpenReview exposes them.

```bash
python main.py --config configs/iclr_2025_full_analysis.yaml
```

### Multi-Conference Accepted-Paper Trend Analysis

Run the large workflow in `tmux` or another long-running session.

```bash
# 1. Crawl public metadata.
python src/crawl_recent_conferences.py \
  --output-dir data/recent_conferences \
  --run-name recent_ai_conference_papers \
  --fetch-detail-abstracts

# 2. Build the strict main-conference accepted view.
python src/build_main_accepted_view.py \
  --input data/recent_conferences/recent_ai_conference_papers.jsonl \
  --output data/recent_conferences/main_accepted_ai_conference_papers.jsonl

# 3. Cluster each venue-year independently with kNN + Louvain.
python src/analyze_venue_year_louvain_topics.py \
  --input data/recent_conferences/main_accepted_ai_conference_papers.jsonl \
  --output-dir results/venue_year_main_accepted_topics_louvain \
  --model-name models/bge-base-en-v1.5 \
  --device mps
```

If only naming rules or aggregate tables changed, rebuild outputs without rerunning embeddings/clustering:

```bash
python src/rebuild_venue_year_outputs.py \
  --results-dir results/venue_year_main_accepted_topics_louvain \
  --top-k 10
```

Generate the composition atlas:

```bash
python src/create_xhs_composition_atlas.py \
  --results-dir docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25 \
  --output-dir docs/visuals/xhs_composition_atlas_2020_2026
```

## Project Structure

```text
.
├── configs/                     # Legacy single-venue OpenReview configs
├── data/                        # Local crawls and intermediate files; gitignored
├── docs/results/                # Lightweight committed analysis artifacts
├── docs/visuals/                # Topic-composition atlas visuals
├── main.py                      # Legacy OpenReview entry point
├── notebooks/                   # Notebook examples
├── src/
│   ├── crawl_recent_conferences.py
│   ├── crawl_2020_2025_backfill.py
│   ├── build_main_accepted_view.py
│   ├── analyze_venue_year_louvain_topics.py
│   ├── create_xhs_composition_atlas.py
│   ├── refine_topic_names.py
│   ├── rebuild_venue_year_outputs.py
│   └── ...
└── requirements.txt
```

## Notes

- The multi-conference trend analysis uses public paper metadata only.
- Review scores and rejected submissions are not uniformly public across venues.
- OpenReview-based single-conference analysis can include submissions/rejections/reviews when available, but those fields should not be mixed directly with accepted-only proceedings data from other venues.

## License

MIT License.
