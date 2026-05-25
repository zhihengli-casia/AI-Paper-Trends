<div align="right"><strong>English</strong> | <a href="README_cn.md">中文</a></div>

# AI Paper Trends

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## AI Paper Database Index

**Start here: [AI Paper Topic Atlas](docs/topic-atlas/README.md)**

Browse the paper database as **year -> venue -> topic -> paper**. The current index covers **117,100 accepted/main-track papers**, **84 venue-year groups**, and **5,183 fine-grained topics**.

Quick links:

| Entry | Link |
|---|---|
| Full atlas | [docs/topic-atlas/README.md](docs/topic-atlas/README.md) |
| 2020-2026 conference topic snapshot | [docs/clean-2020-plus/README.md](docs/clean-2020-plus/README.md) |
| 2026 papers | [docs/topic-atlas/2026/README.md](docs/topic-atlas/2026/README.md) |
| ICLR 2026 topics | [docs/topic-atlas/2026/ICLR/README.md](docs/topic-atlas/2026/ICLR/README.md) |
| Example topic | [ICLR 2026: 可验证奖励驱动的大模型推理](docs/topic-atlas/2026/ICLR/topic-004.md) |
| Topic CSV index | [docs/topic-atlas/data/topic_index.csv](docs/topic-atlas/data/topic_index.csv) |
| Venue-year CSV summary | [docs/topic-atlas/data/venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) |

Example path:

`2026 -> ICLR -> 可验证奖励驱动的大模型推理 -> Reinforcement Learning with Verifiable Rewards...`

This repository provides a browsable AI top-conference paper database and fine-grained topic atlas. It also includes the original single-conference OpenReview pipeline for focused conference-level experiments.

## 2020-2026 Conference Topic Snapshot

The [2020-2026 conference topic snapshot](docs/clean-2020-plus/README.md) summarizes topic distributions for conference papers under a conference/main accepted or published scope. It provides aggregate tables by year, venue, venue-year, and topic.

Snapshot coverage:

| Metric | Value |
|---|---:|
| Clustered conference papers | 142,799 |
| Conferences | 25 |
| Venue-year units | 132 |
| Topics across venue-year units | 1,307 |
| Years | 2020-2026 |
| Final outliers | 0 |

The CSV tables include full topic summaries by venue-year, latest-year topic lists by conference, yearly coverage, and venue coverage.

## What Is Included

| Area | Files |
|---|---|
| Browsable database | [docs/topic-atlas/](docs/topic-atlas/README.md) |
| 2020+ conference topic snapshot | [docs/clean-2020-plus/](docs/clean-2020-plus/README.md) |
| Topic indexes | [topic_index.csv](docs/topic-atlas/data/topic_index.csv), [venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) |
| Atlas generation | [scripts/build_topic_atlas.py](scripts/build_topic_atlas.py) |
| Fine-grained clustering | [scripts/fine_grained_topic_analysis.py](scripts/fine_grained_topic_analysis.py) |
| Automatic updates | [configs/auto_update.yaml](configs/auto_update.yaml), [scripts/auto_update_atlas.py](scripts/auto_update_atlas.py), [docs/auto-update/status.md](docs/auto-update/status.md) |
| Single-conference OpenReview pipeline | [main.py](main.py), [src/](src), [configs/](configs) |

## 🚀 Browse the Database

The easiest way to use this repository is to browse the static atlas:

- [Atlas home](docs/topic-atlas/README.md)
- [2026 venue list](docs/topic-atlas/2026/README.md)
- [ICLR 2026 topic list](docs/topic-atlas/2026/ICLR/README.md)
- [Example topic page](docs/topic-atlas/2026/ICLR/topic-004.md)

Each topic page contains:

- a Chinese display topic name,
- reproducible English keyword labels,
- representative papers,
- paper-level links such as OpenReview, DOI, Semantic Scholar, or source URLs when available.

The CSV indexes are useful if you want to build your own visualizations:

- [topic_index.csv](docs/topic-atlas/data/topic_index.csv)
- [venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv)

## 🔁 Rebuild the Topic Atlas

The atlas is generated from cached paper metadata and embeddings. Large embedding caches are stored outside the repository.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run fine-grained clustering for cached venue-year embeddings:

```bash
python scripts/fine_grained_topic_analysis.py \
  --input-root results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25 \
  --output-root results/fine_grained_venue_year_topics_2020_2026_mcs_fine
```

Generate the static atlas:

```bash
python scripts/build_topic_atlas.py \
  --topic-root results/fine_grained_venue_year_topics_2020_2026_mcs_fine \
  --output-root docs/topic-atlas \
  --clean
```

The current atlas was built using venue-year independent clustering:

- BGE embedding cache,
- UMAP dimensionality reduction,
- HDBSCAN leaf clustering,
- centroid reassignment for HDBSCAN outliers,
- c-TF-IDF style keyword extraction,
- heuristic Chinese topic naming for browsing.

## 🔄 Automatic Updates

The repository includes a GitHub Actions update engine:

- Weekly hosted check: detects venue-year volumes that are due or worth watching and updates [docs/auto-update/status.md](docs/auto-update/status.md).
- Full refresh: rebuilds `docs/topic-atlas` on a self-hosted runner labeled `ai-paper-trends`. This mode requires the external embedding/result cache. Set the repository variable `AUTO_REFRESH_ATLAS=true` to run this on the weekly schedule.

Run the lightweight check locally:

```bash
python scripts/auto_update_atlas.py check --write-report
```

Run a full refresh on a machine with the ignored `results/` cache:

```bash
python scripts/auto_update_atlas.py refresh
```

Tracked venue schedules and source notes live in [configs/auto_update.yaml](configs/auto_update.yaml).

## 🧪 Single-Conference Pipeline

The OpenReview + BERTopic workflow supports small single-conference experiments, including review-score plots and OpenReview-only analyses.

### 1. Environment Setup

```bash
conda create --name ai-trend-analysis python=3.10
conda activate ai-trend-analysis
pip install -r requirements.txt
```

### 2. Run an OpenReview Conference Task

Configure a task in `configs/`, then run:

```bash
python main.py --config configs/iclr_2025_full_analysis.yaml
```

Use this path when you want review-score plots or a quick OpenReview-only experiment. Use the atlas scripts above for the multi-conference database.

## 🤝 Contributing

Contributions are welcome. Please feel free to report issues, suggest features, or submit code contributions via [Issues](https://github.com/zhihengli-casia/AI-Paper-Trends/issues) or [Pull Requests](https://github.com/zhihengli-casia/AI-Paper-Trends/pulls).

## 📄 License

This project is released under the [MIT License](LICENSE).
