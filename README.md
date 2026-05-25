<div align="right"><strong>English</strong> | <a href="README_cn.md">中文</a></div>

# AI Paper Trends

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![GitHub stars](https://img.shields.io/github/stars/zhihengli-casia/AI-Paper-Trends?style=social)](https://github.com/zhihengli-casia/AI-Paper-Trends/stargazers) [![Last commit](https://img.shields.io/github/last-commit/zhihengli-casia/AI-Paper-Trends)](https://github.com/zhihengli-casia/AI-Paper-Trends/commits/main)

AI Paper Trends is an open-source index of AI papers and research topics. It tracks major AI conferences and selected journals, then turns them into a static atlas you can browse as:

**year -> venue -> topic -> paper**

The project started from a simple need: see what each AI venue is publishing every year without digging through dozens of proceedings pages.

[![Browse the Atlas](https://img.shields.io/badge/Browse-Topic%20Atlas-0969da?style=for-the-badge)](docs/topic-atlas/README.md)
[![Download Topic Index](https://img.shields.io/badge/CSV-Topic%20Index-2da44e?style=for-the-badge)](docs/topic-atlas/data/topic_index.csv)

**155,662 papers · 7,378 fine-grained topics · 160 venue-year groups · 2020-2026 · conferences and journals across ML, CV, NLP, robotics, data mining, HCI, multimedia, and medical AI.**

## Current Snapshot

The checked-in static atlas currently contains:

| Metric | Value |
|---|---:|
| Papers | 155,662 |
| Venue-year groups | 160 |
| Fine-grained topic pages | 7,378 |
| Years | 2020-2026 |
| Unassigned papers after reassignment | 25 |

This is a living index. New venue-years are added as proceedings and public metadata become available, so the latest year may be partial during an active conference cycle.

Start browsing here: [AI Paper Topic Atlas](docs/topic-atlas/README.md)

## What You Can Do

- Browse papers by year, venue, fine-grained topic, and individual paper.
- Compare research directions across conferences, journals, and years.
- Use the CSV indexes for custom analysis and visualization.
- Rebuild the static atlas from cached paper metadata and topic outputs.
- Extend the tracked venue list through the update configuration.

## Coverage

Counts below describe the current repository snapshot. They are maintained as a continuously updated view and will change as more venues, years, and proceedings are added.

<details>
<summary><strong>Coverage by area</strong></summary>

### Conferences

| Area | Indexed venues | Papers | Fine topics |
|---|---|---:|---:|
| ML / learning theory | ICLR, ICML, NeurIPS | 46,161 | 1,724 |
| CV top conferences | CVPR, ICCV, ECCV | 24,999 | 1,130 |
| NLP / language | ACL, EMNLP, NAACL, COLM | 15,368 | 811 |
| General AI | AAAI, IJCAI | 21,179 | 963 |
| Embodied AI / robotics | ICRA, IROS, RSS | 16,590 | 873 |
| Multimedia / graphics / HCI | ACMMM, SIGGRAPH, SIGGRAPH-Asia, CHI | 10,473 | 591 |
| Data mining / IR / Web / DB | KDD, SIGIR, WWW, ICDE, SIGMOD | 7,601 | 482 |
| Medical AI | MICCAI | 428 | 41 |

### Journals

| Area | Indexed journals | Papers | Fine topics |
|---|---|---:|---:|
| ML / general AI journals | AIJ, JMLR, TNNLS | 2,768 | 183 |
| Vision / image journals | TPAMI, IJCV, TIP, PR | 8,009 | 456 |
| Multimedia / data journals | TMM, TKDE | 2,086 | 124 |

### Planned Additions

Candidate sources are tracked for future expansion. Inclusion depends on public metadata availability and source quality.

| Area | Candidate venues and journals |
|---|---|
| ML / AI | AISTATS, UAI, COLT, JAIR, Machine Learning |
| CV / graphics | WACV, BMVC, ACCV, 3DV, TVCG |
| NLP / speech | EACL, TACL, Computational Linguistics, Interspeech |
| Robotics / embodied AI | CoRL, RA-L, T-RO, IJRR, Autonomous Robots |
| Data / IR / Web | CIKM, WSDM, RecSys, ICDM, SDM, VLDB, EDBT, PODS |
| HCI / systems | UIST, CSCW, IMWUT, UbiComp |
| Medical AI | TMI, Medical Image Analysis, ISBI |

</details>

## Explore the Atlas

| Entry | Link |
|---|---|
| Atlas home | [docs/topic-atlas/README.md](docs/topic-atlas/README.md) |
| Latest indexed year | [docs/topic-atlas/2026/README.md](docs/topic-atlas/2026/README.md) |
| Topic CSV index | [docs/topic-atlas/data/topic_index.csv](docs/topic-atlas/data/topic_index.csv) |
| Venue-year CSV summary | [docs/topic-atlas/data/venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) |
| Auto-update status | [docs/auto-update/status.md](docs/auto-update/status.md) |

Each topic page includes:

- a Chinese display topic name,
- reproducible English keyword labels,
- representative papers,
- paper-level links such as OpenReview, DOI, Semantic Scholar, or source URLs when available.

## Data Files

| File | Description |
|---|---|
| [docs/topic-atlas/data/topic_index.csv](docs/topic-atlas/data/topic_index.csv) | Topic-level index with year, venue, topic ID, labels, paper counts, representative papers, topic page paths, and GitHub URLs. |
| [docs/topic-atlas/data/venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) | Venue-year coverage summary with paper counts, topic counts, and clustering diagnostics. |
| [docs/clean-2020-plus/README.md](docs/clean-2020-plus/README.md) | Clean 2020-2026 conference-only topic snapshot. |
| [docs/auto-update/status.md](docs/auto-update/status.md) | Latest lightweight update-check report. |

Large raw metadata, embeddings, and intermediate clustering outputs are intentionally kept outside the repository or ignored by Git.

## Scope and Quality Notes

- The public atlas focuses on papers with available public metadata. Review scores, rejected submissions, and private reviewer discussions are not included unless they are publicly released by the venue.
- Coverage can vary by venue and year because conferences and publishers expose metadata through different channels.
- Topic labels are generated from paper metadata and then cleaned for browsing. Treat them as navigation labels, not fixed field definitions.
- Paper links point to public pages when available, including OpenReview, DOI pages, Semantic Scholar, publisher pages, or source metadata URLs.

## Methodology

The current atlas is generated with venue-year independent clustering so that each conference or journal year gets its own topic structure.

Pipeline summary:

1. Collect paper metadata from public sources such as OpenReview, DBLP, OpenAlex, proceedings pages, and publisher metadata when available.
2. Normalize venue-year records and keep the repository-facing scope focused on published, accepted, or otherwise public paper metadata.
3. Embed paper title and abstract text using cached BGE embeddings.
4. Run UMAP dimensionality reduction.
5. Run HDBSCAN leaf clustering for fine-grained topics.
6. Reassign HDBSCAN outliers to nearest topic centroids when confidence is sufficient.
7. Extract reproducible English keyword labels with c-TF-IDF style statistics.
8. Generate heuristic Chinese display names and disambiguate duplicate topic names within each venue-year.
9. Build a static Markdown atlas for browsing and CSV indexes for analysis.

Topic names are meant to help readers navigate the atlas. Representative papers and keyword pools are included so each label can be checked and improved.

## Rebuild

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

## Automatic Updates

The repository includes a GitHub Actions update engine:

| Mode | Purpose | Notes |
|---|---|---|
| Weekly lightweight check | Check which venue-years may have new public metadata. | Runs on a hosted GitHub runner and updates [docs/auto-update/status.md](docs/auto-update/status.md). |
| Full atlas refresh | Rebuild `docs/topic-atlas`. | Requires a self-hosted runner labeled `ai-paper-trends` and external embedding/result caches. |

Run the lightweight check locally:

```bash
python scripts/auto_update_atlas.py check --write-report
```

Run a full refresh on a machine with the ignored `results/` cache:

```bash
python scripts/auto_update_atlas.py refresh
```

Tracked venue schedules and source notes live in [configs/auto_update.yaml](configs/auto_update.yaml).

## Project Structure

| Path | Purpose |
|---|---|
| [docs/topic-atlas/](docs/topic-atlas/README.md) | Static paper-topic atlas. |
| [docs/clean-2020-plus/](docs/clean-2020-plus/README.md) | Conference-only 2020-2026 topic snapshot. |
| [scripts/build_topic_atlas.py](scripts/build_topic_atlas.py) | Static atlas generator. |
| [scripts/fine_grained_topic_analysis.py](scripts/fine_grained_topic_analysis.py) | Fine-grained venue-year topic clustering. |
| [configs/auto_update.yaml](configs/auto_update.yaml) | Venue update schedule and source configuration. |
| [scripts/auto_update_atlas.py](scripts/auto_update_atlas.py) | Lightweight checks and full refresh entrypoint. |
| [main.py](main.py), [src/](src), [configs/](configs) | Original single-conference OpenReview analysis pipeline. |

## Single-Conference OpenReview Pipeline

The repository also retains the original OpenReview + BERTopic workflow for focused conference-level experiments, including review-score plots and OpenReview-only analyses.

Create an environment:

```bash
conda create --name ai-trend-analysis python=3.10
conda activate ai-trend-analysis
pip install -r requirements.txt
```

Run a configured OpenReview task:

```bash
python main.py --config configs/iclr_2025_full_analysis.yaml
```

Use this path for review-score plots, acceptance-type analysis, or small single-conference experiments. Use the atlas pipeline for the multi-venue database.

## Contributing

Contributions are welcome. Useful contributions include:

- correcting venue-year metadata,
- adding new public data sources,
- improving topic labels,
- checking representative papers,
- improving update schedules,
- adding visualization notebooks or analysis examples.

Before opening a PR, please keep changes focused and run the relevant checks:

```bash
git diff --check
python -m py_compile scripts/build_topic_atlas.py
```

If you update the atlas, also verify that `docs/topic-atlas/data/topic_index.csv` links resolve to real topic pages.

Please use [Issues](https://github.com/zhihengli-casia/AI-Paper-Trends/issues) or [Pull Requests](https://github.com/zhihengli-casia/AI-Paper-Trends/pulls).

## License

This project is released under the [MIT License](LICENSE).
