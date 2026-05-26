# Metadata Coverage Snapshot

This directory contains compact coverage tables generated from the local expanded metadata cache. It is separate from the topic atlas: these files describe what has been crawled and normalized, not what has already been clustered into topic pages.

Last exported: `2026-05-26 02:59 UTC`.

| Metric | Value |
|---|---|
| Total metadata records | 338,496 |
| Venues | 35 |
| Venue-year rows | 781 |
| 2020+ records | 197,830 |
| Pre-2020 records | 140,666 |
| Years | 1974-2026 |

## Data Files

| File | Description |
|---|---|
| [venue_year_metadata_summary.csv](data/venue_year_metadata_summary.csv) | One row per venue-year with paper, abstract, DOI, source type, and area coverage. |
| [venue_metadata_summary.csv](data/venue_metadata_summary.csv) | One row per venue with year range, paper counts, abstract coverage, and DOI coverage. |
| [year_metadata_summary.csv](data/year_metadata_summary.csv) | Annual metadata coverage across venues. |
| [area_metadata_summary.csv](data/area_metadata_summary.csv) | Coverage grouped by research area. |
| [remaining_2020_2026_gaps.csv](data/remaining_2020_2026_gaps.csv) | Known 2020+ venue-year cells that still need source-specific follow-up. |

## 2020+ Year Summary

| Year | Venues | Venue-year rows | Papers | Abstracts | Abstract rate |
|---|---|---|---|---|---|
| 2020 | 26 | 26 | 18,955 | 14,570 | 0.7687 |
| 2021 | 28 | 28 | 21,612 | 16,165 | 0.748 |
| 2022 | 26 | 26 | 21,035 | 15,812 | 0.7517 |
| 2023 | 28 | 28 | 27,915 | 21,378 | 0.7658 |
| 2024 | 28 | 28 | 35,894 | 28,328 | 0.7892 |
| 2025 | 27 | 27 | 46,402 | 37,821 | 0.8151 |
| 2026 | 7 | 7 | 26,017 | 24,982 | 0.9602 |

## Area Summary

| Area | Venues | Venue-year rows | Papers | Abstract rate |
|---|---|---|---|---|
| ML / learning | 5 | 115 | 103,691 | 0.895 |
| Robotics / embodied AI | 3 | 100 | 55,982 | 0.0889 |
| Computer vision | 7 | 100 | 53,860 | 0.7473 |
| General AI | 2 | 43 | 35,072 | 0.8164 |
| Multimedia / HCI / graphics | 5 | 125 | 28,334 | 0.3312 |
| Data / IR / Web / DB | 7 | 156 | 28,236 | 0.2991 |
| NLP / language | 4 | 92 | 25,405 | 0.728 |
| Medical AI | 1 | 28 | 5,781 | 0.0016 |
| AI journals | 1 | 22 | 2,135 | 0.1274 |

## Notes

- The full paper-level metadata files are intentionally not committed because they are hundreds of MB.
- Some publishers expose only titles/authors/DOIs through public indexes, so abstract coverage varies by source.
- The topic atlas will need a separate clustering refresh before these expanded metadata records appear as topic pages.
