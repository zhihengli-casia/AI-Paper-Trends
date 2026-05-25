# AI Paper Topic Atlas

Continuously updated fine-grained topic index generated from AI conference and journal papers.

Navigation pattern: **year -> venue -> topic -> paper**.

The numbers below describe the current checked-in index. They are expected to grow as new venues, years, and proceedings are added.

- Indexed venue-year groups: **160**
- Indexed papers: **155,662**
- Fine-grained topic pages: **7,378**
- Unassigned papers after reassignment: **25**

## Years

- [2026](2026/README.md) - 7 venues, 11,531 papers, 358 topics
- [2025](2025/README.md) - 25 venues, 37,057 papers, 1,613 topics
- [2024](2024/README.md) - 26 venues, 29,499 papers, 1,312 topics
- [2023](2023/README.md) - 26 venues, 24,374 papers, 1,229 topics
- [2022](2022/README.md) - 25 venues, 18,517 papers, 972 topics
- [2021](2021/README.md) - 26 venues, 18,187 papers, 967 topics
- [2020](2020/README.md) - 25 venues, 16,497 papers, 927 topics

## Venue Groups

| Group | Indexed venues | Venue-years | Papers | Fine topics |
|---|---|---:|---:|---:|
| ML / learning theory | ICLR, ICML, NeurIPS | 19 | 46,161 | 1,724 |
| CV top conferences | CVPR, ICCV, ECCV | 12 | 24,999 | 1,130 |
| NLP / language | ACL, EMNLP, NAACL, COLM | 18 | 15,368 | 811 |
| General AI | AAAI, IJCAI | 13 | 21,179 | 963 |
| Embodied AI / robotics | ICRA, IROS, RSS | 17 | 16,590 | 873 |
| Multimedia / graphics / HCI | ACMMM, SIGGRAPH, SIGGRAPH-Asia, CHI | 20 | 10,473 | 591 |
| Data mining / IR / Web / DB | KDD, SIGIR, WWW, ICDE, SIGMOD | 27 | 7,601 | 482 |
| Medical AI | MICCAI | 6 | 428 | 41 |
| Selected journals | AIJ, JMLR, TPAMI, IJCV, TIP, PR, TMM, TKDE, TNNLS | 28 | 12,863 | 763 |

## Venues

| Venue | Years | Papers | Fine topics | Avg topics/year |
|---|---:|---:|---:|---:|
| AAAI | 2020-2026 (7) | 15,638 | 646 | 92.3 |
| ACL | 2020-2025 (6) | 5,902 | 308 | 51.3 |
| ACMMM | 2020-2025 (6) | 5,006 | 277 | 46.2 |
| AIJ | 2020-2026 (7) | 660 | 61 | 8.7 |
| CHI | 2020-2025 (5) | 4,538 | 229 | 45.8 |
| COLM | 2024-2025 (2) | 717 | 38 | 19.0 |
| CVPR | 2020-2025 (6) | 13,140 | 589 | 98.2 |
| ECCV | 2020-2024 (3) | 5,390 | 265 | 88.3 |
| EMNLP | 2020-2025 (6) | 6,550 | 347 | 57.8 |
| ICCV | 2021-2025 (3) | 6,469 | 276 | 92.0 |
| ICDE | 2020-2025 (6) | 2,059 | 126 | 21.0 |
| ICLR | 2020-2026 (7) | 15,452 | 591 | 84.4 |
| ICML | 2020-2025 (6) | 11,268 | 506 | 84.3 |
| ICRA | 2020-2025 (6) | 8,028 | 412 | 68.7 |
| IJCAI | 2020-2025 (6) | 5,541 | 317 | 52.8 |
| IJCV | 2021-2023 (2) | 355 | 31 | 15.5 |
| IROS | 2020-2025 (6) | 8,068 | 416 | 69.3 |
| JMLR | 2020-2020 (1) | 66 | 8 | 8.0 |
| KDD | 2020-2025 (6) | 1,985 | 117 | 19.5 |
| MICCAI | 2020-2025 (6) | 428 | 41 | 6.8 |
| NAACL | 2021-2025 (4) | 2,199 | 118 | 29.5 |
| NeurIPS | 2020-2025 (6) | 19,441 | 627 | 104.5 |
| PR | 2020-2026 (7) | 5,944 | 330 | 47.1 |
| RSS | 2020-2024 (5) | 494 | 45 | 9.0 |
| SIGGRAPH | 2022-2025 (4) | 413 | 37 | 9.2 |
| SIGGRAPH-Asia | 2021-2025 (5) | 516 | 48 | 9.6 |
| SIGIR | 2020-2025 (6) | 1,077 | 86 | 14.3 |
| SIGMOD | 2020-2022 (3) | 515 | 40 | 13.3 |
| TIP | 2024-2024 (1) | 478 | 26 | 26.0 |
| TKDE | 2020-2026 (2) | 652 | 40 | 20.0 |
| TMM | 2020-2026 (3) | 1,434 | 84 | 28.0 |
| TNNLS | 2023-2026 (3) | 2,042 | 114 | 38.0 |
| TPAMI | 2021-2023 (2) | 1,232 | 69 | 34.5 |
| WWW | 2020-2025 (6) | 1,965 | 113 | 18.8 |

## Data Files

- [venue_year_summary.csv](data/venue_year_summary.csv)
- [topic_index.csv](data/topic_index.csv)

Topic labels include a reproducible English keyword label and a heuristic Chinese display name. Chinese display names are disambiguated within each venue-year when multiple fine topics share the same base label. Use representative paper titles for audit.
