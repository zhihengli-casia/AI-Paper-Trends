"""Regenerate README data sections from committed analysis CSV files.

The README should describe the current dataset, not a hand-maintained snapshot.
This script reads the lightweight CSV artifacts under docs/ and rewrites the
auto-generated blocks in both README files.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25"
VISUALS_DIR = ROOT / "docs/visuals/xhs_composition_atlas_2020_2026"

SUMMARY_CSV = RESULTS_DIR / "run_summary_by_venue_year.csv"
TOP10_CSV = RESULTS_DIR / "top10_topics_by_venue_year.csv"
TOPIC_SUMMARY_CSV = RESULTS_DIR / "topic_summary_by_venue_year.csv"
LABEL_TREND_CSV = RESULTS_DIR / "label_trend_by_venue_year.csv"
FAMILY_CSV = VISUALS_DIR / "venue_year_family_composition.csv"
FULL_TOPIC_COMPOSITION_CSV = VISUALS_DIR / "venue_year_topic_composition_full.csv"

EN_START = "## Current Analysis"
EN_END = "## Quick Start"
CN_START = "## 当前全量分析"
CN_END = "## 快速上手"
AUTO_START = "<!-- AI-PAPER-TRENDS:START -->"
AUTO_END = "<!-- AI-PAPER-TRENDS:END -->"

VENUE_AREAS_EN = {
    "AAAI": "General AI",
    "IJCAI": "General AI",
    "ICLR": "Machine Learning",
    "ICML": "Machine Learning",
    "NeurIPS": "Machine Learning",
    "CVPR": "Computer Vision",
    "ICCV": "Computer Vision",
    "ECCV": "Computer Vision",
    "ACL": "NLP",
    "EMNLP": "NLP",
    "NAACL": "NLP",
    "ACMMM": "Multimedia",
    "KDD": "Data Mining",
    "SIGIR": "Information Retrieval",
    "WWW": "Web / Recommender Systems",
}

VENUE_AREAS_CN = {
    "AAAI": "综合 AI",
    "IJCAI": "综合 AI",
    "ICLR": "机器学习",
    "ICML": "机器学习",
    "NeurIPS": "机器学习",
    "CVPR": "计算机视觉",
    "ICCV": "计算机视觉",
    "ECCV": "计算机视觉",
    "ACL": "NLP",
    "EMNLP": "NLP",
    "NAACL": "NLP",
    "ACMMM": "多媒体",
    "KDD": "数据挖掘",
    "SIGIR": "信息检索",
    "WWW": "Web / 推荐系统",
}

VENUE_ORDER = [
    "AAAI",
    "IJCAI",
    "ICLR",
    "ICML",
    "NeurIPS",
    "CVPR",
    "ICCV",
    "ECCV",
    "ACL",
    "EMNLP",
    "NAACL",
    "ACMMM",
    "KDD",
    "SIGIR",
    "WWW",
]

AREA_ORDER_EN = [
    "Computer Vision",
    "Machine Learning",
    "NLP",
    "General AI",
    "Multimedia",
    "Data Mining / IR / Web",
    "Other / Emerging",
]

AREA_ORDER_CN = [
    "计算机视觉",
    "机器学习",
    "NLP",
    "综合 AI",
    "多媒体",
    "数据挖掘 / 检索 / Web",
    "其他 / 新增会议",
]

AREA_VENUES_EN = {
    "Computer Vision": ["CVPR", "ICCV", "ECCV"],
    "Machine Learning": ["ICLR", "ICML", "NeurIPS"],
    "NLP": ["ACL", "EMNLP", "NAACL"],
    "General AI": ["AAAI", "IJCAI"],
    "Multimedia": ["ACMMM"],
    "Data Mining / IR / Web": ["KDD", "SIGIR", "WWW"],
}

AREA_VENUES_CN = {
    "计算机视觉": ["CVPR", "ICCV", "ECCV"],
    "机器学习": ["ICLR", "ICML", "NeurIPS"],
    "NLP": ["ACL", "EMNLP", "NAACL"],
    "综合 AI": ["AAAI", "IJCAI"],
    "多媒体": ["ACMMM"],
    "数据挖掘 / 检索 / Web": ["KDD", "SIGIR", "WWW"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def fmt_int(value: int | float | str) -> str:
    return f"{int(float(value)):,}"


def fmt_pct(value: str | float) -> str:
    return f"{float(value):.2f}%"


def sort_venues(venues: set[str]) -> list[str]:
    order = {venue: i for i, venue in enumerate(VENUE_ORDER)}
    return sorted(venues, key=lambda venue: (order.get(venue, 999), venue))


def years_label(years: list[int]) -> str:
    return f"{min(years)}-{max(years)}" if years else "-"


def load_context() -> dict[str, object]:
    summary = read_csv(SUMMARY_CSV)
    top10 = read_csv(TOP10_CSV)
    topics = read_csv(TOPIC_SUMMARY_CSV)
    family = read_csv(FAMILY_CSV)

    venues = sort_venues({row["venue"] for row in summary})
    years = sorted({int(row["year"]) for row in summary})
    by_venue: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_year: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        by_venue[row["venue"]].append(row)
        by_year[int(row["year"])].append(row)

    top_by_venue_year: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in top10:
        top_by_venue_year[(row["venue"], int(row["year"]))].append(row)
    for rows in top_by_venue_year.values():
        rows.sort(key=lambda row: int(row["rank"]))

    topics_by_venue_year: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in topics:
        topics_by_venue_year[(row["venue"], int(row["year"]))].append(row)
    for rows in topics_by_venue_year.values():
        rows.sort(key=lambda row: int(row["count"]), reverse=True)

    families = sorted({row["family"] for row in family})
    return {
        "summary": summary,
        "top10": top10,
        "topics": topics,
        "family": family,
        "venues": venues,
        "years": years,
        "by_venue": by_venue,
        "by_year": by_year,
        "top_by_venue_year": top_by_venue_year,
        "topics_by_venue_year": topics_by_venue_year,
        "families": families,
    }


def latest_top_row(
    venue: str,
    by_venue: dict[str, list[dict[str, str]]],
    topics_by_venue_year: dict[tuple[str, int], list[dict[str, str]]],
) -> tuple[int, dict[str, str]]:
    latest_year = max(int(row["year"]) for row in by_venue[venue])
    rows = topics_by_venue_year[(venue, latest_year)]
    return latest_year, rows[0]


def metric_lines(ctx: dict[str, object], lang: str) -> list[str]:
    summary: list[dict[str, str]] = ctx["summary"]  # type: ignore[assignment]
    years: list[int] = ctx["years"]  # type: ignore[assignment]
    families: list[str] = ctx["families"]  # type: ignore[assignment]
    total_papers = sum(int(row["papers"]) for row in summary)
    rows = [
        ("Venues", len({row["venue"] for row in summary})),
        ("Venue-year groups", len(summary)),
        ("Papers used for clustering", total_papers),
        ("Venue-year topics", csv_rows(TOPIC_SUMMARY_CSV)),
        ("Broad topic families in the atlas", len(families)),
        ("Final outlier papers", sum(int(row["final_outliers"]) for row in summary)),
        ("Years covered", years_label(years)),
        ("2026 accepted-paper data currently included", "ICLR, AAAI"),
    ]
    if lang == "cn":
        rows = [
            ("覆盖会议", len({row["venue"] for row in summary})),
            ("会议-年份组", len(summary)),
            ("进入聚类论文", total_papers),
            ("会议-年份细主题", csv_rows(TOPIC_SUMMARY_CSV)),
            ("图谱中的主题大类", len(families)),
            ("最终离群论文", sum(int(row["final_outliers"]) for row in summary)),
            ("年份范围", years_label(years)),
            ("2026 已纳入口径", "ICLR accepted, AAAI Technical Tracks accepted"),
        ]
    lines = ["| Metric | Value |", "|---|---:|"] if lang == "en" else ["| 指标 | 数值 |", "|---|---:|"]
    for label, value in rows:
        value_text = fmt_int(value) if isinstance(value, int) else str(value)
        lines.append(f"| {label} | {value_text} |")
    return lines


def result_table_lines(lang: str) -> list[str]:
    if lang == "en":
        return [
            "| File | Rows | What it contains |",
            "|---|---:|---|",
            f"| [REPORT_CN.md]({RESULTS_DIR.relative_to(ROOT) / 'REPORT_CN.md'}) | - | Full Chinese narrative report |",
            f"| [run_summary_by_venue_year.csv]({SUMMARY_CSV.relative_to(ROOT)}) | {csv_rows(SUMMARY_CSV)} | Per venue-year paper counts, graph sizes, topic counts, outlier rates |",
            f"| [top10_topics_by_venue_year.csv]({TOP10_CSV.relative_to(ROOT)}) | {csv_rows(TOP10_CSV)} | Top 10 topics for every available venue-year |",
            f"| [topic_summary_by_venue_year.csv]({TOPIC_SUMMARY_CSV.relative_to(ROOT)}) | {csv_rows(TOPIC_SUMMARY_CSV)} | Full topic summary with labels, keywords, counts, shares, representative titles |",
            f"| [label_trend_by_venue_year.csv]({LABEL_TREND_CSV.relative_to(ROOT)}) | {csv_rows(LABEL_TREND_CSV)} | Topic-label trend table across years and venues |",
            f"| [venue_year_family_composition.csv]({FAMILY_CSV.relative_to(ROOT)}) | {csv_rows(FAMILY_CSV)} | Broad family composition for every venue-year |",
            f"| [venue_year_topic_composition_full.csv]({FULL_TOPIC_COMPOSITION_CSV.relative_to(ROOT)}) | {csv_rows(FULL_TOPIC_COMPOSITION_CSV)} | Complete fine-grained topic composition for every venue-year |",
        ]
    return [
        "| 文件 | 行数 | 内容 |",
        "|---|---:|---|",
        f"| [REPORT_CN.md]({RESULTS_DIR.relative_to(ROOT) / 'REPORT_CN.md'}) | - | 完整中文报告 |",
        f"| [run_summary_by_venue_year.csv]({SUMMARY_CSV.relative_to(ROOT)}) | {csv_rows(SUMMARY_CSV)} | 每个会议-年份的论文数、图规模、主题数、离群率 |",
        f"| [top10_topics_by_venue_year.csv]({TOP10_CSV.relative_to(ROOT)}) | {csv_rows(TOP10_CSV)} | 每个会议-年份的 Top 10 主题 |",
        f"| [topic_summary_by_venue_year.csv]({TOPIC_SUMMARY_CSV.relative_to(ROOT)}) | {csv_rows(TOPIC_SUMMARY_CSV)} | 全部细主题，含命名、关键词、篇数、占比、代表论文标题 |",
        f"| [label_trend_by_venue_year.csv]({LABEL_TREND_CSV.relative_to(ROOT)}) | {csv_rows(LABEL_TREND_CSV)} | 跨会议、跨年份的主题趋势表 |",
        f"| [venue_year_family_composition.csv]({FAMILY_CSV.relative_to(ROOT)}) | {csv_rows(FAMILY_CSV)} | 每个会议-年份的主题大类组成 |",
        f"| [venue_year_topic_composition_full.csv]({FULL_TOPIC_COMPOSITION_CSV.relative_to(ROOT)}) | {csv_rows(FULL_TOPIC_COMPOSITION_CSV)} | 每个会议-年份的完整细主题组成 |",
    ]


def coverage_by_year_lines(ctx: dict[str, object], lang: str) -> list[str]:
    by_year: dict[int, list[dict[str, str]]] = ctx["by_year"]  # type: ignore[assignment]
    if lang == "en":
        lines = ["| Year | Venue-Year Groups | Papers | Topics |", "|---:|---:|---:|---:|"]
    else:
        lines = ["| 年份 | 会议-年份组 | 论文数 | 主题数 |", "|---:|---:|---:|---:|"]
    for year in sorted(by_year):
        rows = by_year[year]
        papers = sum(int(row["papers"]) for row in rows)
        topics = sum(int(row["topics_excluding_outlier"]) for row in rows)
        lines.append(f"| {year} | {len(rows)} | {fmt_int(papers)} | {fmt_int(topics)} |")
    return lines


def coverage_by_area_lines(ctx: dict[str, object], lang: str) -> list[str]:
    summary: list[dict[str, str]] = ctx["summary"]  # type: ignore[assignment]
    known_area_venues = AREA_VENUES_EN if lang == "en" else AREA_VENUES_CN
    area_order = AREA_ORDER_EN if lang == "en" else AREA_ORDER_CN
    area_for_venue = VENUE_AREAS_EN if lang == "en" else VENUE_AREAS_CN
    other_area = "Other / Emerging" if lang == "en" else "其他 / 新增会议"

    all_venues = {row["venue"] for row in summary}
    area_venues = {area: [venue for venue in venues if venue in all_venues] for area, venues in known_area_venues.items()}
    unknown = [venue for venue in sort_venues(all_venues) if venue not in area_for_venue]
    if unknown:
        area_venues[other_area] = unknown

    if lang == "en":
        lines = ["| Area | Venues | Venue-Year Groups | Papers | Topics |", "|---|---|---:|---:|---:|"]
    else:
        lines = ["| 领域 | 会议 | 会议-年份组 | 论文数 | 主题数 |", "|---|---|---:|---:|---:|"]
    for area in area_order:
        venues = area_venues.get(area, [])
        if not venues:
            continue
        rows = [row for row in summary if row["venue"] in venues]
        papers = sum(int(row["papers"]) for row in rows)
        topics = sum(int(row["topics_excluding_outlier"]) for row in rows)
        lines.append(f"| {area} | {', '.join(venues)} | {len(rows)} | {fmt_int(papers)} | {fmt_int(topics)} |")
    return lines


def coverage_by_venue_lines(ctx: dict[str, object], lang: str) -> list[str]:
    venues: list[str] = ctx["venues"]  # type: ignore[assignment]
    by_venue: dict[str, list[dict[str, str]]] = ctx["by_venue"]  # type: ignore[assignment]
    topics_by_venue_year: dict[tuple[str, int], list[dict[str, str]]] = ctx["topics_by_venue_year"]  # type: ignore[assignment]
    area_for_venue = VENUE_AREAS_EN if lang == "en" else VENUE_AREAS_CN
    other_area = "Other / Emerging" if lang == "en" else "其他 / 新增会议"
    if lang == "en":
        lines = ["| Area | Venue | Years | Venue-Year Groups | Papers | Topics | Latest #1 topic |", "|---|---|---:|---:|---:|---:|---|"]
    else:
        lines = ["| 领域 | 会议 | 年份 | 会议-年份组 | 论文数 | 主题数 | 最新年份 Top1 主题 |", "|---|---|---:|---:|---:|---:|---|"]
    for venue in venues:
        rows = by_venue[venue]
        years = sorted(int(row["year"]) for row in rows)
        papers = sum(int(row["papers"]) for row in rows)
        topics = sum(int(row["topics_excluding_outlier"]) for row in rows)
        latest_year, top = latest_top_row(venue, by_venue, topics_by_venue_year)
        topic = top["specific_label_cn"]
        count = top["count"]
        share = fmt_pct(float(top["share"]) * 100)
        connector = ": " if lang == "en" else "："
        lines.append(
            f"| {area_for_venue.get(venue, other_area)} | {venue} | {years_label(years)} | {len(rows)} | "
            f"{fmt_int(papers)} | {fmt_int(topics)} | {latest_year}{connector}{topic} ({count}, {share}) |"
        )
    return lines


def venue_year_matrix_lines(ctx: dict[str, object], lang: str) -> list[str]:
    venues: list[str] = ctx["venues"]  # type: ignore[assignment]
    years: list[int] = ctx["years"]  # type: ignore[assignment]
    by_venue: dict[str, list[dict[str, str]]] = ctx["by_venue"]  # type: ignore[assignment]
    row_by_venue_year = {
        (venue, int(row["year"])): row
        for venue, rows in by_venue.items()
        for row in rows
    }
    lines: list[str] = []
    if lang == "en":
        lines.extend(
            [
                "<details>",
                "<summary><strong>Full venue-year coverage matrix</strong>: each cell is papers/topics</summary>",
                "",
                "| Venue | " + " | ".join(str(year) for year in years) + " |",
                "|---|" + "|".join("---:" for _ in years) + "|",
            ]
        )
    else:
        lines.extend(
            [
                "<details>",
                "<summary><strong>完整会议-年份覆盖矩阵</strong>：每个单元格为 论文数/主题数</summary>",
                "",
                "| 会议 | " + " | ".join(str(year) for year in years) + " |",
                "|---|" + "|".join("---:" for _ in years) + "|",
            ]
        )
    for venue in venues:
        cells = []
        for year in years:
            row = row_by_venue_year.get((venue, year))
            if row is None:
                cells.append("-")
            else:
                cells.append(f"{fmt_int(row['papers'])}/{row['topics_excluding_outlier']}")
        lines.append(f"| {venue} | " + " | ".join(cells) + " |")
    lines.extend(["", "</details>"])
    return lines


def latest_full_topic_lines(ctx: dict[str, object], lang: str) -> list[str]:
    venues: list[str] = ctx["venues"]  # type: ignore[assignment]
    by_venue: dict[str, list[dict[str, str]]] = ctx["by_venue"]  # type: ignore[assignment]
    topics_by_venue_year: dict[tuple[str, int], list[dict[str, str]]] = ctx["topics_by_venue_year"]  # type: ignore[assignment]
    lines: list[str] = []
    for venue in venues:
        latest_year = max(int(row["year"]) for row in by_venue[venue])
        rows = topics_by_venue_year[(venue, latest_year)]
        total = int(rows[0]["venue_year_total"]) if rows else 0
        if lang == "en":
            lines.extend(
                [
                    f"<details>",
                    f"<summary><strong>{venue} {latest_year}</strong>: all {len(rows)} topics, {fmt_int(total)} papers</summary>",
                    "",
                    "| Rank | Topic label | Papers | Share |",
                    "|---:|---|---:|---:|",
                ]
            )
        else:
            lines.extend(
                [
                    f"<details>",
                    f"<summary><strong>{venue} {latest_year}</strong>：全部 {len(rows)} 个主题，{fmt_int(total)} 篇论文</summary>",
                    "",
                    "| 排名 | 主题 | 篇数 | 占比 |",
                    "|---:|---|---:|---:|",
                ]
            )
        for rank, row in enumerate(rows, start=1):
            lines.append(f"| {rank} | {row['specific_label_cn']} | {row['count']} | {fmt_pct(float(row['share']) * 100)} |")
        lines.extend(["", "</details>", ""])
    return lines


def generated_block(ctx: dict[str, object], lang: str) -> str:
    if lang == "en":
        lines = [
            "<!-- AI-PAPER-TRENDS:START -->",
            "## Current Analysis",
            "",
            "The latest committed run is an accepted-paper analysis for major AI venues from 2020 to 2026.",
            "",
            *metric_lines(ctx, lang),
            "",
            "Method:",
            "",
            "`BGE embeddings -> cosine kNN graph -> Louvain community detection -> small-community merge -> deterministic Chinese topic naming`",
            "",
            "Scope:",
            "",
            "- Accepted main-conference papers only.",
            "- NLP venues exclude Findings, Industry, SRW, and other non-main tracks.",
            "- 2026 currently includes legally confirmed `ICLR 2026 accepted` and `AAAI 2026 Technical Tracks accepted`.",
            "- Large raw crawls, per-paper topic assignments, embedding files, and model caches are intentionally not committed.",
            "",
            "Committed result tables:",
            "",
            *result_table_lines(lang),
            "",
            "## Topic Composition Atlas",
            "",
            "The repository keeps the visual atlas as linked files instead of embedding large PNGs in the README.",
            "",
            "- Year-by-year venue composition: [docs/visuals/xhs_composition_atlas_2020_2026/by_year/](docs/visuals/xhs_composition_atlas_2020_2026/by_year/)",
            "- Venue-by-venue yearly composition: [docs/visuals/xhs_composition_atlas_2020_2026/by_venue/](docs/visuals/xhs_composition_atlas_2020_2026/by_venue/)",
            "",
            "Each horizontal composition bar represents all accepted papers from one venue-year. Colors encode broad topic families and segment lengths encode shares. The right-side labels are only reading aids; the full fine-grained composition is available in CSV.",
            "",
            "## Coverage by Year",
            "",
            *coverage_by_year_lines(ctx, lang),
            "",
            "## Coverage by Area",
            "",
            *coverage_by_area_lines(ctx, lang),
            "",
            "## Coverage by Venue",
            "",
            *coverage_by_venue_lines(ctx, lang),
            "",
            "## Full Venue-Year Matrix",
            "",
            *venue_year_matrix_lines(ctx, lang),
            "",
            "## Latest Full Topic Lists by Venue",
            "",
            "The README lists all topics for the latest available year of every venue, rather than a few hand-picked examples. Complete topic rows for every venue-year are in `topic_summary_by_venue_year.csv` and `venue_year_topic_composition_full.csv`.",
            "",
            *latest_full_topic_lines(ctx, lang),
            "## Automatic README Updates",
            "",
            "README statistics are generated from the committed CSV artifacts by `src/update_readme_data.py`. When a new venue or year is added to the result CSVs, rerun the script and the coverage tables plus latest full-topic sections will update automatically.",
            "",
            "A GitHub Actions workflow also runs the script when result CSVs change on `main`.",
            "<!-- AI-PAPER-TRENDS:END -->",
        ]
    else:
        lines = [
            "<!-- AI-PAPER-TRENDS:START -->",
            "## 当前全量分析",
            "",
            "最新一次分析使用 2020-2026 年主会 accepted 论文，目标是给出“每一年、每个会议”的完整主题组成，而不是只列几个热门方向。",
            "",
            *metric_lines(ctx, lang),
            "",
            "聚类口径：",
            "",
            "- 论文范围：主会 accepted paper；NLP 不含 Findings / Industry 等非主会 track。",
            "- 聚类单位：每个会议的每一年单独聚类，例如 `CVPR 2025`、`ICLR 2026` 分开跑。",
            "- 主题算法：`BGE embedding -> kNN cosine graph -> Louvain community detection -> 小社区合并 -> 中文规则命名`。",
            "- 2026 数据：目前只纳入已能合法确认的 `ICLR 2026 accepted` 和 `AAAI 2026 Technical Tracks accepted`。",
            "",
            "已提交的轻量结果文件：",
            "",
            *result_table_lines(lang),
            "",
            "没有把完整 per-paper 结果和原始抓取数据提交到仓库，因为体积较大；这些文件应在本地 `data/` 和 `results/` 下生成。",
            "",
            "## 主题组成图谱",
            "",
            "README 不再直接嵌入大 PNG，图谱文件以链接形式保留，避免首页被低质量长图占满。",
            "",
            "- 按年份查看所有会议组成：[docs/visuals/xhs_composition_atlas_2020_2026/by_year/](docs/visuals/xhs_composition_atlas_2020_2026/by_year/)",
            "- 按会议查看历年组成：[docs/visuals/xhs_composition_atlas_2020_2026/by_venue/](docs/visuals/xhs_composition_atlas_2020_2026/by_venue/)",
            "",
            "每条横向组成条都代表某个会议某一年的全部论文，颜色表示主题大类，长度表示占比。右侧文字只作为导览；精确到每个细主题的完整构成在 CSV 中。",
            "",
            "## 按年份覆盖",
            "",
            *coverage_by_year_lines(ctx, lang),
            "",
            "## 按领域覆盖",
            "",
            *coverage_by_area_lines(ctx, lang),
            "",
            "## 按会议覆盖",
            "",
            *coverage_by_venue_lines(ctx, lang),
            "",
            "## 完整会议-年份覆盖矩阵",
            "",
            *venue_year_matrix_lines(ctx, lang),
            "",
            "## 每个会议最新年份完整主题清单",
            "",
            "这里展示每个会议最新年份的全部主题，不再只放前 5、前 10 或少数手工挑选的例子。每个会议-年份的完整细主题在 `topic_summary_by_venue_year.csv` 和 `venue_year_topic_composition_full.csv` 中。",
            "",
            *latest_full_topic_lines(ctx, lang),
            "## README 自动更新",
            "",
            "README 的统计表和完整主题清单由 `src/update_readme_data.py` 从已提交的 CSV 结果自动生成。以后新增会议或新增年份，只要结果 CSV 里出现新 venue/year，重新运行脚本即可自动进入覆盖表和最新年份完整主题清单。",
            "",
            "`main` 分支还配置了 GitHub Actions：当结果 CSV 变化时，会自动运行 README 更新脚本并提交变更。",
            "<!-- AI-PAPER-TRENDS:END -->",
        ]
    return "\n".join(lines)


def replace_block(path: Path, start_heading: str, end_heading: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if AUTO_START in text and AUTO_END in text:
        start = text.index(AUTO_START)
        end = text.index(AUTO_END) + len(AUTO_END)
    else:
        start = text.index(start_heading)
        end = text.index(end_heading)
    suffix = text[end:].lstrip("\n")
    new_text = text[:start] + block.rstrip() + "\n\n" + suffix
    path.write_text(new_text, encoding="utf-8")


def main() -> None:
    ctx = load_context()
    replace_block(ROOT / "README.md", EN_START, EN_END, generated_block(ctx, "en"))
    replace_block(ROOT / "README_cn.md", CN_START, CN_END, generated_block(ctx, "cn"))
    print("Updated README.md and README_cn.md from analysis CSVs.")


if __name__ == "__main__":
    main()
