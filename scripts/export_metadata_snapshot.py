#!/usr/bin/env python3
"""Export a GitHub-friendly metadata coverage snapshot.

The full metadata CSV/JSONL files are too large for normal GitHub storage.
This script exports compact coverage tables and a Markdown summary under
docs/metadata-snapshot/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


AREA_BY_VENUE = {
    "AAAI": "General AI",
    "IJCAI": "General AI",
    "ICLR": "ML / learning",
    "ICML": "ML / learning",
    "NeurIPS": "ML / learning",
    "JMLR": "ML / learning",
    "TNNLS": "ML / learning",
    "CVPR": "Computer vision",
    "ICCV": "Computer vision",
    "ECCV": "Computer vision",
    "TPAMI": "Computer vision",
    "IJCV": "Computer vision",
    "TIP": "Computer vision",
    "PR": "Computer vision",
    "ACL": "NLP / language",
    "EMNLP": "NLP / language",
    "NAACL": "NLP / language",
    "COLM": "NLP / language",
    "ICRA": "Robotics / embodied AI",
    "IROS": "Robotics / embodied AI",
    "RSS": "Robotics / embodied AI",
    "ACMMM": "Multimedia / HCI / graphics",
    "CHI": "Multimedia / HCI / graphics",
    "SIGGRAPH": "Multimedia / HCI / graphics",
    "SIGGRAPH-Asia": "Multimedia / HCI / graphics",
    "TMM": "Multimedia / HCI / graphics",
    "KDD": "Data / IR / Web / DB",
    "SIGIR": "Data / IR / Web / DB",
    "WWW": "Data / IR / Web / DB",
    "ICDE": "Data / IR / Web / DB",
    "SIGMOD": "Data / IR / Web / DB",
    "VLDB": "Data / IR / Web / DB",
    "TKDE": "Data / IR / Web / DB",
    "AIJ": "AI journals",
    "MICCAI": "Medical AI",
}


EXPECTED_2020_PLUS = {
    "AAAI": range(2020, 2027),
    "ACL": range(2020, 2026),
    "ACMMM": range(2020, 2026),
    "CHI": range(2020, 2026),
    "COLM": range(2024, 2026),
    "CVPR": range(2020, 2026),
    "ECCV": [2020, 2022, 2024],
    "EMNLP": range(2020, 2026),
    "ICCV": [2021, 2023, 2025],
    "ICDE": range(2020, 2026),
    "ICLR": range(2020, 2027),
    "ICML": range(2020, 2026),
    "ICRA": range(2020, 2026),
    "IJCAI": range(2020, 2026),
    "IROS": range(2020, 2026),
    "KDD": range(2020, 2026),
    "MICCAI": range(2020, 2026),
    "NAACL": [2021, 2022, 2024, 2025],
    "NeurIPS": range(2020, 2026),
    "RSS": range(2020, 2026),
    "SIGGRAPH": range(2020, 2026),
    "SIGGRAPH-Asia": range(2020, 2026),
    "SIGIR": range(2020, 2026),
    "SIGMOD": range(2020, 2026),
    "VLDB": range(2020, 2026),
    "WWW": range(2020, 2026),
    "TPAMI": range(2020, 2027),
    "IJCV": range(2020, 2027),
    "TIP": range(2020, 2027),
    "TMM": range(2020, 2027),
    "PR": range(2020, 2027),
    "TKDE": range(2020, 2027),
    "AIJ": range(2020, 2027),
    "TNNLS": range(2020, 2027),
    "JMLR": range(2020, 2027),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "metadata_expansion"
        / "expanded_paper_metadata_1969_2026_merged_summary.csv",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "docs" / "metadata-snapshot")
    return parser.parse_args()


def write_markdown_table(file, rows: list[dict[str, object]], columns: list[str]) -> None:
    file.write("| " + " | ".join(columns) + " |\n")
    file.write("|" + "|".join("---" for _ in columns) + "|\n")
    for row in rows:
        file.write("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n")


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.rename(columns={"count": "papers"})
    df["venue_type"] = df["venue_type"].fillna("conference")
    df["area"] = df["venue"].map(AREA_BY_VENUE).fillna("Other")
    df["abstract_rate"] = (df["abstracts"] / df["papers"]).fillna(0).round(4)
    df = df[["area", "venue", "venue_type", "year", "papers", "abstracts", "abstract_rate", "doi"]]
    df = df.sort_values(["year", "area", "venue"])
    df.to_csv(data_dir / "venue_year_metadata_summary.csv", index=False)

    year_summary = (
        df.groupby("year")
        .agg(
            venues=("venue", "nunique"),
            venue_year_units=("venue", "count"),
            papers=("papers", "sum"),
            abstracts=("abstracts", "sum"),
            doi=("doi", "sum"),
        )
        .reset_index()
        .sort_values("year")
    )
    year_summary["abstract_rate"] = (year_summary["abstracts"] / year_summary["papers"]).round(4)
    year_summary.to_csv(data_dir / "year_metadata_summary.csv", index=False)

    venue_summary = (
        df.groupby(["area", "venue", "venue_type"])
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            years=("year", "nunique"),
            papers=("papers", "sum"),
            abstracts=("abstracts", "sum"),
            doi=("doi", "sum"),
        )
        .reset_index()
        .sort_values(["area", "venue"])
    )
    venue_summary["abstract_rate"] = (venue_summary["abstracts"] / venue_summary["papers"]).round(4)
    venue_summary.to_csv(data_dir / "venue_metadata_summary.csv", index=False)

    area_summary = (
        df.groupby("area")
        .agg(
            venues=("venue", "nunique"),
            venue_year_units=("venue", "count"),
            papers=("papers", "sum"),
            abstracts=("abstracts", "sum"),
            doi=("doi", "sum"),
        )
        .reset_index()
        .sort_values("papers", ascending=False)
    )
    area_summary["abstract_rate"] = (area_summary["abstracts"] / area_summary["papers"]).round(4)
    area_summary.to_csv(data_dir / "area_metadata_summary.csv", index=False)

    gaps = []
    for venue, years in EXPECTED_2020_PLUS.items():
        present = set(df.loc[(df["venue"] == venue) & (df["year"] >= 2020), "year"].astype(int))
        for year in years:
            if int(year) not in present:
                gaps.append({"area": AREA_BY_VENUE.get(venue, "Other"), "venue": venue, "year": int(year)})
    gap_df = pd.DataFrame(gaps, columns=["area", "venue", "year"])
    gap_df.to_csv(data_dir / "remaining_2020_2026_gaps.csv", index=False)

    latest = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_papers = int(df["papers"].sum())
    total_units = len(df)
    total_venues = df["venue"].nunique()
    y2020 = df[df["year"] >= 2020]
    pre2020 = df[df["year"] < 2020]

    with (output_root / "README.md").open("w", encoding="utf-8") as file:
        file.write("# Metadata Coverage Snapshot\n\n")
        file.write(
            "This directory contains compact coverage tables generated from the local expanded metadata cache. "
            "It is separate from the topic atlas: these files describe what has been crawled and normalized, "
            "not what has already been clustered into topic pages.\n\n"
        )
        file.write(f"Last exported: `{latest}`.\n\n")
        write_markdown_table(
            file,
            [
                {"Metric": "Total metadata records", "Value": f"{total_papers:,}"},
                {"Metric": "Venues", "Value": f"{total_venues:,}"},
                {"Metric": "Venue-year rows", "Value": f"{total_units:,}"},
                {"Metric": "2020+ records", "Value": f"{int(y2020['papers'].sum()):,}"},
                {"Metric": "Pre-2020 records", "Value": f"{int(pre2020['papers'].sum()):,}"},
                {"Metric": "Years", "Value": f"{int(df['year'].min())}-{int(df['year'].max())}"},
            ],
            ["Metric", "Value"],
        )
        file.write("\n## Data Files\n\n")
        write_markdown_table(
            file,
            [
                {
                    "File": "[venue_year_metadata_summary.csv](data/venue_year_metadata_summary.csv)",
                    "Description": "One row per venue-year with paper, abstract, DOI, source type, and area coverage.",
                },
                {
                    "File": "[venue_metadata_summary.csv](data/venue_metadata_summary.csv)",
                    "Description": "One row per venue with year range, paper counts, abstract coverage, and DOI coverage.",
                },
                {
                    "File": "[year_metadata_summary.csv](data/year_metadata_summary.csv)",
                    "Description": "Annual metadata coverage across venues.",
                },
                {
                    "File": "[area_metadata_summary.csv](data/area_metadata_summary.csv)",
                    "Description": "Coverage grouped by research area.",
                },
                {
                    "File": "[remaining_2020_2026_gaps.csv](data/remaining_2020_2026_gaps.csv)",
                    "Description": "Known 2020+ venue-year cells that still need source-specific follow-up.",
                },
            ],
            ["File", "Description"],
        )
        file.write("\n## 2020+ Year Summary\n\n")
        rows = year_summary[year_summary["year"] >= 2020].copy()
        rows["papers"] = rows["papers"].map(lambda value: f"{int(value):,}")
        rows["abstracts"] = rows["abstracts"].map(lambda value: f"{int(value):,}")
        write_markdown_table(
            file,
            rows[["year", "venues", "venue_year_units", "papers", "abstracts", "abstract_rate"]]
            .rename(
                columns={
                    "year": "Year",
                    "venues": "Venues",
                    "venue_year_units": "Venue-year rows",
                    "papers": "Papers",
                    "abstracts": "Abstracts",
                    "abstract_rate": "Abstract rate",
                }
            )
            .to_dict("records"),
            ["Year", "Venues", "Venue-year rows", "Papers", "Abstracts", "Abstract rate"],
        )
        file.write("\n## Area Summary\n\n")
        area_rows = area_summary.copy()
        area_rows["papers"] = area_rows["papers"].map(lambda value: f"{int(value):,}")
        write_markdown_table(
            file,
            area_rows[["area", "venues", "venue_year_units", "papers", "abstract_rate"]]
            .rename(
                columns={
                    "area": "Area",
                    "venues": "Venues",
                    "venue_year_units": "Venue-year rows",
                    "papers": "Papers",
                    "abstract_rate": "Abstract rate",
                }
            )
            .to_dict("records"),
            ["Area", "Venues", "Venue-year rows", "Papers", "Abstract rate"],
        )
        file.write(
            "\n## Notes\n\n"
            "- The full paper-level metadata files are intentionally not committed because they are hundreds of MB.\n"
            "- Some publishers expose only titles/authors/DOIs through public indexes, so abstract coverage varies by source.\n"
            "- The topic atlas will need a separate clustering refresh before these expanded metadata records appear as topic pages.\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
