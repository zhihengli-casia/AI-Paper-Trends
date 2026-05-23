#!/usr/bin/env python3
"""Automation entry point for keeping the topic atlas fresh.

The update engine has two responsibilities:

1. ``check`` runs cheaply on GitHub-hosted runners. It compares the committed
   atlas coverage with the configured venue schedule and writes an update queue.
2. ``refresh`` runs the local heavy pipeline when a runner has access to the
   ignored ``results/`` cache and model files.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "auto_update.yaml"


@dataclass(frozen=True)
class CoverageRow:
    venue: str
    year: int
    papers: int
    topics: int


@dataclass(frozen=True)
class QueueItem:
    venue: str
    year: int
    state: str
    expected_month: int
    source: str
    scope: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Write the atlas update queue.")
    check.add_argument("--today", type=str, help="Override today's date, formatted as YYYY-MM-DD.")
    check.add_argument("--write-report", action="store_true", help="Write Markdown and JSON reports.")
    check.add_argument("--print-json", action="store_true", help="Print the report JSON to stdout.")

    refresh = subparsers.add_parser(
        "refresh",
        help="Run fine-grained clustering and rebuild docs/topic-atlas from local caches.",
    )
    refresh.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Only rebuild docs/topic-atlas from an existing fine-topic result root.",
    )
    refresh.add_argument("--no-clean", action="store_true", help="Do not clean docs/topic-atlas first.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_coverage(summary_path: Path) -> list[CoverageRow]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing atlas summary: {summary_path}")
    rows: list[CoverageRow] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
                CoverageRow(
                    venue=str(row["venue"]),
                    year=int(row["year"]),
                    papers=int(float(row["papers"])),
                    topics=int(float(row["final_topics"])),
                )
            )
    return rows


def should_run_in_year(frequency: str, year: int) -> bool:
    if frequency == "annual":
        return True
    if frequency == "biennial_even":
        return year % 2 == 0
    if frequency == "biennial_odd":
        return year % 2 == 1
    raise ValueError(f"Unsupported venue frequency: {frequency}")


def build_update_queue(
    config: dict[str, Any],
    coverage_rows: list[CoverageRow],
    today: date,
) -> list[QueueItem]:
    coverage = config.get("coverage", {})
    covered = {(row.venue, row.year) for row in coverage_rows}
    default_start_year = int(coverage.get("default_start_year", 2020))
    lookahead_years = int(coverage.get("lookahead_years", 0))
    max_year = today.year + lookahead_years

    queue: list[QueueItem] = []
    for venue_config in config.get("venues", []):
        if not venue_config.get("active", True):
            continue
        venue = str(venue_config["venue"])
        frequency = str(venue_config.get("frequency", "annual"))
        start_year = int(venue_config.get("start_year", default_start_year))
        end_year = int(venue_config.get("end_year", max_year))
        expected_month = int(venue_config.get("expected_month", 12))
        only_years = {int(year) for year in venue_config.get("only_years", [])}
        skip_years = {int(year) for year in venue_config.get("skip_years", [])}

        for year in range(start_year, min(end_year, max_year) + 1):
            if only_years and year not in only_years:
                continue
            if year in skip_years:
                continue
            if not should_run_in_year(frequency, year):
                continue
            if (venue, year) in covered:
                continue
            if year < today.year or (year == today.year and today.month >= expected_month):
                state = "pending_due"
            else:
                state = "watching"
            queue.append(
                QueueItem(
                    venue=venue,
                    year=year,
                    state=state,
                    expected_month=expected_month,
                    source=str(venue_config.get("source", "")),
                    scope=str(venue_config.get("scope", "")),
                )
            )

    return sorted(queue, key=lambda item: (item.state != "pending_due", item.year, item.venue))


def coverage_by_venue(rows: list[CoverageRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CoverageRow]] = defaultdict(list)
    for row in rows:
        grouped[row.venue].append(row)

    summary = []
    for venue, venue_rows in sorted(grouped.items()):
        years = sorted(row.year for row in venue_rows)
        summary.append(
            {
                "venue": venue,
                "covered_years": ", ".join(str(year) for year in years),
                "year_count": len(years),
                "papers": sum(row.papers for row in venue_rows),
                "topics": sum(row.topics for row in venue_rows),
            }
        )
    return summary


def build_report(
    coverage_rows: list[CoverageRow],
    queue: list[QueueItem],
    config: dict[str, Any],
) -> dict[str, Any]:
    pending_due = [item for item in queue if item.state == "pending_due"]
    watching = [item for item in queue if item.state == "watching"]
    return {
        "summary": {
            "covered_venue_years": len(coverage_rows),
            "covered_papers": sum(row.papers for row in coverage_rows),
            "covered_topics": sum(row.topics for row in coverage_rows),
            "pending_due": len(pending_due),
            "watching": len(watching),
        },
        "coverage_by_venue": coverage_by_venue(coverage_rows),
        "update_queue": [item.__dict__ for item in queue],
        "pipeline": config.get("pipeline", {}),
    }


def markdown_table(rows: list[list[Any]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Auto Update Status",
        "",
        "This file is generated by `scripts/auto_update_atlas.py check`.",
        "",
        "State meanings:",
        "",
        "- `pending_due`: the expected proceedings month has passed, but the atlas has no entry yet.",
        "- `watching`: the venue-year is expected later and should be checked again.",
        "",
        "## Summary",
        "",
        f"- Covered venue-year groups: **{summary['covered_venue_years']}**",
        f"- Covered papers: **{summary['covered_papers']:,}**",
        f"- Covered fine topics: **{summary['covered_topics']:,}**",
        f"- Pending due venue-years: **{summary['pending_due']}**",
        f"- Watching venue-years: **{summary['watching']}**",
        "",
        "## Update Queue",
        "",
    ]

    queue_rows = [
        [
            item["venue"],
            item["year"],
            f"`{item['state']}`",
            item["expected_month"],
            item["source"],
            item["scope"],
        ]
        for item in report["update_queue"]
    ]
    if queue_rows:
        lines.extend(
            markdown_table(
                queue_rows,
                ["Venue", "Year", "State", "Expected month", "Source", "Scope"],
            )
        )
    else:
        lines.append("No missing venue-year volumes are currently expected.")

    lines.extend(["", "## Current Coverage", ""])
    coverage_rows = [
        [
            item["venue"],
            item["covered_years"],
            item["year_count"],
            f"{item['papers']:,}",
            f"{item['topics']:,}",
        ]
        for item in report["coverage_by_venue"]
    ]
    lines.extend(
        markdown_table(
            coverage_rows,
            ["Venue", "Covered years", "Count", "Papers", "Fine topics"],
        )
    )
    lines.extend(
        [
            "",
            "## Full Refresh",
            "",
            "The hosted GitHub runner can update this status page. A full atlas refresh requires a runner "
            "with access to the ignored `results/` embedding and fine-topic caches.",
            "",
            "```bash",
            "python scripts/auto_update_atlas.py refresh",
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_reports(config: dict[str, Any], report: dict[str, Any]) -> None:
    coverage = config.get("coverage", {})
    report_markdown = resolve_path(coverage.get("report_markdown", "docs/auto-update/status.md"))
    report_json = resolve_path(coverage.get("report_json", "docs/auto-update/status.json"))
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.write_text(render_markdown(report), encoding="utf-8")
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_check(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    coverage = config.get("coverage", {})
    summary_path = resolve_path(coverage.get("summary_path", "docs/topic-atlas/data/venue_year_summary.csv"))
    today = date.fromisoformat(args.today) if args.today else date.today()
    coverage_rows = read_coverage(summary_path)
    queue = build_update_queue(config, coverage_rows, today)
    report = build_report(coverage_rows, queue, config)
    if args.write_report:
        write_reports(config, report)
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(
        "Auto-update check complete: "
        f"{report['summary']['pending_due']} pending_due, "
        f"{report['summary']['watching']} watching."
    )
    return report


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_refresh(args: argparse.Namespace, config: dict[str, Any]) -> None:
    pipeline = config.get("pipeline", {})
    cached_embedding_root = resolve_path(
        pipeline.get(
            "cached_embedding_root",
            "results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25",
        )
    )
    fine_topic_root = resolve_path(
        pipeline.get("fine_topic_root", "results/fine_grained_venue_year_topics_2020_2026_mcs_fine")
    )
    atlas_root = resolve_path(pipeline.get("atlas_root", "docs/topic-atlas"))

    if not args.skip_clustering:
        if not cached_embedding_root.exists():
            raise FileNotFoundError(
                "Cannot run full refresh because the cached embedding root is missing: "
                f"{cached_embedding_root}. Use a self-hosted runner or run with --skip-clustering "
                "after preparing fine-topic outputs."
            )
        run_command(
            [
                "python",
                "scripts/fine_grained_topic_analysis.py",
                "--input-root",
                str(cached_embedding_root),
                "--output-root",
                str(fine_topic_root),
            ]
        )

    if not fine_topic_root.exists():
        raise FileNotFoundError(f"Missing fine-topic result root: {fine_topic_root}")

    build_command = [
        "python",
        "scripts/build_topic_atlas.py",
        "--topic-root",
        str(fine_topic_root),
        "--output-root",
        str(atlas_root),
    ]
    if not args.no_clean:
        build_command.append("--clean")
    run_command(build_command)
    print("Atlas refresh complete.")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "check":
        run_check(args, config)
    elif args.command == "refresh":
        run_refresh(args, config)
    else:
        raise SystemExit(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
