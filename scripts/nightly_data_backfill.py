#!/usr/bin/env python3
"""Overnight metadata backfill runner.

Priority:
1. Fill known 2020+ metadata gaps and source-specific crawler misses.
2. Rebuild the 1969-2026 merged metadata snapshot from all completed cache.

This runner is metadata-only. It does not run embeddings or clustering.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crawl_metadata_expansion import (  # noqa: E402
    DblpJob,
    JournalJob,
    build_journal_jobs,
    clean_doi,
    clean_text,
    dedupe,
    enrich_doi_abstracts,
    external_cache_path,
    make_record,
    merge_with_base,
    read_jsonl,
    should_include_dblp_entry,
    split_authors,
    write_csv,
    write_jsonl,
    write_missing_abstracts,
    write_summary,
)

HEADERS = {"User-Agent": "AI-Paper-Trends/0.2 (+public metadata research)"}
DEFAULT_TIMEOUT = 90


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def http_get(url: str, *, retries: int = 5, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 404:
                raise
            last_error = exc
            time.sleep(min(90, 8 * (attempt + 1)))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(90, 8 * (attempt + 1)))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def soup_from_url(url: str) -> BeautifulSoup:
    return BeautifulSoup(http_get(url).text, "lxml")


def parse_dblp_page(job: DblpJob, source_url: str, *, source: str = "DBLP + OpenAlex") -> list[dict[str, Any]]:
    soup = soup_from_url(source_url)
    records: list[dict[str, Any]] = []
    for entry in soup.select("li.entry.inproceedings, li.entry.article"):
        cite = entry.select_one("cite")
        title_el = entry.select_one("cite .title")
        if not cite or not title_el:
            continue
        title = clean_text(title_el.get_text(" ", strip=True)).rstrip(".")
        section_el = entry.find_previous(["h2", "h3"])
        section = clean_text(section_el.get_text(" ", strip=True) if section_el else "")
        if job.venue == "SIGMOD" and "PACMMOD" in source and "SIGMOD" not in section.upper():
            continue
        if not should_include_dblp_entry(job.venue, section, title):
            continue
        authors = [
            clean_text(author.get_text(" ", strip=True))
            for author in cite.select('span[itemprop="author"] span[itemprop="name"]')
        ]
        doi = ""
        html_url = ""
        for link in entry.select('nav.publ a[href^="https://doi.org/"]'):
            doi = clean_doi(link["href"])
            html_url = link["href"]
            break
        dblp_url = ""
        detail_link = entry.select_one('li.details a[href], a[href*="dblp.org/rec/"][href$=".html"]')
        if detail_link:
            dblp_url = detail_link["href"]
        records.append(
            make_record(
                domain=job.domain,
                venue=job.venue,
                year=job.year,
                source=source,
                status="accepted",
                track=section,
                title=title,
                authors=authors,
                html_url=html_url,
                doi=doi,
                source_url=source_url,
                venue_type=job.venue_type,
                extra={"dblp_url": dblp_url},
            )
        )
    return records


def crawl_dblp_urls(
    job: DblpJob,
    urls: list[str],
    *,
    cache_dir: Path,
    refresh: bool,
    enrich_openalex: bool,
    max_workers: int,
    source: str = "DBLP + OpenAlex",
) -> list[dict[str, Any]]:
    label = f"dblp_{job.venue}_{job.year}"
    path = external_cache_path(cache_dir, label)
    if path.exists() and not refresh:
        records = read_jsonl(path)
        log(f"[CACHE] {label}: {len(records):,}")
        return records

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for url in urls:
        try:
            page_records = parse_dblp_page(job, url, source=source)
            if page_records:
                log(f"[OK] {label}: {len(page_records):,} from {url}")
                records.extend(page_records)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{url}: {exc}")
            log(f"[WARN] {label}: {exc}")
    records = dedupe(records)
    if enrich_openalex and records:
        filled = enrich_doi_abstracts(records, max_workers=max_workers, desc=f"{job.venue} {job.year} OpenAlex")
        log(f"[OK] {label}: OpenAlex filled {filled:,} abstracts")
    write_jsonl(path, records)
    if failures:
        (cache_dir / f"{label}_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    log(f"[DONE] {label}: {len(records):,}")
    return records


def miccai_urls(year: int) -> list[str]:
    return [f"https://dblp.org/db/conf/miccai/miccai{year}{suffix}.html" for suffix in ["", *[f"-{i}" for i in range(1, 9)]]]


def pvldb_urls(year: int) -> list[str]:
    volume = year - 2007
    return [f"https://dblp.org/db/journals/pvldb/pvldb{volume}.html"]


def pacmmod_urls(year: int) -> list[str]:
    volume = year - 2022
    if volume <= 0:
        return []
    return [f"https://dblp.org/db/journals/pacmmod/pacmmod{volume}.html"]


def rss_urls(year: int) -> list[str]:
    return [f"https://dblp.org/db/conf/rss/rss{year}.html"]


def siggraph_urls(venue: str, year: int) -> list[str]:
    code = "siggrapha" if venue == "SIGGRAPH-Asia" else "siggraph"
    return [f"https://dblp.org/db/conf/{code}/{code}{year}.html"]


def run_2020plus_special_backfill(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[DblpJob, list[str], bool, str]] = []
    for year in range(2020, 2026):
        jobs.append((DblpJob("MICCAI", year, "medical_ai", ("miccai",), ("miccai",)), miccai_urls(year), True, "DBLP + OpenAlex"))
        jobs.append((DblpJob("VLDB", year, "database", ("pvldb",), ("pvldb",)), pvldb_urls(year), True, "DBLP/PVLDB + OpenAlex"))
    for year in range(2023, 2026):
        jobs.append((DblpJob("SIGMOD", year, "database", ("pacmmod",), ("pacmmod",)), pacmmod_urls(year), True, "DBLP/PACMMOD + OpenAlex"))
    jobs.extend(
        [
            (DblpJob("CHI", 2021, "hci", ("chi",), ("chi",)), ["https://dblp.org/db/conf/chi/chi2021.html"], False, "DBLP + OpenAlex"),
            (DblpJob("RSS", 2025, "robotics", ("rss",), ("rss",)), rss_urls(2025), True, "DBLP + OpenAlex"),
            (DblpJob("SIGGRAPH", 2020, "graphics", ("siggraph",), ("siggraph",)), siggraph_urls("SIGGRAPH", 2020), True, "DBLP + OpenAlex"),
            (DblpJob("SIGGRAPH", 2021, "graphics", ("siggraph",), ("siggraph",)), siggraph_urls("SIGGRAPH", 2021), True, "DBLP + OpenAlex"),
            (DblpJob("SIGGRAPH-Asia", 2020, "graphics", ("siggrapha",), ("siggrapha",)), siggraph_urls("SIGGRAPH-Asia", 2020), True, "DBLP + OpenAlex"),
        ]
    )

    for job, urls, refresh, source in jobs:
        if args.venues and job.venue.upper() not in args.venues:
            continue
        records.extend(
            crawl_dblp_urls(
                job,
                urls,
                cache_dir=args.cache_dir,
                refresh=refresh,
                enrich_openalex=not args.no_openalex_doi,
                max_workers=args.max_workers,
                source=source,
            )
        )
        time.sleep(args.polite_sleep)
    return records


def run_journal_backfill(args: argparse.Namespace, *, start_year: int, end_year: int) -> list[dict[str, Any]]:
    from crawl_metadata_expansion import load_or_crawl_journal

    records: list[dict[str, Any]] = []
    for job in build_journal_jobs(start_year, end_year):
        if args.venues and job.venue.upper() not in args.venues:
            continue
        label = f"journal_{job.venue}_{job.year}"
        path = external_cache_path(args.cache_dir, label)
        refresh = args.refresh_empty and (not path.exists() or path.stat().st_size == 0)
        for attempt in range(args.journal_retries):
            try:
                records.extend(load_or_crawl_journal(job, cache_dir=args.cache_dir, refresh=refresh))
                break
            except Exception as exc:  # noqa: BLE001
                wait = min(args.max_retry_sleep, args.retry_sleep * (attempt + 1))
                log(f"[WARN] {label}: {exc}; retry in {wait}s")
                time.sleep(wait)
        time.sleep(args.polite_sleep)
    return records


def rebuild_outputs(args: argparse.Namespace, run_name: str, extra_records: list[dict[str, Any]]) -> None:
    cache_records: list[dict[str, Any]] = []
    for path in sorted(args.cache_dir.glob("*.jsonl")):
        cache_records.extend(read_jsonl(path))
    added_records = dedupe([*cache_records, *extra_records])
    merged_records = merge_with_base(args.base_jsonl, added_records) if args.base_jsonl.exists() else []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    added_jsonl = args.output_dir / f"{run_name}_added.jsonl"
    added_csv = args.output_dir / f"{run_name}_added.csv"
    added_summary = args.output_dir / f"{run_name}_added_summary.csv"
    missing_csv = args.output_dir / f"{run_name}_missing_abstracts.csv"
    write_jsonl(added_jsonl, added_records)
    write_csv(added_csv, added_records)
    write_summary(added_summary, added_records)
    write_missing_abstracts(missing_csv, added_records)

    merged_jsonl = args.output_dir / f"{run_name}_merged.jsonl"
    merged_csv = args.output_dir / f"{run_name}_merged.csv"
    merged_summary = args.output_dir / f"{run_name}_merged_summary.csv"
    if merged_records:
        write_jsonl(merged_jsonl, merged_records)
        write_csv(merged_csv, merged_records)
        write_summary(merged_summary, merged_records)

    report = {
        "run_name": run_name,
        "added_records": len(added_records),
        "added_abstracts": sum(1 for record in added_records if clean_text(record.get("abstract"))),
        "added_missing_abstracts": sum(1 for record in added_records if not clean_text(record.get("abstract"))),
        "merged_records": len(merged_records),
        "base_jsonl": str(args.base_jsonl),
        "added_jsonl": str(added_jsonl),
        "added_csv": str(added_csv),
        "added_summary": str(added_summary),
        "missing_abstracts_csv": str(missing_csv),
        "merged_jsonl": str(merged_jsonl) if merged_records else "",
        "merged_csv": str(merged_csv) if merged_records else "",
        "merged_summary": str(merged_summary) if merged_records else "",
        "updated_at": now_iso(),
    }
    (args.output_dir / f"{run_name}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[REBUILT] {run_name}: added={len(added_records):,}, merged={len(merged_records):,}")


def run_full_cache_first_crawler(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "crawl_metadata_expansion.py"),
        "--output-dir",
        str(args.output_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--base-jsonl",
        str(args.base_jsonl),
        "--run-name",
        args.final_run_name,
        "--start-year",
        str(args.full_start_year),
        "--end-year",
        str(args.full_end_year),
        "--max-workers",
        str(args.max_workers),
    ]
    if args.no_openalex_doi:
        command.append("--no-openalex-doi")
    if args.no_detail_abstracts:
        command.append("--no-detail-abstracts")
    log("[RUN] " + " ".join(command))
    subprocess.run(command, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "metadata_expansion")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "metadata_expansion" / "cache")
    parser.add_argument(
        "--base-jsonl",
        type=Path,
        default=PROJECT_ROOT.parent
        / "AI-Paper-Trends-main"
        / "data"
        / "recent_conferences"
        / "main_accepted_ai_conference_papers_2020_2026_v1.jsonl",
    )
    parser.add_argument("--venues", nargs="*", help="Optional venue filter for debugging.")
    parser.add_argument("--phase1-run-name", default="expanded_paper_metadata_2020plus_backfill")
    parser.add_argument("--final-run-name", default="expanded_paper_metadata_1969_2026")
    parser.add_argument("--full-start-year", type=int, default=1969)
    parser.add_argument("--full-end-year", type=int, default=2026)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--polite-sleep", type=float, default=2.0)
    parser.add_argument("--retry-sleep", type=int, default=45)
    parser.add_argument("--max-retry-sleep", type=int, default=180)
    parser.add_argument("--journal-retries", type=int, default=4)
    parser.add_argument("--refresh-empty", action="store_true", default=True)
    parser.add_argument("--no-openalex-doi", action="store_true")
    parser.add_argument("--no-detail-abstracts", action="store_true")
    parser.add_argument("--skip-special-2020plus", action="store_true")
    parser.add_argument("--skip-journals-2020plus", action="store_true")
    parser.add_argument("--skip-full-cache-first", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.venues:
        args.venues = {venue.upper() for venue in args.venues}

    log("Starting metadata-only overnight backfill.")
    phase1_records: list[dict[str, Any]] = []
    if not args.skip_special_2020plus:
        phase1_records.extend(run_2020plus_special_backfill(args))
    if not args.skip_journals_2020plus:
        phase1_records.extend(run_journal_backfill(args, start_year=2020, end_year=2026))
    rebuild_outputs(args, args.phase1_run_name, phase1_records)

    if not args.skip_full_cache_first:
        run_full_cache_first_crawler(args)
        rebuild_outputs(args, args.final_run_name, [])

    log("Metadata-only overnight backfill finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
