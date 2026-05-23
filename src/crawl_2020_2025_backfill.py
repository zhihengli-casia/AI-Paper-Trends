"""Backfill 2020-2025 paper metadata for long-horizon trend analysis.

This script fills older years that were missing from the recent-cycle crawl and
merges them into the existing v9 full dataset. It writes per venue-year cache
files so interrupted runs can resume without discarding completed work.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawl_acmmm import merge_records, write_csv, write_jsonl, write_summary
from src.crawl_expanded_venues import (
    apply_updates,
    dedupe,
    openalex_by_doi,
    openalex_title_match,
    parse_dblp_venue_year,
)
from src.crawl_recent_conferences import (
    CrawlJob,
    clean_text,
    crawl_aaai,
    crawl_acl,
    crawl_cvf,
    crawl_ijcai,
    crawl_neurips,
    crawl_pmlr,
    http_get,
    make_record,
    soup_from_url,
    split_authors,
)


DEFAULT_BASE = "data/recent_conferences/recent_ai_conference_papers_v9_with_kdd_sigir_www_chi_colm.jsonl"
DEFAULT_OUTPUT_PREFIX = "recent_ai_conference_papers_v10_2020_2025_backfilled"
DEFAULT_ADDED_PREFIX = "backfill_2020_2025"
DEFAULT_CACHE_DIR = "data/recent_conferences/backfill_2020_2025_cache"

HEADERS = {"User-Agent": "AI-Paper-Trends/0.1 (+public metadata research)"}

ICML_VOLUMES = {
    2020: "v119",
    2021: "v139",
    2022: "v162",
}

CVPR_2020_DAYS = ["2020-06-16", "2020-06-17", "2020-06-18"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cache_path(cache_dir: Path, label: str) -> Path:
    return cache_dir / f"{label}.jsonl"


def write_cache(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp_path, records)
    tmp_path.replace(path)


def cvpr_2020_source_urls() -> list[str]:
    return [f"https://openaccess.thecvf.com/CVPR2020?day={day}" for day in CVPR_2020_DAYS]


def parse_cvf_listing(source_url: str, job: CrawlJob) -> list[dict[str, Any]]:
    soup = soup_from_url(source_url)
    records: list[dict[str, Any]] = []
    for dt in soup.select("dt.ptitle"):
        title_link = dt.select_one("a[href]")
        if not title_link:
            continue
        html_url = urljoin(source_url, title_link["href"])
        dds = dt.find_next_siblings("dd", limit=2)
        authors = split_authors(dds[0].get_text(" ", strip=True) if dds else "")
        pdf_url = ""
        if len(dds) > 1:
            pdf_link = dds[1].select_one("a[href$='.pdf']")
            if pdf_link:
                pdf_url = urljoin(source_url, pdf_link["href"])
        records.append(
            make_record(
                domain=job.domain,
                venue=job.venue,
                year=job.year,
                source=job.source,
                status="accepted",
                title=title_link.get_text(" ", strip=True),
                authors=authors,
                html_url=html_url,
                pdf_url=pdf_url,
                source_url=source_url,
            )
        )
    return records


def fetch_cvf_abstract(record: dict[str, Any]) -> dict[str, str]:
    page = soup_from_url(record["html_url"])
    abstract = page.select_one("#abstract")
    return {"abstract": abstract.get_text(" ", strip=True) if abstract else ""}


def enrich_records_with_fetcher(
    records: list[dict[str, Any]],
    fetcher,
    *,
    max_workers: int,
    desc: str,
) -> list[dict[str, Any]]:
    target_indices = [index for index, record in enumerate(records) if not clean_text(record.get("abstract"))]
    if not target_indices:
        return records
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetcher, records[index]): index for index in target_indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            index = futures[future]
            try:
                updates = future.result()
            except Exception:
                continue
            for key, value in updates.items():
                if value:
                    records[index][key] = value
    return records


def crawl_cvpr_2020(fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    job = CrawlJob("cv", "CVPR", 2020, "CVF Open Access", "cvf", {"conf": "CVPR"})
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_url in cvpr_2020_source_urls():
        for record in parse_cvf_listing(source_url, job):
            key = record.get("html_url") or record.get("title", "")
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    if fetch_detail_abstracts:
        records = enrich_records_with_fetcher(records, fetch_cvf_abstract, max_workers=max_workers, desc="CVPR 2020 abstracts")
    return records


def old_openreview_get(params: dict[str, Any], *, retries: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get("https://api.openreview.net/notes", params=params, headers=HEADERS, timeout=90)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"OpenReview v1 request failed: {last_error}")


def content_value(content: dict[str, Any], key: str, default: Any = "") -> Any:
    return content.get(key, default)


def make_iclr_record(note: dict[str, Any], year: int, status: str) -> dict[str, Any]:
    content = note.get("content", {})
    title = clean_text(content_value(content, "title"))
    authors = content_value(content, "authors", [])
    if not isinstance(authors, list):
        authors = []
    keywords = content_value(content, "keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    pdf_field = clean_text(content_value(content, "pdf"))
    forum = note.get("forum") or note.get("id") or ""
    forum_url = f"https://openreview.net/forum?id={forum}" if forum else ""
    pdf_url = urljoin("https://openreview.net", pdf_field) if pdf_field.startswith("/") else pdf_field
    record = make_record(
        domain="ml",
        venue="ICLR",
        year=year,
        source="OpenReview",
        status=status,
        title=title,
        authors=authors,
        abstract=content_value(content, "abstract"),
        keywords=keywords,
        track=content_value(content, "primary_area", "") or content_value(content, "subject_areas", ""),
        paper_id=note.get("id") or forum or f"ICLR-{year}-{title}",
        html_url=forum_url,
        pdf_url=pdf_url,
        source_url=f"https://openreview.net/group?id=ICLR.cc/{year}/Conference",
        openreview_url=forum_url,
    )
    if content.get("venueid"):
        record["venueid"] = content.get("venueid")
    if content.get("_bibtex"):
        record["_bibtex"] = content.get("_bibtex")
    return record


def crawl_iclr_2020(max_workers: int) -> list[dict[str, Any]]:
    """ICLR 2020 accepted status is only available in decision replies."""

    del max_workers  # kept for a uniform call signature
    records: list[dict[str, Any]] = []
    offset = 0
    invitation = "ICLR.cc/2020/Conference/-/Blind_Submission"
    while True:
        data = old_openreview_get(
            {
                "invitation": invitation,
                "limit": 100,
                "offset": offset,
                "details": "directReplies",
            }
        )
        notes = data.get("notes", [])
        if not notes:
            break
        for note in tqdm(notes, desc=f"ICLR 2020 decisions offset={offset}", leave=False):
            decision = ""
            for reply in note.get("details", {}).get("directReplies", []):
                if str(reply.get("invitation", "")).endswith("/-/Decision"):
                    decision = clean_text((reply.get("content") or {}).get("decision"))
                    break
            if "accept" not in decision.lower():
                continue
            records.append(make_iclr_record(note, 2020, decision or "accepted"))
        if len(notes) < 100:
            break
        offset += 100
    return records


def crawl_iclr_accepted_year(year: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        data = old_openreview_get(
            {
                "content.venueid": f"ICLR.cc/{year}/Conference",
                "limit": 1000,
                "offset": offset,
            }
        )
        notes = data.get("notes", [])
        if not notes:
            break
        for note in notes:
            content = note.get("content", {})
            status = clean_text(content.get("venue"))
            status_lower = status.lower()
            if not status or "submitted" in status_lower or "withdrawn" in status_lower:
                continue
            if not str(note.get("invitation", "")).startswith(f"ICLR.cc/{year}/Conference"):
                continue
            records.append(make_iclr_record(note, year, status))
        if len(notes) < 1000:
            break
        offset += 1000
    return records


def crawl_core_job(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    if job.venue == "CVPR" and job.year == 2020:
        return crawl_cvpr_2020(fetch_detail_abstracts, max_workers)
    if job.venue == "ICLR" and job.year == 2020:
        return crawl_iclr_2020(max_workers)
    if job.venue == "ICLR":
        return crawl_iclr_accepted_year(job.year)
    crawler = {
        "cvf": crawl_cvf,
        "neurips": crawl_neurips,
        "pmlr": crawl_pmlr,
        "acl": crawl_acl,
        "aaai": crawl_aaai,
        "ijcai": crawl_ijcai,
    }[job.crawler]
    return crawler(job, fetch_detail_abstracts=fetch_detail_abstracts, max_workers=max_workers)


def default_core_jobs() -> list[CrawlJob]:
    jobs: list[CrawlJob] = []
    jobs.extend(CrawlJob("cv", "CVPR", year, "CVF Open Access", "cvf", {"conf": "CVPR"}) for year in [2020, 2021, 2022])
    jobs.extend(CrawlJob("ml", "NeurIPS", year, "NeurIPS Proceedings", "neurips", {}) for year in [2020, 2021, 2022])
    jobs.extend(CrawlJob("ml", "ICLR", year, "OpenReview", "iclr_old", {}) for year in [2020, 2021, 2022, 2023])
    jobs.extend(CrawlJob("ml", "ICML", year, "PMLR", "pmlr", {"volume": ICML_VOLUMES[year]}) for year in [2020, 2021, 2022])
    jobs.extend(CrawlJob("nlp", "ACL", year, "ACL Anthology", "acl", {"event": f"acl-{year}"}) for year in [2020, 2021, 2022])
    jobs.extend(CrawlJob("nlp", "EMNLP", year, "ACL Anthology", "acl", {"event": f"emnlp-{year}"}) for year in [2020, 2021, 2022])
    jobs.append(CrawlJob("nlp", "NAACL", 2021, "ACL Anthology", "acl", {"event": "naacl-2021"}))
    jobs.extend(CrawlJob("general_ai", "AAAI", year, "AAAI OJS", "aaai", {}) for year in [2020, 2021, 2022, 2023])
    jobs.extend(CrawlJob("general_ai", "IJCAI", year, "IJCAI Proceedings", "ijcai", {}) for year in [2020, 2021, 2022])
    return jobs


def default_dblp_jobs() -> list[tuple[str, int]]:
    return [(venue, year) for venue in ["KDD", "SIGIR", "WWW"] for year in [2020, 2021, 2022]]


def should_run(label: str, venues: set[str] | None) -> bool:
    if not venues:
        return True
    return label.split("_", 1)[0].upper() in venues


def load_or_crawl_core(
    job: CrawlJob,
    *,
    cache_dir: Path,
    refresh: bool,
    fetch_detail_abstracts: bool,
    max_workers: int,
) -> list[dict[str, Any]]:
    label = f"{job.venue}_{job.year}"
    path = cache_path(cache_dir, label)
    if path.exists() and not refresh:
        records = read_jsonl(path)
        print(f"[CACHE] {label}: {len(records):,}")
        return records
    started = time.time()
    records = crawl_core_job(job, fetch_detail_abstracts=fetch_detail_abstracts, max_workers=max_workers)
    write_cache(path, records)
    print(f"[OK] {label}: {len(records):,} records in {time.time() - started:.1f}s")
    return records


def load_or_crawl_dblp(
    venue: str,
    year: int,
    *,
    cache_dir: Path,
    refresh: bool,
    max_workers: int,
    openalex: bool,
    title_match: bool,
) -> list[dict[str, Any]]:
    label = f"{venue}_{year}"
    path = cache_path(cache_dir, label)
    if path.exists() and not refresh:
        records = read_jsonl(path)
        print(f"[CACHE] {label}: {len(records):,}")
        return records
    started = time.time()
    records = parse_dblp_venue_year(venue, year)
    records = dedupe(records)
    if openalex:
        doi_indices = [index for index, record in enumerate(records) if record.get("doi")]
        filled = apply_updates(records, doi_indices, openalex_by_doi, max_workers=max_workers, desc=f"{label} OpenAlex DOI")
        print(f"{label} OpenAlex DOI filled abstracts: {filled:,}")
        if title_match:
            missing = [index for index, record in enumerate(records) if not clean_text(record.get("abstract"))]
            filled = apply_updates(
                records,
                missing,
                openalex_title_match,
                max_workers=max(2, max_workers // 2),
                desc=f"{label} OpenAlex title-match",
            )
            print(f"{label} OpenAlex title-match filled abstracts: {filled:,}")
    write_cache(path, records)
    print(f"[OK] {label}: {len(records):,} records in {time.time() - started:.1f}s")
    return records


def filter_year_range(records: list[dict[str, Any]], start_year: int, end_year: int) -> list[dict[str, Any]]:
    return [record for record in records if start_year <= int(record.get("year") or 0) <= end_year]


def write_missing_csv(path: Path, records: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = ["venue", "year", "title", "track", "status", "doi", "html_url", "openreview_url"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if clean_text(record.get("abstract")):
                continue
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill 2020-2025 conference metadata.")
    parser.add_argument("--base-jsonl", default=DEFAULT_BASE)
    parser.add_argument("--output-dir", default="data/recent_conferences")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--added-prefix", default=DEFAULT_ADDED_PREFIX)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--venues", nargs="*", help="Optional venue filter.")
    parser.add_argument("--refresh", action="store_true", help="Ignore per-year cache files.")
    parser.add_argument("--no-detail-abstracts", action="store_true")
    parser.add_argument("--no-openalex", action="store_true")
    parser.add_argument("--no-title-match", action="store_true")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    venue_filter = {venue.upper() for venue in args.venues} if args.venues else None
    records: list[dict[str, Any]] = []

    for job in default_core_jobs():
        if not should_run(job.venue, venue_filter):
            continue
        records.extend(
            load_or_crawl_core(
                job,
                cache_dir=cache_dir,
                refresh=args.refresh,
                fetch_detail_abstracts=not args.no_detail_abstracts,
                max_workers=args.max_workers,
            )
        )

    for venue, year in default_dblp_jobs():
        if not should_run(venue, venue_filter):
            continue
        records.extend(
            load_or_crawl_dblp(
                venue,
                year,
                cache_dir=cache_dir,
                refresh=args.refresh,
                max_workers=args.max_workers,
                openalex=not args.no_openalex,
                title_match=not args.no_title_match,
            )
        )

    records = dedupe(records)
    added_jsonl = output_dir / f"{args.added_prefix}_papers.jsonl"
    added_csv = output_dir / f"{args.added_prefix}_papers.csv"
    added_summary = output_dir / f"{args.added_prefix}_summary.csv"
    added_missing = output_dir / f"{args.added_prefix}_missing_abstracts.csv"
    write_jsonl(added_jsonl, records)
    write_csv(added_csv, records)
    write_summary(added_summary, records)
    write_missing_csv(added_missing, records)

    merged = merge_records(Path(args.base_jsonl), records)
    merged_2020_2025 = filter_year_range(merged, args.start_year, args.end_year)
    merged_jsonl = output_dir / f"{args.output_prefix}.jsonl"
    merged_csv = output_dir / f"{args.output_prefix}.csv"
    merged_summary = output_dir / f"{args.output_prefix}_summary.csv"
    write_jsonl(merged_jsonl, merged_2020_2025)
    write_csv(merged_csv, merged_2020_2025)
    write_summary(merged_summary, merged_2020_2025)

    report = {
        "added_records": len(records),
        "added_abstracts": sum(1 for record in records if clean_text(record.get("abstract"))),
        "added_missing_abstracts": sum(1 for record in records if not clean_text(record.get("abstract"))),
        "merged_2020_2025_records": len(merged_2020_2025),
        "base_jsonl": args.base_jsonl,
        "added_jsonl": str(added_jsonl),
        "merged_jsonl": str(merged_jsonl),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / f"{args.added_prefix}_crawl_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
