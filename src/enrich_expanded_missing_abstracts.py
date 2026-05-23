"""Fill missing abstracts for expanded venues using Semantic Scholar.

This script only fills currently empty abstracts. It keeps the existing
metadata and rebuilds the merged v9 dataset from the enriched added-venue file.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.crawl_acmmm import clean_text, merge_records, write_csv, write_jsonl, write_summary
from src.enrich_acmmm_missing_abstracts import title_similarity


S2_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
    "?fields=title,abstract,externalIds,year,url,openAccessPdf"
)
HEADERS = {"User-Agent": "AI-Paper-Trends/0.1 (mailto:research@example.com)"}


def semantic_scholar_batch(ids: list[str], *, retries: int = 8) -> list[dict[str, Any] | None]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(S2_BATCH_URL, json={"ids": ids}, headers=HEADERS, timeout=90)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = int(retry_after) if retry_after and retry_after.isdigit() else min(180, 20 * (attempt + 1))
                time.sleep(sleep_seconds)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(180, 10 * (attempt + 1)))
    raise RuntimeError(f"Semantic Scholar batch failed: {last_error}")


def update_from_s2(record: dict[str, Any], item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    abstract = clean_text(item.get("abstract"))
    if not abstract:
        return False

    item_title = clean_text(item.get("title"))
    if item_title and title_similarity(record.get("title", ""), item_title) < 0.72:
        external_ids = item.get("externalIds") or {}
        record_doi = clean_text(record.get("doi")).lower()
        item_doi = clean_text(external_ids.get("DOI")).lower()
        if record_doi and item_doi and record_doi != item_doi:
            return False

    record["abstract"] = abstract
    record["abstract_source"] = "Semantic Scholar"
    record["abstract_enriched_at"] = datetime.now(timezone.utc).isoformat()
    record["semantic_scholar_id"] = clean_text(item.get("paperId"))
    record["semantic_scholar_url"] = clean_text(item.get("url"))
    record["abstract_match_title"] = item_title
    if item_title:
        record["abstract_match_similarity"] = round(title_similarity(record.get("title", ""), item_title), 4)
    open_access_pdf = item.get("openAccessPdf") or {}
    pdf_url = clean_text(open_access_pdf.get("url"))
    if pdf_url and not clean_text(record.get("pdf_url")):
        record["pdf_url"] = pdf_url
    return True


def write_missing_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = ["venue", "year", "doi", "title", "track", "html_url", "openreview_url"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if clean_text(record.get("abstract")):
                continue
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich expanded venue missing abstracts with Semantic Scholar.")
    parser.add_argument(
        "--added-jsonl",
        default="data/recent_conferences/expanded_venues_kdd_sigir_www_chi_colm_2023_2025_papers.jsonl",
    )
    parser.add_argument("--base-jsonl", default="data/recent_conferences/recent_ai_conference_papers_v8_with_acmmm.jsonl")
    parser.add_argument("--output-prefix", default="recent_ai_conference_papers_v9_with_kdd_sigir_www_chi_colm")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    added_path = Path(args.added_jsonl)
    output_dir = added_path.parent
    records = [json.loads(line) for line in added_path.open(encoding="utf-8")]
    missing_indices = [
        index
        for index, record in enumerate(records)
        if not clean_text(record.get("abstract")) and clean_text(record.get("doi"))
    ]
    if args.limit:
        missing_indices = missing_indices[: args.limit]

    before = sum(1 for record in records if clean_text(record.get("abstract")))
    filled = 0
    attempted = 0
    errors = 0

    for start in tqdm(range(0, len(missing_indices), args.batch_size), desc="Semantic Scholar batches"):
        batch_indices = missing_indices[start : start + args.batch_size]
        ids = [f"DOI:{records[index]['doi']}" for index in batch_indices]
        attempted += len(ids)
        try:
            items = semantic_scholar_batch(ids)
        except Exception:
            errors += len(ids)
            time.sleep(max(args.sleep, 5.0))
            continue
        for index, item in zip(batch_indices, items):
            if update_from_s2(records[index], item):
                filled += 1
        time.sleep(args.sleep)

    added_csv = output_dir / added_path.name.replace(".jsonl", ".csv")
    summary_path = output_dir / added_path.name.replace("_papers.jsonl", "_summary.csv")
    missing_path = output_dir / added_path.name.replace("_papers.jsonl", "_missing_abstracts.csv")
    write_jsonl(added_path, records)
    write_csv(added_csv, records)
    write_summary(summary_path, records)
    write_missing_csv(missing_path, records)

    merged = merge_records(Path(args.base_jsonl), records)
    merged_jsonl = output_dir / f"{args.output_prefix}.jsonl"
    merged_csv = output_dir / f"{args.output_prefix}.csv"
    merged_summary = output_dir / f"{args.output_prefix}_summary.csv"
    write_jsonl(merged_jsonl, merged)
    write_csv(merged_csv, merged)
    write_summary(merged_summary, merged)

    after = sum(1 for record in records if clean_text(record.get("abstract")))
    report = {
        "records": len(records),
        "before_abstracts": before,
        "after_abstracts": after,
        "filled": filled,
        "attempted": attempted,
        "errors": errors,
        "missing_abstracts": len(records) - after,
        "merged_records": len(merged),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / added_path.name.replace("_papers.jsonl", "_semantic_scholar_enrichment_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Added JSONL: {added_path}")
    print(f"Merged JSONL: {merged_jsonl}")


if __name__ == "__main__":
    main()
