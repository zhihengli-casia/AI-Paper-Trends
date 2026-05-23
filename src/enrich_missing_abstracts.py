"""Fill missing abstracts in a conference metadata JSONL file.

This script reads an existing JSONL dataset produced by
``crawl_recent_conferences.py``, fetches paper detail pages for records whose
``abstract`` is empty, and writes an enriched JSONL/CSV pair. A JSON cache is
maintained so long runs can be resumed without re-fetching completed pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


REQUEST_HEADERS = {"User-Agent": "AI-Paper-Trends/0.1 (+public metadata research)"}


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def http_get(url: str, timeout: int = 35, retries: int = 3) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - defensive network retry
            last_error = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def soup_from_url(url: str) -> BeautifulSoup:
    return BeautifulSoup(http_get(url).text, "lxml")


def extract_abstract(record: dict[str, Any]) -> dict[str, str]:
    url = record.get("html_url") or record.get("openreview_url")
    if not url:
        return {"abstract": "", "doi": ""}

    venue = record.get("venue", "")
    source = record.get("source", "")
    page = soup_from_url(url)
    abstract = ""
    doi = ""

    if source in {"CVF Open Access", "ECVA"} or venue in {"CVPR", "ICCV", "ECCV"}:
        abstract_el = page.select_one("#abstract")
        abstract = abstract_el.get_text(" ", strip=True) if abstract_el else ""

    elif venue == "NeurIPS":
        abstract_el = page.select_one(".paper-abstract")
        abstract = abstract_el.get_text(" ", strip=True) if abstract_el else ""
        if not abstract:
            heading = page.find(["h2", "h3"], string=re.compile(r"^\s*Abstract\s*$", re.I))
            section = heading.find_parent("section") if heading else None
            if section:
                for paragraph in section.find_all("p"):
                    abstract = paragraph.get_text(" ", strip=True)
                    if abstract:
                        break
            if not abstract and heading:
                abstract_el = heading.find_next("p")
                abstract = abstract_el.get_text(" ", strip=True) if abstract_el else ""

    elif venue == "ICML" or source == "PMLR":
        abstract_el = page.select_one("#abstract") or page.select_one(".abstract")
        abstract = abstract_el.get_text(" ", strip=True) if abstract_el else ""

    elif venue in {"ACL", "EMNLP", "NAACL"} or source == "ACL Anthology":
        abstract_el = page.select_one(".acl-abstract")
        if not abstract_el:
            abstract_el = page.select_one(".card-body.acl-abstract")
        if abstract_el:
            abstract = abstract_el.get_text(" ", strip=True).removeprefix("Abstract ")

    elif venue == "AAAI" or source == "AAAI OJS":
        abstract_el = page.select_one(".item.abstract") or page.select_one("section.item.abstract")
        if abstract_el:
            abstract = abstract_el.get_text(" ", strip=True).removeprefix("Abstract ")
        doi_el = page.select_one(".item.doi a[href]") or page.select_one("meta[name='citation_doi']")
        if doi_el:
            doi = doi_el.get("content") or doi_el.get_text(" ", strip=True)

    elif venue == "IJCAI" or source == "IJCAI Proceedings":
        abstract_el = page.select_one(".proceedings-detail .row .col-md-12")
        if abstract_el:
            abstract = abstract_el.get_text(" ", strip=True)
        doi_el = page.select_one("meta[name='citation_doi']")
        if doi_el:
            doi = doi_el.get("content", "")

    return {"abstract": clean_text(abstract), "doi": clean_text(doi)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp_path.replace(path)


def write_outputs(records: list[dict[str, Any]], output_jsonl: Path, output_csv: Path, summary_csv: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for record in records for key in record.keys()})
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["authors"] = "; ".join(record.get("authors") or [])
            row["keywords"] = "; ".join(record.get("keywords") or [])
            writer.writerow(row)

    summary: dict[tuple[str, int], dict[str, int]] = {}
    for record in records:
        key = (record["venue"], int(record["year"]))
        bucket = summary.setdefault(key, {"count": 0, "abstract_count": 0})
        bucket["count"] += 1
        if record.get("abstract"):
            bucket["abstract_count"] += 1
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["venue", "year", "count", "abstract_count"])
        for (venue, year), values in sorted(summary.items()):
            writer.writerow([venue, year, values["count"], values["abstract_count"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing abstracts in a JSONL conference dataset.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--summary-csv", required=True, help="Output summary CSV path.")
    parser.add_argument("--cache", required=True, help="JSON cache path for fetched abstracts.")
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent detail page requests.")
    parser.add_argument("--venues", nargs="*", help="Optional venue filter for enrichment.")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    venue_filter = {venue.upper() for venue in args.venues} if args.venues else None

    target_indices: list[int] = []
    for index, record in enumerate(records):
        if record.get("abstract"):
            continue
        if venue_filter and record.get("venue", "").upper() not in venue_filter:
            continue
        url = record.get("html_url") or record.get("openreview_url")
        if not url:
            continue
        if url in cache and cache[url].get("abstract"):
            updates = cache[url]
            record["abstract"] = updates.get("abstract", "")
            if updates.get("doi") and not record.get("doi"):
                record["doi"] = updates["doi"]
            continue
        target_indices.append(index)

    print(f"records={len(records)} missing_targets={len(target_indices)} cache_hits={len(cache)}")

    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_index = {executor.submit(extract_abstract, records[index]): index for index in target_indices}
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Fetching abstracts"):
            index = future_to_index[future]
            record = records[index]
            url = record.get("html_url") or record.get("openreview_url")
            try:
                updates = future.result()
            except Exception as exc:
                updates = {"abstract": "", "doi": "", "error": str(exc)}
            if url:
                cache[url] = updates
            if updates.get("abstract"):
                record["abstract"] = updates["abstract"]
                record["abstract_enriched_at"] = datetime.now(timezone.utc).isoformat()
            if updates.get("doi") and not record.get("doi"):
                record["doi"] = updates["doi"]

            completed_since_save += 1
            if completed_since_save >= 250:
                save_cache(cache_path, cache)
                completed_since_save = 0

    save_cache(cache_path, cache)
    write_outputs(records, Path(args.output_jsonl), Path(args.output_csv), Path(args.summary_csv))

    abstract_count = sum(1 for record in records if record.get("abstract"))
    print(f"done records={len(records)} abstract_count={abstract_count}")


if __name__ == "__main__":
    main()
