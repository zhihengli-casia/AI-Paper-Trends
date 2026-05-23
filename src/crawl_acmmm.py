"""Crawl ACM Multimedia accepted-paper metadata and merge it into the dataset.

Primary source:
- DBLP proceedings pages for title/authors/DOI/session.

Enrichment source:
- OpenAlex by DOI for abstract and open-access metadata.

ACM DL itself often serves anti-bot challenge pages to scripts, so this crawler
uses public bibliographic metadata rather than scraping ACM DL pages directly.
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
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


USER_AGENT = "AI-Paper-Trends/0.1 (+public metadata research)"
HEADERS = {"User-Agent": USER_AGENT}
OPENALEX_HEADERS = {
    "User-Agent": "AI-Paper-Trends/0.1 (mailto:research@example.com)",
}
DEFAULT_TIMEOUT = 60


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def http_get(url: str, *, headers: dict[str, str] | None = None, retries: int = 3) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers or HEADERS, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, word_positions in index.items():
        for position in word_positions:
            positions.append((int(position), word))
    return " ".join(word for _, word in sorted(positions))


def acmmm_regular_section(year: int, section: str) -> bool:
    section = clean_text(section)
    if year == 2025:
        return section.startswith(("Content:", "Engagement:", "Experience:", "Generative AI:", "Systems:"))
    if year in {2024, 2023, 2022}:
        return section.startswith(("Oral Session", "Poster Session"))
    if year == 2021:
        if section.startswith("Poster Session"):
            return True
        if not section.startswith("Session"):
            return False
        excluded = (
            "Brave New Idea",
            "Industrial Track",
            "Video Program and Demo Session",
            "Doctoral Symposium",
            "Open Source Competition",
            "Multimedia Grand Challenge",
        )
        return not any(token in section for token in excluded)
    if year == 2020:
        # One 2020 poster heading is missing the "Poster" word in DBLP but has
        # the same regular-paper session format.
        return section.startswith(("Oral Session", "Poster Session", "Session H2:"))
    return False


def parse_dblp_year(year: int) -> list[dict[str, Any]]:
    source_url = f"https://dblp.org/db/conf/mm/mm{year}.html"
    soup = BeautifulSoup(http_get(source_url).text, "lxml")
    records: list[dict[str, Any]] = []

    for entry in soup.select("li.entry.inproceedings"):
        section_el = entry.find_previous(["h2", "h3"])
        section = clean_text(section_el.get_text(" ", strip=True) if section_el else "")
        if not acmmm_regular_section(year, section):
            continue

        cite = entry.select_one("cite")
        title_el = entry.select_one("cite .title")
        if not cite or not title_el:
            continue

        title = clean_text(title_el.get_text(" ", strip=True)).rstrip(".")
        authors = [
            clean_text(author.get_text(" ", strip=True))
            for author in cite.select('span[itemprop="author"] span[itemprop="name"]')
        ]

        doi = ""
        html_url = ""
        for link in entry.select('nav.publ a[href^="https://doi.org/"]'):
            doi = link["href"].split("https://doi.org/", 1)[1]
            html_url = link["href"]
            break

        dblp_url = ""
        detail_link = entry.select_one('li.details a[href], a[href*="dblp.org/rec/conf/mm/"][href$=".html"]')
        if detail_link:
            dblp_url = detail_link["href"]

        records.append(
            {
                "paper_id": doi or dblp_url or f"ACMMM-{year}-{title}",
                "domain": "multimedia",
                "venue": "ACMMM",
                "year": year,
                "source": "DBLP + OpenAlex",
                "status": "accepted",
                "track": section,
                "title": title,
                "authors": authors,
                "abstract": "",
                "keywords": [],
                "html_url": html_url,
                "pdf_url": "",
                "openreview_url": "",
                "doi": doi,
                "source_url": source_url,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "dblp_url": dblp_url,
            }
        )
    return records


def enrich_one_openalex(record: dict[str, Any]) -> dict[str, Any]:
    doi = clean_text(record.get("doi"))
    if not doi:
        return {}
    url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
    response = http_get(url, headers=OPENALEX_HEADERS, retries=2)
    data = response.json()

    updates: dict[str, Any] = {
        "openalex_id": data.get("id", ""),
        "abstract": abstract_from_inverted_index(data.get("abstract_inverted_index")),
        "abstract_source": "OpenAlex" if data.get("abstract_inverted_index") else "",
        "abstract_enriched_at": datetime.now(timezone.utc).isoformat() if data.get("abstract_inverted_index") else "",
    }

    open_access = data.get("open_access") or {}
    oa_url = clean_text(open_access.get("oa_url"))
    if oa_url:
        updates["open_access_url"] = oa_url
        if oa_url.lower().split("?", 1)[0].endswith(".pdf"):
            updates["pdf_url"] = oa_url

    primary = data.get("primary_location") or {}
    landing = clean_text(primary.get("landing_page_url"))
    if landing and not record.get("html_url"):
        updates["html_url"] = landing
    pdf_url = clean_text(primary.get("pdf_url"))
    if pdf_url:
        updates["pdf_url"] = pdf_url

    return updates


def enrich_openalex(records: list[dict[str, Any]], *, max_workers: int, limit: int = 0) -> list[dict[str, Any]]:
    target_indices = [index for index, record in enumerate(records) if record.get("doi")]
    if limit:
        target_indices = target_indices[:limit]
    if not target_indices:
        return records

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(enrich_one_openalex, records[index]): index for index in target_indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc="OpenAlex abstracts"):
            index = futures[future]
            try:
                updates = future.result()
            except Exception:
                continue
            for key, value in updates.items():
                if value:
                    records[index][key] = value
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def merge_records(base_path: Path, acmmm_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    with base_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            merged.append(record)
            seen_keys.add((record.get("venue", ""), int(record.get("year", 0)), clean_text(record.get("title", "")).lower()))

    for record in acmmm_records:
        key = (record.get("venue", ""), int(record.get("year", 0)), clean_text(record.get("title", "")).lower())
        if key in seen_keys:
            continue
        merged.append(record)
        seen_keys.add(key)
    return merged


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    summary: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record["venue"], int(record["year"]))
        row = summary.setdefault(
            key,
            {"venue": record["venue"], "year": int(record["year"]), "count": 0, "abstracts": 0, "doi": 0},
        )
        row["count"] += 1
        row["abstracts"] += 1 if clean_text(record.get("abstract")) else 0
        row["doi"] += 1 if clean_text(record.get("doi")) else 0

    rows = sorted(summary.values(), key=lambda item: (item["venue"], item["year"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["venue", "year", "count", "abstracts", "doi"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl ACM Multimedia metadata and merge it with the AI conference dataset.")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--output-dir", default="data/recent_conferences")
    parser.add_argument("--base-jsonl", default="data/recent_conferences/recent_ai_conference_papers_v7.jsonl")
    parser.add_argument("--output-prefix", default="recent_ai_conference_papers_v8_with_acmmm")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-openalex", action="store_true")
    parser.add_argument("--openalex-limit", type=int, default=0, help="Optional smoke-test limit for OpenAlex enrichment.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for year in args.years:
        year_records = parse_dblp_year(year)
        print(f"ACMMM {year}: {len(year_records)} records from DBLP")
        records.extend(year_records)

    if not args.no_openalex:
        records = enrich_openalex(records, max_workers=args.max_workers, limit=args.openalex_limit)

    acmmm_jsonl = output_dir / "acmmm_2020_2025_papers.jsonl"
    acmmm_csv = output_dir / "acmmm_2020_2025_papers.csv"
    write_jsonl(acmmm_jsonl, records)
    write_csv(acmmm_csv, records)
    write_summary(output_dir / "acmmm_2020_2025_summary.csv", records)

    merged = merge_records(Path(args.base_jsonl), records)
    merged_jsonl = output_dir / f"{args.output_prefix}.jsonl"
    merged_csv = output_dir / f"{args.output_prefix}.csv"
    write_jsonl(merged_jsonl, merged)
    write_csv(merged_csv, merged)
    write_summary(output_dir / f"{args.output_prefix}_summary.csv", merged)

    print(f"ACMMM records: {len(records):,}")
    print(f"Merged records: {len(merged):,}")
    print(f"ACMMM JSONL: {acmmm_jsonl}")
    print(f"Merged JSONL: {merged_jsonl}")


if __name__ == "__main__":
    main()
