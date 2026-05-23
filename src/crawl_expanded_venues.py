"""Crawl additional AI-adjacent conference metadata and merge it.

Targets:
- KDD: DBLP for 2023-2024, official KDD 2025 pages for main tracks.
- SIGIR, WWW, CHI: DBLP main proceedings pages for 2023-2025.
- COLM: OpenReview accepted submissions for 2024-2025.

The crawler uses public bibliographic metadata and only accepted/public records.
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

from src.crawl_acmmm import (
    OPENALEX_HEADERS,
    abstract_from_inverted_index,
    clean_text,
    merge_records,
    write_csv,
    write_jsonl,
    write_summary,
)
from src.enrich_acmmm_missing_abstracts import normalize_title, title_similarity


USER_AGENT = "AI-Paper-Trends/0.1 (+public metadata research)"
HEADERS = {"User-Agent": USER_AGENT}
DEFAULT_TIMEOUT = 60


DBLP_VENUES = {
    "KDD": {
        "code": "kdd",
        "domain": "data_mining",
        "years": [2023, 2024],
    },
    "SIGIR": {
        "code": "sigir",
        "domain": "information_retrieval",
        "years": [2023, 2024, 2025],
    },
    "WWW": {
        "code": "www",
        "domain": "web",
        "years": [2023, 2024, 2025],
    },
    "CHI": {
        "code": "chi",
        "domain": "hci",
        "years": [2023, 2024, 2025],
    },
}

KDD_2025_TRACKS = [
    (
        "Research Track Papers",
        "https://kdd2025.kdd.org/research-track-papers/",
    ),
    (
        "Applied Data Science (ADS) Track Papers",
        "https://kdd2025.kdd.org/applied-data-science-ads-track-papers/",
    ),
    (
        "Datasets and Benchmarks Track Papers",
        "https://kdd2025.kdd.org/datasets-and-benchmarks-track-papers/",
    ),
]

COLM_VENUEIDS = {
    2024: "colmweb.org/COLM/2024/Conference",
    2025: "colmweb.org/COLM/2025/Conference",
}


def http_get(url: str, *, retries: int = 6, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(min(45, 2.5 * (attempt + 1)))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def should_include_dblp(venue: str, section: str) -> bool:
    section_lower = clean_text(section).lower()
    excluded = (
        "tutorial",
        "workshop",
        "doctoral",
        "demo paper",
        "demo papers",
        "demonstration",
        "keynote",
        "invited talk",
        "special day",
        "panel",
        "front matter",
    )
    if any(token in section_lower for token in excluded):
        return False

    if venue == "KDD":
        included = (
            "research track",
            "applied data",
            "datasets",
            "benchmarks",
        )
        return any(token in section_lower for token in included)

    if venue == "WWW" and "keynote" in section_lower:
        return False

    return True


def parse_dblp_venue_year(venue: str, year: int) -> list[dict[str, Any]]:
    cfg = DBLP_VENUES[venue]
    code = cfg["code"]
    source_url = f"https://dblp.org/db/conf/{code}/{code}{year}.html"
    soup = BeautifulSoup(http_get(source_url).text, "lxml")
    records: list[dict[str, Any]] = []

    for entry in soup.select("li.entry.inproceedings"):
        section_el = entry.find_previous(["h2", "h3"])
        section = clean_text(section_el.get_text(" ", strip=True) if section_el else "")
        if not should_include_dblp(venue, section):
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
        detail_link = entry.select_one(
            f'li.details a[href], a[href*="dblp.org/rec/conf/{code}/"][href$=".html"]'
        )
        if detail_link:
            dblp_url = detail_link["href"]

        records.append(
            {
                "paper_id": doi or dblp_url or f"{venue}-{year}-{title}",
                "domain": cfg["domain"],
                "venue": venue,
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


def strip_affiliation(author: str) -> str:
    author = clean_text(author)
    author = re.sub(r"\s*\([^)]*\)", "", author)
    return clean_text(author)


def parse_kdd_2025_track(track: str, url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(http_get(url).text, "lxml")
    rows = [clean_text(row.get_text(" ", strip=True)) for row in soup.select("table tr")]
    records: list[dict[str, Any]] = []
    doi_pattern = re.compile(r"(?:https://doi\.org/)?(10\.1145/[0-9.]+)")

    index = 0
    while index < len(rows):
        title_row = rows[index]
        author_row = rows[index + 1] if index + 1 < len(rows) else ""
        index += 2
        match = doi_pattern.search(title_row)
        if not match:
            continue
        doi = match.group(1)
        title = clean_text(title_row[: match.start()]).rstrip(".")
        title = re.sub(r"\s*DOI:\s*$", "", title).strip()
        title = title.strip("\"“”")
        authors = [strip_affiliation(author) for author in author_row.split(";") if strip_affiliation(author)]

        records.append(
            {
                "paper_id": doi,
                "domain": "data_mining",
                "venue": "KDD",
                "year": 2025,
                "source": "KDD official + OpenAlex",
                "status": "accepted",
                "track": track,
                "title": title,
                "authors": authors,
                "abstract": "",
                "keywords": [],
                "html_url": f"https://doi.org/{doi}",
                "pdf_url": "",
                "openreview_url": "",
                "doi": doi,
                "source_url": url,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return records


def parse_kdd_2025() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for track, url in KDD_2025_TRACKS:
        track_records = parse_kdd_2025_track(track, url)
        print(f"KDD 2025 {track}: {len(track_records)} records")
        records.extend(track_records)
    return records


def content_value(content: dict[str, Any], key: str, default: Any = "") -> Any:
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value.get("value", default)
    return value


def parse_colm_year(year: int) -> list[dict[str, Any]]:
    venueid = COLM_VENUEIDS[year]
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = f"https://api2.openreview.net/notes?content.venueid={quote(venueid)}&limit=1000&offset={offset}"
        data = http_get(url).json()
        notes = data.get("notes", [])
        if not notes:
            break
        for note in notes:
            content = note.get("content", {})
            title = clean_text(content_value(content, "title"))
            if not title:
                continue
            authors = content_value(content, "authors", [])
            if not isinstance(authors, list):
                authors = []
            keywords = content_value(content, "keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            abstract = clean_text(content_value(content, "abstract"))
            forum = note.get("forum") or note.get("id") or ""
            openreview_url = f"https://openreview.net/forum?id={forum}" if forum else ""
            pdf_url = f"https://openreview.net/pdf?id={note.get('id')}" if note.get("id") else ""
            venue = clean_text(content_value(content, "venue")) or "COLM"

            records.append(
                {
                    "paper_id": note.get("id") or f"COLM-{year}-{title}",
                    "domain": "language_modeling",
                    "venue": "COLM",
                    "year": year,
                    "source": "OpenReview",
                    "status": "accepted",
                    "track": venue,
                    "title": title,
                    "authors": [clean_text(author) for author in authors],
                    "abstract": abstract,
                    "keywords": [clean_text(keyword) for keyword in keywords],
                    "html_url": openreview_url,
                    "pdf_url": pdf_url,
                    "openreview_url": openreview_url,
                    "doi": "",
                    "source_url": url,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "openreview_id": note.get("id", ""),
                    "openreview_forum": forum,
                    "abstract_source": "OpenReview" if abstract else "",
                }
            )
        if len(notes) < 1000:
            break
        offset += 1000
    return records


def openalex_by_doi(record: dict[str, Any]) -> dict[str, Any]:
    doi = clean_text(record.get("doi"))
    if not doi:
        return {}
    url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
    response = requests.get(url, headers=OPENALEX_HEADERS, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    updates: dict[str, Any] = {
        "openalex_id": data.get("id", ""),
    }
    abstract = abstract_from_inverted_index(data.get("abstract_inverted_index"))
    if abstract:
        updates["abstract"] = abstract
        updates["abstract_source"] = "OpenAlex"
        updates["abstract_enriched_at"] = datetime.now(timezone.utc).isoformat()
    open_access = data.get("open_access") or {}
    oa_url = clean_text(open_access.get("oa_url"))
    if oa_url:
        updates["open_access_url"] = oa_url
        if oa_url.lower().split("?", 1)[0].endswith(".pdf"):
            updates["pdf_url"] = oa_url
    primary = data.get("primary_location") or {}
    if clean_text(primary.get("landing_page_url")) and not record.get("html_url"):
        updates["html_url"] = clean_text(primary.get("landing_page_url"))
    if clean_text(primary.get("pdf_url")):
        updates["pdf_url"] = clean_text(primary.get("pdf_url"))
    return updates


def openalex_title_match(record: dict[str, Any]) -> dict[str, Any]:
    title = clean_text(record.get("title"))
    if not title:
        return {}
    url = (
        "https://api.openalex.org/works"
        f"?search={quote(title)}"
        "&per-page=5"
        "&select=id,title,doi,publication_year,abstract_inverted_index,open_access,primary_location"
    )
    response = requests.get(url, headers=OPENALEX_HEADERS, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    best: dict[str, Any] | None = None
    best_similarity = 0.0
    record_title = normalize_title(title)
    for candidate in data.get("results", []):
        candidate_title = clean_text(candidate.get("title"))
        if not candidate_title:
            continue
        similarity = title_similarity(record_title, normalize_title(candidate_title))
        abstract = abstract_from_inverted_index(candidate.get("abstract_inverted_index"))
        if abstract and similarity > best_similarity:
            best = candidate
            best_similarity = similarity

    if not best or best_similarity < 0.965:
        return {}

    updates: dict[str, Any] = {
        "abstract": abstract_from_inverted_index(best.get("abstract_inverted_index")),
        "abstract_source": "OpenAlex title-match",
        "abstract_enriched_at": datetime.now(timezone.utc).isoformat(),
        "abstract_match_openalex_id": clean_text(best.get("id")),
        "abstract_match_title": clean_text(best.get("title")),
        "abstract_match_doi": clean_text(best.get("doi")),
        "abstract_match_similarity": round(best_similarity, 4),
    }
    open_access = best.get("open_access") or {}
    oa_url = clean_text(open_access.get("oa_url"))
    if oa_url:
        updates["open_access_url"] = oa_url
        if oa_url.lower().split("?", 1)[0].endswith(".pdf"):
            updates["pdf_url"] = oa_url
    primary = best.get("primary_location") or {}
    if clean_text(primary.get("pdf_url")):
        updates["pdf_url"] = clean_text(primary.get("pdf_url"))
    return updates


def apply_updates(records: list[dict[str, Any]], target_indices: list[int], fn, *, max_workers: int, desc: str) -> int:
    filled = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, records[index]): index for index in target_indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            index = futures[future]
            try:
                updates = future.result()
            except Exception:
                continue
            had_abstract = bool(clean_text(records[index].get("abstract")))
            for key, value in updates.items():
                if value:
                    records[index][key] = value
            if not had_abstract and clean_text(records[index].get("abstract")):
                filled += 1
    return filled


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        key = (
            clean_text(record.get("venue")),
            int(record.get("year") or 0),
            clean_text(record.get("doi")).lower() or normalize_title(record.get("title", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def write_missing_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = ["venue", "year", "doi", "title", "track", "html_url", "openreview_url"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if clean_text(record.get("abstract")):
                continue
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def crawl_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for venue, cfg in DBLP_VENUES.items():
        for year in cfg["years"]:
            year_records = parse_dblp_venue_year(venue, year)
            print(f"{venue} {year}: {len(year_records)} records")
            records.extend(year_records)
    records.extend(parse_kdd_2025())
    for year in sorted(COLM_VENUEIDS):
        year_records = parse_colm_year(year)
        print(f"COLM {year}: {len(year_records)} records")
        records.extend(year_records)
    return dedupe(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl KDD/SIGIR/WWW/CHI/COLM and merge with the dataset.")
    parser.add_argument("--output-dir", default="data/recent_conferences")
    parser.add_argument("--base-jsonl", default="data/recent_conferences/recent_ai_conference_papers_v8_with_acmmm.jsonl")
    parser.add_argument("--output-prefix", default="recent_ai_conference_papers_v9_with_kdd_sigir_www_chi_colm")
    parser.add_argument("--added-prefix", default="expanded_venues_kdd_sigir_www_chi_colm_2023_2025")
    parser.add_argument(
        "--input-records-jsonl",
        default="",
        help="Optional pre-crawled expanded venue records JSONL. When set, skip network crawling and enrich/merge these records.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-openalex", action="store_true")
    parser.add_argument("--no-title-match", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_records_jsonl:
        records = [json.loads(line) for line in Path(args.input_records_jsonl).open(encoding="utf-8")]
        records = dedupe(records)
        print(f"Loaded pre-crawled expanded venue records: {len(records):,}")
    else:
        records = crawl_records()
    before_abstracts = sum(1 for record in records if clean_text(record.get("abstract")))
    print(f"Expanded venue records before enrichment: {len(records):,}")
    print(f"Abstracts before enrichment: {before_abstracts:,}")

    if not args.no_openalex:
        doi_indices = [index for index, record in enumerate(records) if record.get("doi")]
        filled = apply_updates(records, doi_indices, openalex_by_doi, max_workers=args.max_workers, desc="OpenAlex DOI")
        print(f"OpenAlex DOI filled abstracts: {filled:,}")

    if not args.no_openalex and not args.no_title_match:
        missing_indices = [index for index, record in enumerate(records) if not clean_text(record.get("abstract"))]
        filled = apply_updates(
            records,
            missing_indices,
            openalex_title_match,
            max_workers=max(2, args.max_workers // 2),
            desc="OpenAlex title-match",
        )
        print(f"OpenAlex title-match filled abstracts: {filled:,}")

    added_jsonl = output_dir / f"{args.added_prefix}_papers.jsonl"
    added_csv = output_dir / f"{args.added_prefix}_papers.csv"
    added_summary = output_dir / f"{args.added_prefix}_summary.csv"
    missing_csv = output_dir / f"{args.added_prefix}_missing_abstracts.csv"
    write_jsonl(added_jsonl, records)
    write_csv(added_csv, records)
    write_summary(added_summary, records)
    write_missing_csv(missing_csv, records)

    merged = merge_records(Path(args.base_jsonl), records)
    merged_jsonl = output_dir / f"{args.output_prefix}.jsonl"
    merged_csv = output_dir / f"{args.output_prefix}.csv"
    merged_summary = output_dir / f"{args.output_prefix}_summary.csv"
    write_jsonl(merged_jsonl, merged)
    write_csv(merged_csv, merged)
    write_summary(merged_summary, merged)

    report = {
        "records": len(records),
        "abstracts": sum(1 for record in records if clean_text(record.get("abstract"))),
        "missing_abstracts": sum(1 for record in records if not clean_text(record.get("abstract"))),
        "merged_records": len(merged),
        "added_jsonl": str(added_jsonl),
        "merged_jsonl": str(merged_jsonl),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / f"{args.added_prefix}_crawl_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
