"""Fill missing ACM Multimedia abstracts from title-matched public metadata.

The first ACMMM crawler enriches abstracts by ACM DOI. Some recent ACM records
exist in OpenAlex without an abstract, while an exact-title arXiv/preprint
version often has one. This script only fills currently empty abstracts and
rebuilds the merged v8 dataset.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests
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


DEFAULT_TIMEOUT = 45
ARXIV_HEADERS = {"User-Agent": "AI-Paper-Trends/0.1 (mailto:research@example.com)"}
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def normalize_title(value: str) -> str:
    value = clean_text(value).lower()
    value = value.replace("&amp;", "and").replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def title_similarity(left: str, right: str) -> float:
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def openalex_search_title(title: str, *, retries: int = 3) -> list[dict[str, Any]]:
    url = (
        "https://api.openalex.org/works"
        f"?search={quote(title)}"
        "&per-page=10"
        "&select=id,title,doi,publication_year,abstract_inverted_index,open_access,primary_location"
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=OPENALEX_HEADERS, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenAlex title search failed for {title!r}: {last_error}")


def candidate_score(record: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, int, int]:
    abstract = abstract_from_inverted_index(candidate.get("abstract_inverted_index"))
    if not abstract:
        return (0.0, 0, 0)

    similarity = title_similarity(record.get("title", ""), candidate.get("title", ""))
    if similarity < 0.965:
        return (0.0, 0, 0)

    record_year = int(record.get("year") or 0)
    candidate_year = int(candidate.get("publication_year") or 0)
    year_ok = 1 if not candidate_year or not record_year or abs(candidate_year - record_year) <= 2 else 0

    doi = clean_text(candidate.get("doi")).lower()
    preprint_bonus = 1 if "10.48550/arxiv" in doi or "arxiv" in doi else 0
    return (similarity, year_ok, preprint_bonus)


def fill_one(record: dict[str, Any]) -> dict[str, Any]:
    title = clean_text(record.get("title"))
    if not title:
        return {}

    candidates = openalex_search_title(title)
    best: dict[str, Any] | None = None
    best_score = (0.0, 0, 0)
    for candidate in candidates:
        score = candidate_score(record, candidate)
        if score > best_score:
            best = candidate
            best_score = score

    if not best:
        return {}

    abstract = abstract_from_inverted_index(best.get("abstract_inverted_index"))
    if not abstract:
        return {}

    updates: dict[str, Any] = {
        "abstract": abstract,
        "abstract_source": "OpenAlex title-match",
        "abstract_enriched_at": datetime.now(timezone.utc).isoformat(),
        "abstract_match_openalex_id": clean_text(best.get("id")),
        "abstract_match_title": clean_text(best.get("title")),
        "abstract_match_doi": clean_text(best.get("doi")),
        "abstract_match_similarity": round(best_score[0], 4),
    }

    open_access = best.get("open_access") or {}
    oa_url = clean_text(open_access.get("oa_url"))
    if oa_url:
        updates["open_access_url"] = oa_url
        if oa_url.lower().split("?", 1)[0].endswith(".pdf"):
            updates["pdf_url"] = oa_url

    primary = best.get("primary_location") or {}
    pdf_url = clean_text(primary.get("pdf_url"))
    if pdf_url:
        updates["pdf_url"] = pdf_url
    return updates


def arxiv_phrase(title: str) -> str:
    title = clean_text(title)
    prefix = clean_text(title.split(":", 1)[0])
    if 1 <= len(prefix.split()) <= 7 and len(prefix) <= 80:
        return prefix

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", title)
    stopwords = {"a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "via", "with"}
    selected: list[str] = []
    for word in words:
        if word.lower() in stopwords and selected:
            continue
        selected.append(word)
        if len(selected) >= 6:
            break
    return " ".join(selected) or title[:80]


def arxiv_search_title(title: str) -> list[dict[str, str]]:
    phrase = arxiv_phrase(title)
    query = f'ti:"{phrase}"'
    url = f"https://export.arxiv.org/api/query?search_query={quote(query)}&start=0&max_results=5"
    response = requests.get(url, headers=ARXIV_HEADERS, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    results: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        entry_title = clean_text(entry.findtext("atom:title", default="", namespaces=ARXIV_NS))
        summary = clean_text(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS))
        entry_id = clean_text(entry.findtext("atom:id", default="", namespaces=ARXIV_NS))
        published = clean_text(entry.findtext("atom:published", default="", namespaces=ARXIV_NS))
        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = clean_text(link.attrib.get("href"))
                break
        results.append(
            {
                "title": html.unescape(entry_title),
                "abstract": html.unescape(summary),
                "arxiv_id": entry_id.rsplit("/", 1)[-1],
                "url": entry_id,
                "published": published,
                "pdf_url": pdf_url,
            }
        )
    return results


def fill_one_arxiv(record: dict[str, Any]) -> dict[str, Any]:
    title = clean_text(record.get("title"))
    if not title:
        return {}

    best: dict[str, str] | None = None
    best_similarity = 0.0
    for candidate in arxiv_search_title(title):
        abstract = clean_text(candidate.get("abstract"))
        if not abstract:
            continue
        similarity = title_similarity(title, candidate.get("title", ""))
        if similarity > best_similarity:
            best = candidate
            best_similarity = similarity

    if not best or best_similarity < 0.88:
        return {}

    updates: dict[str, Any] = {
        "abstract": clean_text(best.get("abstract")),
        "abstract_source": "arXiv title-match",
        "abstract_enriched_at": datetime.now(timezone.utc).isoformat(),
        "abstract_match_arxiv_id": clean_text(best.get("arxiv_id")),
        "abstract_match_title": clean_text(best.get("title")),
        "abstract_match_url": clean_text(best.get("url")),
        "abstract_match_similarity": round(best_similarity, 4),
    }
    if clean_text(best.get("url")):
        updates["open_access_url"] = clean_text(best.get("url"))
    if clean_text(best.get("pdf_url")):
        updates["pdf_url"] = clean_text(best.get("pdf_url"))
    return updates


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_missing_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = ["year", "doi", "title", "html_url", "dblp_url"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing ACMMM abstracts and rebuild the merged v8 dataset.")
    parser.add_argument("--acmmm-jsonl", default="data/recent_conferences/acmmm_2020_2025_papers.jsonl")
    parser.add_argument("--base-jsonl", default="data/recent_conferences/recent_ai_conference_papers_v7.jsonl")
    parser.add_argument("--output-prefix", default="recent_ai_conference_papers_v8_with_acmmm")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit for missing abstracts.")
    parser.add_argument("--skip-openalex", action="store_true", help="Skip OpenAlex title-match enrichment.")
    parser.add_argument("--with-arxiv", action="store_true", help="Also try arXiv title-match enrichment for leftovers.")
    parser.add_argument("--arxiv-delay", type=float, default=3.1, help="Delay between arXiv API calls.")
    parser.add_argument("--arxiv-limit", type=int, default=0, help="Optional limit for arXiv leftover attempts.")
    args = parser.parse_args()

    acmmm_path = Path(args.acmmm_jsonl)
    output_dir = acmmm_path.parent
    records = [json.loads(line) for line in acmmm_path.open(encoding="utf-8")]
    target_indices = [index for index, record in enumerate(records) if not clean_text(record.get("abstract"))]
    if args.limit:
        target_indices = target_indices[: args.limit]

    before_missing = sum(1 for record in records if not clean_text(record.get("abstract")))
    filled_openalex = 0
    filled_arxiv = 0
    failures = 0

    if not args.skip_openalex and target_indices:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(fill_one, records[index]): index for index in target_indices}
            for future in tqdm(as_completed(futures), total=len(futures), desc="OpenAlex title-match"):
                index = futures[future]
                try:
                    updates = future.result()
                except Exception:
                    failures += 1
                    continue
                if not updates.get("abstract"):
                    continue
                records[index].update(updates)
                filled_openalex += 1

    if args.with_arxiv:
        arxiv_indices = [index for index, record in enumerate(records) if not clean_text(record.get("abstract"))]
        if args.arxiv_limit:
            arxiv_indices = arxiv_indices[: args.arxiv_limit]
        for index in tqdm(arxiv_indices, desc="arXiv title-match"):
            try:
                updates = fill_one_arxiv(records[index])
            except Exception:
                failures += 1
                updates = {}
            if updates.get("abstract"):
                records[index].update(updates)
                filled_arxiv += 1
            time.sleep(max(0.0, args.arxiv_delay))

    after_missing = sum(1 for record in records if not clean_text(record.get("abstract")))
    missing_records = [record for record in records if not clean_text(record.get("abstract"))]

    write_jsonl(acmmm_path, records)
    write_csv(output_dir / "acmmm_2020_2025_papers.csv", records)
    write_summary(output_dir / "acmmm_2020_2025_summary.csv", records)
    write_missing_csv(output_dir / "acmmm_2020_2025_missing_abstracts.csv", missing_records)

    merged = merge_records(Path(args.base_jsonl), records)
    merged_jsonl = output_dir / f"{args.output_prefix}.jsonl"
    merged_csv = output_dir / f"{args.output_prefix}.csv"
    write_jsonl(merged_jsonl, merged)
    write_csv(merged_csv, merged)
    write_summary(output_dir / f"{args.output_prefix}_summary.csv", merged)

    report = {
        "before_missing": before_missing,
        "attempted": len(target_indices),
        "filled": filled_openalex + filled_arxiv,
        "filled_openalex_title_match": filled_openalex,
        "filled_arxiv_title_match": filled_arxiv,
        "failures": failures,
        "after_missing": after_missing,
        "acmmm_records": len(records),
        "merged_records": len(merged),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_report(output_dir / "acmmm_missing_abstract_enrichment_report.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"ACMMM JSONL: {acmmm_path}")
    print(f"Merged JSONL: {merged_jsonl}")


if __name__ == "__main__":
    main()
