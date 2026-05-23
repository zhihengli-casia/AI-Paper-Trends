#!/usr/bin/env python3
"""Crawl historical and expanded paper metadata without running clustering.

This script is intentionally network-bound and cache-first. It expands the
paper database with older conference years, selected new conferences, and
selected journals, then writes JSONL/CSV files for later embedding/clustering.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIBLING_MAIN_BASE = (
    PROJECT_ROOT.parent
    / "AI-Paper-Trends-main"
    / "data"
    / "recent_conferences"
    / "main_accepted_ai_conference_papers_2020_2026_v1.jsonl"
)

USER_AGENT = "AI-Paper-Trends/0.2 (+public metadata research)"
HEADERS = {"User-Agent": USER_AGENT}
OPENALEX_HEADERS = {"User-Agent": "AI-Paper-Trends/0.2 (mailto:research@example.com)"}
DEFAULT_TIMEOUT = 60


@dataclass(frozen=True)
class OfficialJob:
    venue: str
    year: int
    domain: str
    crawler: str
    source: str
    args: dict[str, Any]


@dataclass(frozen=True)
class DblpJob:
    venue: str
    year: int
    domain: str
    codes: tuple[str, ...]
    prefixes: tuple[str, ...]
    venue_type: str = "conference"


@dataclass(frozen=True)
class JournalJob:
    venue: str
    year: int
    domain: str
    issns: tuple[str, ...]
    full_name: str
    openalex_source_id: str = ""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_authors(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    if " | " in text:
        return [clean_text(item) for item in text.split("|") if clean_text(item)]
    return [clean_text(item) for item in re.split(r"\s*,\s*", text) if clean_text(item)]


def normalize_title(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return clean_text(text)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_record(
    *,
    domain: str,
    venue: str,
    year: int,
    source: str,
    status: str,
    title: str,
    authors: Iterable[str] | None = None,
    abstract: str = "",
    keywords: Iterable[str] | None = None,
    track: str = "",
    paper_id: str = "",
    html_url: str = "",
    pdf_url: str = "",
    source_url: str = "",
    openreview_url: str = "",
    doi: str = "",
    venue_type: str = "conference",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "paper_id": paper_id or doi or html_url or pdf_url or f"{venue}-{year}-{clean_text(title)}",
        "domain": domain,
        "venue": venue,
        "venue_type": venue_type,
        "year": int(year),
        "source": source,
        "status": status,
        "track": clean_text(track),
        "title": clean_text(title),
        "authors": [clean_text(author) for author in (authors or []) if clean_text(author)],
        "abstract": clean_text(abstract),
        "keywords": [clean_text(keyword) for keyword in (keywords or []) if clean_text(keyword)],
        "html_url": clean_text(html_url),
        "pdf_url": clean_text(pdf_url),
        "openreview_url": clean_text(openreview_url),
        "doi": clean_doi(doi),
        "source_url": clean_text(source_url),
        "scraped_at": now_iso(),
    }
    if extra:
        record.update(extra)
    return record


def clean_doi(value: Any) -> str:
    text = clean_text(value)
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return text.lower()


def http_get(url: str, *, timeout: int = DEFAULT_TIMEOUT, retries: int = 4) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(min(20, 2.0 * (attempt + 1)))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def soup_from_url(url: str, *, timeout: int = DEFAULT_TIMEOUT) -> BeautifulSoup:
    return BeautifulSoup(http_get(url, timeout=timeout).text, "lxml")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            if isinstance(row.get("authors"), list):
                row["authors"] = "; ".join(row["authors"])
            if isinstance(row.get("keywords"), list):
                row["keywords"] = "; ".join(row["keywords"])
            writer.writerow(row)


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    summary: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (clean_text(record.get("venue")), int(record.get("year") or 0))
        row = summary.setdefault(
            key,
            {
                "venue": key[0],
                "year": key[1],
                "count": 0,
                "abstracts": 0,
                "doi": 0,
                "venue_type": clean_text(record.get("venue_type")),
            },
        )
        row["count"] += 1
        row["abstracts"] += 1 if clean_text(record.get("abstract")) else 0
        row["doi"] += 1 if clean_text(record.get("doi")) else 0
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["venue", "year", "venue_type", "count", "abstracts", "abstract_rate", "doi"],
        )
        writer.writeheader()
        for row in sorted(summary.values(), key=lambda item: (item["venue"], item["year"])):
            row = row.copy()
            row["abstract_rate"] = round(row["abstracts"] / row["count"], 4) if row["count"] else 0
            writer.writerow(row)


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        venue = clean_text(record.get("venue"))
        year = int(record.get("year") or 0)
        doi = clean_doi(record.get("doi"))
        identity = doi or clean_text(record.get("openreview_url")) or normalize_title(record.get("title"))
        key = (venue, year, identity)
        if not identity or key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def external_cache_path(cache_dir: Path, label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", label)
    return cache_dir / f"{safe}.jsonl"


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, word_positions in index.items():
        for position in word_positions:
            positions.append((int(position), word))
    return " ".join(word for _, word in sorted(positions))


def openalex_work_by_doi(record: dict[str, Any]) -> dict[str, Any]:
    doi = clean_doi(record.get("doi"))
    if not doi:
        return {}
    url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
    response = requests.get(url, headers=OPENALEX_HEADERS, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return openalex_updates_from_work(response.json(), match_prefix="openalex")


def openalex_updates_from_work(work: dict[str, Any], *, match_prefix: str = "openalex") -> dict[str, Any]:
    abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    updates: dict[str, Any] = {
        "openalex_id": clean_text(work.get("id")),
    }
    doi = clean_doi(work.get("doi"))
    if doi:
        updates["doi"] = doi
    if abstract:
        updates["abstract"] = abstract
        updates["abstract_source"] = f"{match_prefix}:abstract_inverted_index"
        updates["abstract_enriched_at"] = now_iso()
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    landing_page_url = clean_text(primary.get("landing_page_url"))
    pdf_url = clean_text(primary.get("pdf_url"))
    if landing_page_url:
        updates["html_url"] = landing_page_url
    if pdf_url:
        updates["pdf_url"] = pdf_url
    if source:
        updates["source_display_name"] = clean_text(source.get("display_name"))
    open_access = work.get("open_access") or {}
    oa_url = clean_text(open_access.get("oa_url"))
    if oa_url:
        updates["open_access_url"] = oa_url
        if oa_url.lower().split("?", 1)[0].endswith(".pdf"):
            updates["pdf_url"] = oa_url
    return updates


def enrich_doi_abstracts(records: list[dict[str, Any]], *, max_workers: int, desc: str) -> int:
    target_indices = [
        index
        for index, record in enumerate(records)
        if clean_doi(record.get("doi")) and not clean_text(record.get("abstract"))
    ]
    if not target_indices:
        return 0
    filled = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(openalex_work_by_doi, records[index]): index for index in target_indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            index = futures[future]
            before = bool(clean_text(records[index].get("abstract")))
            try:
                updates = future.result()
            except Exception:
                continue
            for key, value in updates.items():
                if value and (key != "html_url" or not records[index].get("html_url")):
                    records[index][key] = value
            if not before and clean_text(records[index].get("abstract")):
                filled += 1
    return filled


def crawl_cvf(job: OfficialJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    conf = job.args["conf"]
    candidate_urls = [f"https://openaccess.thecvf.com/{conf}{job.year}?day=all"]
    if conf == "CVPR" and job.year == 2020:
        candidate_urls = [f"https://openaccess.thecvf.com/CVPR2020?day={day}" for day in ["2020-06-16", "2020-06-17", "2020-06-18"]]

    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    expanded_urls: list[str] = []
    for source_url in candidate_urls:
        soup = soup_from_url(source_url, timeout=90)
        if not soup.select("dt.ptitle"):
            base_url = f"https://openaccess.thecvf.com/{conf}{job.year}"
            base_soup = soup_from_url(base_url, timeout=90)
            for link in base_soup.select(f'a[href*="{conf}{job.year}.py?day="]'):
                expanded_urls.append(urljoin(base_url, link["href"]))
            if expanded_urls:
                continue
        for dt in soup.select("dt.ptitle"):
            title_link = dt.select_one("a[href]")
            if not title_link:
                continue
            html_url = urljoin(source_url, title_link["href"])
            if html_url in seen_urls:
                continue
            seen_urls.add(html_url)
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

    for source_url in list(dict.fromkeys(expanded_urls)):
        soup = soup_from_url(source_url, timeout=90)
        for dt in soup.select("dt.ptitle"):
            title_link = dt.select_one("a[href]")
            if not title_link:
                continue
            html_url = urljoin(source_url, title_link["href"])
            if html_url in seen_urls:
                continue
            seen_urls.add(html_url)
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

    if fetch_detail_abstracts:
        def fetcher(record: dict[str, Any]) -> dict[str, str]:
            page = soup_from_url(record["html_url"], timeout=30)
            abstract = page.select_one("#abstract")
            return {"abstract": abstract.get_text(" ", strip=True) if abstract else ""}

        records = enrich_detail_pages(records, fetcher, max_workers=max_workers, desc=f"{job.venue} {job.year} abstracts")
    return records


def crawl_ecva(job: OfficialJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    source_url = "https://www.ecva.net/papers.php"
    soup = soup_from_url(source_url, timeout=90)
    marker = f"eccv_{job.year}/papers_ECCV/html"
    records: list[dict[str, Any]] = []
    for dt in soup.select("dt.ptitle"):
        title_link = dt.select_one("a[href]")
        if not title_link or marker not in title_link["href"]:
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
    if fetch_detail_abstracts:
        def fetcher(record: dict[str, Any]) -> dict[str, str]:
            page = soup_from_url(record["html_url"], timeout=30)
            abstract = page.select_one("#abstract")
            return {"abstract": abstract.get_text(" ", strip=True) if abstract else ""}

        records = enrich_detail_pages(records, fetcher, max_workers=max_workers, desc=f"{job.venue} {job.year} abstracts")
    return records


def crawl_neurips(job: OfficialJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    candidate_urls = [
        f"https://papers.nips.cc/paper_files/paper/{job.year}",
        f"https://papers.nips.cc/paper/{job.year}",
    ]
    source_url = ""
    soup = None
    for url in candidate_urls:
        try:
            candidate = soup_from_url(url, timeout=90)
            if candidate.select("a[href*='Abstract'][href$='.html']") or candidate.select("a[href*='abstract']"):
                source_url = url
                soup = candidate
                break
        except Exception:
            continue
    if soup is None:
        raise RuntimeError(f"No NeurIPS listing found for {job.year}")

    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for link in soup.select("a[href*='Abstract'][href$='.html'], a[href*='abstract'][href$='.html']"):
        html_url = urljoin(source_url, link["href"])
        if html_url in seen_urls:
            continue
        seen_urls.add(html_url)
        item = link.find_parent("li")
        authors: list[str] = []
        track = ""
        if item:
            author_el = item.select_one(".paper-authors")
            authors = split_authors(author_el.get_text(" ", strip=True) if author_el else "")
            track_el = item.select_one(".paper-track-badge")
            track = track_el.get_text(" ", strip=True) if track_el else clean_text(item.get("data-track"))
        records.append(
            make_record(
                domain=job.domain,
                venue=job.venue,
                year=job.year,
                source=job.source,
                status="accepted",
                title=link.get_text(" ", strip=True),
                authors=authors,
                track=track,
                html_url=html_url,
                pdf_url=html_url.replace("-Abstract-", "-Paper-").replace(".html", ".pdf"),
                source_url=source_url,
            )
        )
    if fetch_detail_abstracts:
        def fetcher(record: dict[str, Any]) -> dict[str, str]:
            page = soup_from_url(record["html_url"], timeout=30)
            abstract_el = page.select_one(".paper-abstract")
            abstract = abstract_el.get_text(" ", strip=True) if abstract_el else ""
            if not abstract:
                heading = page.find(["h2", "h3"], string=re.compile(r"^\s*Abstract\s*$", re.I))
                if heading:
                    paragraph = heading.find_next("p")
                    abstract = paragraph.get_text(" ", strip=True) if paragraph else ""
            return {"abstract": abstract}

        records = enrich_detail_pages(records, fetcher, max_workers=max_workers, desc=f"{job.venue} {job.year} abstracts")
    return records


def crawl_pmlr(job: OfficialJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    volume = job.args["volume"]
    source_url = f"https://proceedings.mlr.press/{volume}/"
    soup = soup_from_url(source_url, timeout=90)
    records: list[dict[str, Any]] = []
    for paper in soup.select(".paper"):
        title_el = paper.select_one(".title")
        author_el = paper.select_one(".authors")
        abs_link = paper.select_one("a[href*='proceedings.mlr.press']")
        pdf_link = paper.find("a", string=re.compile("Download PDF", re.I))
        openreview_link = paper.find("a", string=re.compile("OpenReview", re.I))
        if not title_el or not abs_link:
            continue
        records.append(
            make_record(
                domain=job.domain,
                venue=job.venue,
                year=job.year,
                source=job.source,
                status="accepted",
                title=title_el.get_text(" ", strip=True),
                authors=split_authors(author_el.get_text(" ", strip=True) if author_el else ""),
                html_url=urljoin(source_url, abs_link["href"]),
                pdf_url=urljoin(source_url, pdf_link["href"]) if pdf_link else "",
                openreview_url=openreview_link["href"] if openreview_link else "",
                source_url=source_url,
            )
        )
    if fetch_detail_abstracts:
        def fetcher(record: dict[str, Any]) -> dict[str, str]:
            page = soup_from_url(record["html_url"], timeout=30)
            abstract = page.select_one("#abstract") or page.select_one(".abstract")
            return {"abstract": abstract.get_text(" ", strip=True) if abstract else ""}

        records = enrich_detail_pages(records, fetcher, max_workers=max_workers, desc=f"{job.venue} {job.year} abstracts")
    return records


def crawl_acl(job: OfficialJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    del fetch_detail_abstracts, max_workers
    event = job.args["event"]
    source_url = f"https://aclanthology.org/events/{event}/"
    soup = soup_from_url(source_url, timeout=120)
    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    excluded_tracks = ("findings", "industry", "demo", "system demonstration", "student research", "tutorial")
    for block in soup.select("div.d-sm-flex.align-items-stretch.mb-3"):
        title_link = block.select_one("strong a[href]")
        if not title_link:
            continue
        href = title_link["href"]
        if not re.fullmatch(rf"/{job.year}\.[A-Za-z0-9_.-]+\.\d+/", href):
            continue
        if href.endswith(".0/"):
            continue
        html_url = urljoin(source_url, href)
        if html_url in seen_urls:
            continue
        volume_section = block.find_parent("div")
        track = ""
        if volume_section:
            heading = volume_section.find_previous("h4")
            track = heading.get_text(" ", strip=True) if heading else ""
        if any(token in track.lower() for token in excluded_tracks):
            continue
        seen_urls.add(html_url)
        pdf_link = block.select_one("a[aria-label='Open PDF'][href]")
        author_links = [a.get_text(" ", strip=True) for a in block.select("span.d-block > a[href*='/people/']")]
        abstract = ""
        abstract_block = block.find_next_sibling("div", class_=lambda cls: cls and "abstract-collapse" in cls)
        if abstract_block:
            abstract = abstract_block.get_text(" ", strip=True)
        records.append(
            make_record(
                domain=job.domain,
                venue=job.venue,
                year=job.year,
                source=job.source,
                status="accepted",
                title=title_link.get_text(" ", strip=True),
                authors=author_links,
                abstract=abstract,
                track=track,
                html_url=html_url,
                pdf_url=pdf_link["href"] if pdf_link else "",
                source_url=source_url,
            )
        )
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
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value.get("value", default)
    return value


def make_openreview_record(note: dict[str, Any], *, venue: str, year: int, domain: str, status: str) -> dict[str, Any]:
    content = note.get("content", {})
    title = content_value(content, "title")
    authors = content_value(content, "authors", [])
    keywords = content_value(content, "keywords", [])
    if not isinstance(authors, list):
        authors = []
    if not isinstance(keywords, list):
        keywords = []
    forum = note.get("forum") or note.get("id") or ""
    forum_url = f"https://openreview.net/forum?id={forum}" if forum else ""
    pdf_field = clean_text(content_value(content, "pdf"))
    pdf_url = urljoin("https://openreview.net", pdf_field) if pdf_field.startswith("/") else pdf_field
    if not pdf_url and note.get("id"):
        pdf_url = f"https://openreview.net/pdf?id={note.get('id')}"
    record = make_record(
        domain=domain,
        venue=venue,
        year=year,
        source="OpenReview",
        status=status,
        title=title,
        authors=authors,
        abstract=content_value(content, "abstract"),
        keywords=keywords,
        track=content_value(content, "primary_area") or content_value(content, "subject_areas") or content_value(content, "venue"),
        paper_id=note.get("id") or forum,
        html_url=forum_url,
        pdf_url=pdf_url,
        openreview_url=forum_url,
        source_url=f"https://openreview.net/group?id={venue}.cc/{year}/Conference" if venue == "ICLR" else "https://openreview.net/",
    )
    if content.get("venueid"):
        record["venueid"] = content.get("venueid")
    return record


def crawl_iclr(job: OfficialJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    del fetch_detail_abstracts, max_workers
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        data = old_openreview_get(
            {
                "content.venueid": f"ICLR.cc/{job.year}/Conference",
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
            if not str(note.get("invitation", "")).startswith(f"ICLR.cc/{job.year}/Conference"):
                continue
            records.append(make_openreview_record(note, venue="ICLR", year=job.year, domain=job.domain, status=status))
        if len(notes) < 1000:
            break
        offset += 1000
    return records


def crawl_colm(job: OfficialJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    del fetch_detail_abstracts, max_workers
    venueid = job.args["venueid"]
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = f"https://api2.openreview.net/notes?content.venueid={quote(venueid)}&limit=1000&offset={offset}"
        data = http_get(url, timeout=90).json()
        notes = data.get("notes", [])
        if not notes:
            break
        for note in notes:
            records.append(make_openreview_record(note, venue="COLM", year=job.year, domain=job.domain, status="accepted"))
        if len(notes) < 1000:
            break
        offset += 1000
    return records


def crawl_ijcai(job: OfficialJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    source_url = f"https://www.ijcai.org/proceedings/{job.year}/"
    soup = soup_from_url(source_url, timeout=90)
    records: list[dict[str, Any]] = []
    for block in soup.select(".paper_wrapper"):
        title_el = block.select_one(".title")
        detail_link = block.find("a", string=re.compile("Details", re.I))
        pdf_link = block.find("a", string=re.compile("PDF", re.I))
        if not title_el or not detail_link:
            continue
        text = block.get_text(" ", strip=True)
        title = title_el.get_text(" ", strip=True)
        authors_text = text.replace(title, "", 1).split("( PDF", 1)[0]
        records.append(
            make_record(
                domain=job.domain,
                venue=job.venue,
                year=job.year,
                source=job.source,
                status="accepted",
                title=title,
                authors=split_authors(authors_text),
                html_url=urljoin(source_url, detail_link["href"]),
                pdf_url=urljoin(source_url, pdf_link["href"]) if pdf_link else "",
                source_url=source_url,
            )
        )
    if fetch_detail_abstracts:
        def fetcher(record: dict[str, Any]) -> dict[str, str]:
            page = soup_from_url(record["html_url"], timeout=30)
            abstract_el = page.select_one(".proceedings-detail .row .col-md-12")
            doi_meta = page.select_one("meta[name='citation_doi']")
            return {
                "abstract": abstract_el.get_text(" ", strip=True) if abstract_el else "",
                "doi": doi_meta["content"] if doi_meta and doi_meta.get("content") else "",
            }

        records = enrich_detail_pages(records, fetcher, max_workers=max_workers, desc=f"{job.venue} {job.year} abstracts")
    return records


def enrich_detail_pages(records: list[dict[str, Any]], fetcher, *, max_workers: int, desc: str) -> list[dict[str, Any]]:
    target_indices = [index for index, record in enumerate(records) if record.get("html_url") and not clean_text(record.get("abstract"))]
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


OFFICIAL_CRAWLERS = {
    "cvf": crawl_cvf,
    "ecva": crawl_ecva,
    "neurips": crawl_neurips,
    "pmlr": crawl_pmlr,
    "acl": crawl_acl,
    "iclr": crawl_iclr,
    "colm": crawl_colm,
    "ijcai": crawl_ijcai,
}


GENERIC_EXCLUDED_SECTIONS = (
    "front matter",
    "preface",
    "committee",
    "organization",
    "organizing committee",
    "program committee",
    "reviewers",
    "keynote",
    "invited",
    "tutorial",
    "tutorials",
    "workshop",
    "workshops",
    "doctoral",
    "demo",
    "demonstration",
    "panel",
    "competition",
    "challenge",
    "grand challenge",
    "companion",
    "student",
    "abstract only",
)


def should_include_dblp_entry(venue: str, section: str, title: str) -> bool:
    section_lower = clean_text(section).lower()
    title_lower = clean_text(title).lower()
    if any(token in section_lower for token in GENERIC_EXCLUDED_SECTIONS):
        return False
    if any(token in title_lower for token in ("front matter", "preface", "committee")):
        return False
    if venue == "KDD":
        return any(token in section_lower for token in ("research", "applied data", "datasets", "benchmark")) or not section_lower
    if venue == "ACMMM":
        excluded = ("brave new idea", "industrial", "video program", "doctoral", "open source", "grand challenge")
        return not any(token in section_lower for token in excluded)
    if venue == "CHI":
        return not any(token in section_lower for token in ("case studies", "courses", "alt.chi", "late-breaking", "extended abstracts"))
    return True


def dblp_candidate_urls(job: DblpJob) -> list[str]:
    urls: list[str] = []
    for code in job.codes:
        for prefix in job.prefixes:
            urls.append(f"https://dblp.org/db/conf/{code}/{prefix}{job.year}.html")
            urls.append(f"https://dblp.org/db/conf/{code}/{prefix}{job.year}-1.html")
    return list(dict.fromkeys(urls))


def crawl_dblp(job: DblpJob, *, enrich_openalex: bool, max_workers: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_url = ""
    for candidate_url in dblp_candidate_urls(job):
        try:
            soup = soup_from_url(candidate_url, timeout=90)
        except Exception:
            continue
        entries = soup.select("li.entry.inproceedings, li.entry.article")
        if not entries:
            continue
        source_url = candidate_url
        for entry in entries:
            cite = entry.select_one("cite")
            title_el = entry.select_one("cite .title")
            if not cite or not title_el:
                continue
            title = clean_text(title_el.get_text(" ", strip=True)).rstrip(".")
            section_el = entry.find_previous(["h2", "h3"])
            section = clean_text(section_el.get_text(" ", strip=True) if section_el else "")
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
                    source="DBLP + OpenAlex",
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
        if records:
            break
    records = dedupe(records)
    if enrich_openalex:
        filled = enrich_doi_abstracts(records, max_workers=max_workers, desc=f"{job.venue} {job.year} OpenAlex DOI")
        if filled:
            print(f"{job.venue} {job.year}: OpenAlex DOI filled {filled:,} abstracts")
    return records


def openalex_journal_query(source_filter: str, year: int, cursor: str = "*") -> str:
    filters = ",".join([source_filter, f"publication_year:{year}"])
    select = ",".join(
        [
            "id",
            "doi",
            "title",
            "publication_year",
            "publication_date",
            "authorships",
            "abstract_inverted_index",
            "open_access",
            "primary_location",
            "type",
        ]
    )
    return f"https://api.openalex.org/works?filter={quote(filters, safe=':,')}&per-page=200&cursor={quote(cursor)}&select={select}"


def crawl_journal(job: JournalJob) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_filters = []
    if job.openalex_source_id:
        source_filters.append(f"primary_location.source.id:{job.openalex_source_id}")
    source_filters.extend(f"primary_location.source.issn:{issn}" for issn in job.issns)
    for source_filter in source_filters:
        cursor = "*"
        while cursor:
            url = openalex_journal_query(source_filter, job.year, cursor=cursor)
            response = requests.get(url, headers=OPENALEX_HEADERS, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            for work in data.get("results", []):
                title = clean_text(work.get("title"))
                if not title:
                    continue
                authors = []
                for authorship in work.get("authorships", []) or []:
                    author = authorship.get("author") or {}
                    name = clean_text(author.get("display_name"))
                    if name:
                        authors.append(name)
                updates = openalex_updates_from_work(work, match_prefix="OpenAlex")
                records.append(
                    make_record(
                        domain=job.domain,
                        venue=job.venue,
                        year=int(work.get("publication_year") or job.year),
                        source="OpenAlex",
                        status="published",
                        title=title,
                        authors=authors,
                        abstract=updates.get("abstract", ""),
                        doi=updates.get("doi", ""),
                        html_url=updates.get("html_url", ""),
                        pdf_url=updates.get("pdf_url", ""),
                        source_url=url,
                        venue_type="journal",
                        extra={
                            "journal": job.full_name,
                            "issn": ";".join(job.issns),
                            "openalex_source_filter": source_filter,
                            "openalex_id": updates.get("openalex_id", ""),
                            "abstract_source": updates.get("abstract_source", ""),
                            "publication_date": clean_text(work.get("publication_date")),
                            "open_access_url": updates.get("open_access_url", ""),
                        },
                    )
                )
            next_cursor = (data.get("meta") or {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor or not data.get("results"):
                break
            cursor = next_cursor
            time.sleep(0.15)
    return dedupe(records)


def icml_volumes() -> dict[int, str]:
    return {
        2010: "v9",
        2011: "v15",
        2012: "v22",
        2013: "v28",
        2014: "v32",
        2015: "v37",
        2016: "v48",
        2017: "v70",
        2018: "v80",
        2019: "v97",
    }


def build_official_jobs(start_year: int, end_year: int) -> list[OfficialJob]:
    jobs: list[OfficialJob] = []
    jobs.extend(
        OfficialJob("CVPR", year, "cv", "cvf", "CVF Open Access", {"conf": "CVPR"})
        for year in range(max(start_year, 2013), min(end_year, 2019) + 1)
    )
    jobs.extend(
        OfficialJob("ICCV", year, "cv", "cvf", "CVF Open Access", {"conf": "ICCV"})
        for year in [2013, 2015, 2017, 2019]
        if start_year <= year <= end_year
    )
    jobs.extend(
        OfficialJob("ECCV", year, "cv", "ecva", "ECVA", {})
        for year in [2018]
        if start_year <= year <= end_year
    )
    jobs.extend(
        OfficialJob("NeurIPS", year, "ml", "neurips", "NeurIPS Proceedings", {})
        for year in range(max(start_year, 1987), min(end_year, 2019) + 1)
    )
    jobs.extend(
        OfficialJob("ICLR", year, "ml", "iclr", "OpenReview", {})
        for year in range(max(start_year, 2013), min(end_year, 2019) + 1)
    )
    volumes = icml_volumes()
    jobs.extend(
        OfficialJob("ICML", year, "ml", "pmlr", "PMLR", {"volume": volumes[year]})
        for year in sorted(volumes)
        if start_year <= year <= end_year
    )
    for venue, event_prefix in [("ACL", "acl"), ("EMNLP", "emnlp"), ("NAACL", "naacl")]:
        first_year = {"ACL": 1979, "EMNLP": 1996, "NAACL": 2000}[venue]
        for year in range(max(start_year, first_year), min(end_year, 2019) + 1):
            jobs.append(OfficialJob(venue, year, "nlp", "acl", "ACL Anthology", {"event": f"{event_prefix}-{year}"}))
    jobs.extend(
        OfficialJob("IJCAI", year, "general_ai", "ijcai", "IJCAI Proceedings", {})
        for year in range(max(start_year, 2017), min(end_year, 2019) + 1)
    )
    jobs.extend(
        OfficialJob("COLM", year, "language_modeling", "colm", "OpenReview", {"venueid": f"colmweb.org/COLM/{year}/Conference"})
        for year in [2024, 2025]
        if start_year <= year <= end_year
    )
    return jobs


def build_dblp_jobs(start_year: int, end_year: int) -> list[DblpJob]:
    specs = [
        ("AAAI", "general_ai", ("aaai",), ("aaai",), 1980, 2019),
        ("IJCAI", "general_ai", ("ijcai",), ("ijcai",), 1969, 2016),
        ("ACMMM", "multimedia", ("mm",), ("mm",), 1993, 2019),
        ("KDD", "data_mining", ("kdd",), ("kdd",), 1995, 2019),
        ("SIGIR", "information_retrieval", ("sigir",), ("sigir",), 1978, 2019),
        ("WWW", "web", ("www",), ("www",), 1994, 2019),
        ("CHI", "hci", ("chi",), ("chi",), 1982, 2025),
        ("ECCV", "cv", ("eccv",), ("eccv",), 1990, 2016),
        ("ICML", "ml", ("icml",), ("icml",), 1988, 2009),
        ("SIGGRAPH", "graphics", ("siggraph",), ("siggraph",), 1974, 2025),
        ("SIGGRAPH-Asia", "graphics", ("siggrapha", "sa"), ("siggrapha", "sa"), 2008, 2025),
        ("ICRA", "robotics", ("icra",), ("icra",), 1984, 2025),
        ("IROS", "robotics", ("iros",), ("iros",), 1988, 2025),
        ("RSS", "robotics", ("rss",), ("rss",), 2005, 2025),
        ("MICCAI", "medical_ai", ("miccai",), ("miccai",), 1998, 2025),
        ("SIGMOD", "database", ("sigmod",), ("sigmod",), 1975, 2025),
        ("VLDB", "database", ("vldb",), ("vldb",), 1975, 2025),
        ("ICDE", "database", ("icde",), ("icde",), 1984, 2025),
    ]
    jobs: list[DblpJob] = []
    for venue, domain, codes, prefixes, first_year, last_year in specs:
        for year in range(max(start_year, first_year), min(end_year, last_year) + 1):
            jobs.append(DblpJob(venue=venue, year=year, domain=domain, codes=codes, prefixes=prefixes))
    return jobs


def build_journal_jobs(start_year: int, end_year: int) -> list[JournalJob]:
    specs = [
        ("TPAMI", "cv", ("0162-8828", "2160-9292"), "IEEE Transactions on Pattern Analysis and Machine Intelligence", "S199944782"),
        ("IJCV", "cv", ("0920-5691", "1573-1405"), "International Journal of Computer Vision", "S25538012"),
        ("TIP", "cv", ("1057-7149", "1941-0042"), "IEEE Transactions on Image Processing", "S4210173141"),
        ("TMM", "multimedia", ("1520-9210", "1941-0077"), "IEEE Transactions on Multimedia", "S137030581"),
        ("PR", "cv", ("0031-3203",), "Pattern Recognition", "S414566"),
        ("TKDE", "database", ("1041-4347", "1558-2191"), "IEEE Transactions on Knowledge and Data Engineering", "S30698027"),
        ("AIJ", "general_ai", ("0004-3702",), "Artificial Intelligence", "S196139623"),
        ("TNNLS", "ml", ("2162-237X", "2162-2388"), "IEEE Transactions on Neural Networks and Learning Systems", "S4210175523"),
        ("JMLR", "ml", ("1532-4435",), "Journal of Machine Learning Research", "S118988714"),
    ]
    jobs: list[JournalJob] = []
    for venue, domain, issns, full_name, source_id in specs:
        for year in range(max(start_year, 2000), end_year + 1):
            jobs.append(
                JournalJob(
                    venue=venue,
                    year=year,
                    domain=domain,
                    issns=issns,
                    full_name=full_name,
                    openalex_source_id=source_id,
                )
            )
    return jobs


def should_run_venue(venue: str, selected: set[str] | None) -> bool:
    return not selected or venue.upper() in selected


def load_or_crawl_official(
    job: OfficialJob,
    *,
    cache_dir: Path,
    refresh: bool,
    fetch_detail_abstracts: bool,
    max_workers: int,
) -> list[dict[str, Any]]:
    label = f"official_{job.venue}_{job.year}"
    path = external_cache_path(cache_dir, label)
    if path.exists() and not refresh:
        records = read_jsonl(path)
        print(f"[CACHE] {label}: {len(records):,}")
        return records
    started = time.time()
    crawler = OFFICIAL_CRAWLERS[job.crawler]
    records = crawler(job, fetch_detail_abstracts=fetch_detail_abstracts, max_workers=max_workers)
    records = dedupe(records)
    write_jsonl(path, records)
    print(f"[OK] {label}: {len(records):,} records in {time.time() - started:.1f}s")
    return records


def load_or_crawl_dblp(
    job: DblpJob,
    *,
    cache_dir: Path,
    refresh: bool,
    enrich_openalex: bool,
    max_workers: int,
) -> list[dict[str, Any]]:
    label = f"dblp_{job.venue}_{job.year}"
    path = external_cache_path(cache_dir, label)
    if path.exists() and not refresh:
        records = read_jsonl(path)
        print(f"[CACHE] {label}: {len(records):,}")
        return records
    started = time.time()
    records = crawl_dblp(job, enrich_openalex=enrich_openalex, max_workers=max_workers)
    write_jsonl(path, records)
    print(f"[OK] {label}: {len(records):,} records in {time.time() - started:.1f}s")
    return records


def load_or_crawl_journal(job: JournalJob, *, cache_dir: Path, refresh: bool) -> list[dict[str, Any]]:
    label = f"journal_{job.venue}_{job.year}"
    path = external_cache_path(cache_dir, label)
    if path.exists() and not refresh:
        records = read_jsonl(path)
        print(f"[CACHE] {label}: {len(records):,}")
        return records
    started = time.time()
    records = crawl_journal(job)
    write_jsonl(path, records)
    print(f"[OK] {label}: {len(records):,} records in {time.time() - started:.1f}s")
    return records


def merge_with_base(base_path: Path, added_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = read_jsonl(base_path)
    records.extend(added_records)
    return dedupe(records)


def write_missing_abstracts(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = ["venue", "year", "venue_type", "title", "doi", "html_url", "source_url"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if clean_text(record.get("abstract")):
                continue
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "metadata_expansion")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "metadata_expansion" / "cache")
    parser.add_argument("--base-jsonl", type=Path, default=SIBLING_MAIN_BASE)
    parser.add_argument("--run-name", default="expanded_paper_metadata_1969_2026")
    parser.add_argument("--start-year", type=int, default=1969)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--venues", nargs="*", help="Optional venue filter.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-dblp", action="store_true")
    parser.add_argument("--skip-journals", action="store_true")
    parser.add_argument("--no-detail-abstracts", action="store_true")
    parser.add_argument("--no-openalex-doi", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {venue.upper() for venue in args.venues} if args.venues else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    all_added: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if not args.skip_official:
        for job in build_official_jobs(args.start_year, args.end_year):
            if not should_run_venue(job.venue, selected):
                continue
            try:
                all_added.extend(
                    load_or_crawl_official(
                        job,
                        cache_dir=args.cache_dir,
                        refresh=args.refresh,
                        fetch_detail_abstracts=not args.no_detail_abstracts,
                        max_workers=args.max_workers,
                    )
                )
            except Exception as exc:
                failures.append({"kind": "official", "venue": job.venue, "year": job.year, "error": str(exc)})
                print(f"[ERROR] official {job.venue} {job.year}: {exc}", file=sys.stderr)

    if not args.skip_dblp:
        for job in build_dblp_jobs(args.start_year, args.end_year):
            if not should_run_venue(job.venue, selected):
                continue
            try:
                all_added.extend(
                    load_or_crawl_dblp(
                        job,
                        cache_dir=args.cache_dir,
                        refresh=args.refresh,
                        enrich_openalex=not args.no_openalex_doi,
                        max_workers=args.max_workers,
                    )
                )
            except Exception as exc:
                failures.append({"kind": "dblp", "venue": job.venue, "year": job.year, "error": str(exc)})
                print(f"[ERROR] dblp {job.venue} {job.year}: {exc}", file=sys.stderr)

    if not args.skip_journals:
        for job in build_journal_jobs(args.start_year, args.end_year):
            if not should_run_venue(job.venue, selected):
                continue
            try:
                all_added.extend(load_or_crawl_journal(job, cache_dir=args.cache_dir, refresh=args.refresh))
            except Exception as exc:
                failures.append({"kind": "journal", "venue": job.venue, "year": job.year, "error": str(exc)})
                print(f"[ERROR] journal {job.venue} {job.year}: {exc}", file=sys.stderr)

    added_records = dedupe(all_added)
    added_jsonl = args.output_dir / f"{args.run_name}_added.jsonl"
    added_csv = args.output_dir / f"{args.run_name}_added.csv"
    added_summary = args.output_dir / f"{args.run_name}_added_summary.csv"
    missing_csv = args.output_dir / f"{args.run_name}_missing_abstracts.csv"
    write_jsonl(added_jsonl, added_records)
    write_csv(added_csv, added_records)
    write_summary(added_summary, added_records)
    write_missing_abstracts(missing_csv, added_records)

    merged_jsonl = ""
    merged_csv = ""
    merged_summary = ""
    merged_records: list[dict[str, Any]] = []
    if args.base_jsonl.exists():
        merged_records = merge_with_base(args.base_jsonl, added_records)
        merged_jsonl_path = args.output_dir / f"{args.run_name}_merged.jsonl"
        merged_csv_path = args.output_dir / f"{args.run_name}_merged.csv"
        merged_summary_path = args.output_dir / f"{args.run_name}_merged_summary.csv"
        write_jsonl(merged_jsonl_path, merged_records)
        write_csv(merged_csv_path, merged_records)
        write_summary(merged_summary_path, merged_records)
        merged_jsonl = str(merged_jsonl_path)
        merged_csv = str(merged_csv_path)
        merged_summary = str(merged_summary_path)

    report = {
        "added_records": len(added_records),
        "added_abstracts": sum(1 for record in added_records if clean_text(record.get("abstract"))),
        "added_missing_abstracts": sum(1 for record in added_records if not clean_text(record.get("abstract"))),
        "merged_records": len(merged_records) if merged_records else 0,
        "base_jsonl": str(args.base_jsonl),
        "added_jsonl": str(added_jsonl),
        "added_csv": str(added_csv),
        "added_summary": str(added_summary),
        "missing_abstracts_csv": str(missing_csv),
        "merged_jsonl": merged_jsonl,
        "merged_csv": merged_csv,
        "merged_summary": merged_summary,
        "failures": failures,
        "updated_at": now_iso(),
    }
    report_path = args.output_dir / f"{args.run_name}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
