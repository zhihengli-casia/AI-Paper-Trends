"""Crawl recent public conference paper metadata from official sources.

The default plan covers the last three public editions for:

- CV: CVPR, ICCV, ECCV
- NLP: ACL, EMNLP, NAACL
- ML: ICLR, NeurIPS, ICML
- General AI: AAAI, IJCAI

By default this performs a list-level crawl. It captures title, authors, paper
page and PDF links, plus abstracts when they are available directly from the
listing/API pages. Use ``--fetch-detail-abstracts`` to visit individual paper
pages for sources that require a per-paper request.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin

import openreview.api
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


USER_AGENT = "AI-Paper-Trends/0.1 (+public metadata research)"
REQUEST_HEADERS = {"User-Agent": USER_AGENT}
DEFAULT_TIMEOUT = 90


@dataclass(frozen=True)
class CrawlJob:
    domain: str
    venue: str
    year: int
    source: str
    crawler: str
    args: dict[str, Any]


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def split_authors(value: str | None) -> list[str]:
    value = clean_text(value)
    if not value:
        return []
    if " | " in value:
        return [clean_text(x) for x in value.split("|") if clean_text(x)]
    return [clean_text(x) for x in re.split(r"\s*,\s*", value) if clean_text(x)]


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT, retries: int = 3) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - defensive network retry
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def soup_from_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> BeautifulSoup:
    return BeautifulSoup(http_get(url, timeout=timeout).text, "lxml")


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
) -> dict[str, Any]:
    return {
        "paper_id": paper_id or html_url or pdf_url or f"{venue}-{year}-{clean_text(title)}",
        "domain": domain,
        "venue": venue,
        "year": year,
        "source": source,
        "status": status,
        "track": clean_text(track),
        "title": clean_text(title),
        "authors": [clean_text(a) for a in (authors or []) if clean_text(a)],
        "abstract": clean_text(abstract),
        "keywords": [clean_text(k) for k in (keywords or []) if clean_text(k)],
        "html_url": html_url,
        "pdf_url": pdf_url,
        "openreview_url": openreview_url,
        "doi": doi,
        "source_url": source_url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def enrich_abstracts(
    records: list[dict[str, Any]],
    fetcher: Callable[[dict[str, Any]], dict[str, str]],
    *,
    enabled: bool,
    max_workers: int,
    desc: str,
) -> list[dict[str, Any]]:
    if not enabled or not records:
        return records

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(fetcher, record): index for index, record in enumerate(records)}
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=desc):
            index = future_to_index[future]
            try:
                updates = future.result()
            except Exception:
                continue
            for key, value in updates.items():
                if value:
                    records[index][key] = value
    return records


def crawl_cvf(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    conf = job.args["conf"]
    source_url = f"https://openaccess.thecvf.com/{conf}{job.year}?day=all"
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

    def fetcher(record: dict[str, Any]) -> dict[str, str]:
        page = soup_from_url(record["html_url"], timeout=30)
        abstract = page.select_one("#abstract")
        return {"abstract": abstract.get_text(" ", strip=True) if abstract else ""}

    return enrich_abstracts(
        records,
        fetcher,
        enabled=fetch_detail_abstracts,
        max_workers=max_workers,
        desc=f"{job.venue} {job.year} abstracts",
    )


def crawl_ecva(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    source_url = "https://www.ecva.net/papers.php"
    soup = soup_from_url(source_url)
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

    def fetcher(record: dict[str, Any]) -> dict[str, str]:
        page = soup_from_url(record["html_url"], timeout=30)
        abstract = page.select_one("#abstract")
        return {"abstract": abstract.get_text(" ", strip=True) if abstract else ""}

    return enrich_abstracts(
        records,
        fetcher,
        enabled=fetch_detail_abstracts,
        max_workers=max_workers,
        desc=f"{job.venue} {job.year} abstracts",
    )


def crawl_neurips(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    source_url = f"https://papers.nips.cc/paper_files/paper/{job.year}"
    soup = soup_from_url(source_url)
    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for link in soup.select("a[href*='Abstract'][href$='.html']"):
        html_url = urljoin(source_url, link["href"])
        if html_url in seen_urls:
            continue
        seen_urls.add(html_url)
        item = link.find_parent("li")
        authors = []
        track = ""
        if item:
            author_el = item.select_one(".paper-authors")
            authors = split_authors(author_el.get_text(" ", strip=True) if author_el else "")
            track_el = item.select_one(".paper-track-badge")
            track = track_el.get_text(" ", strip=True) if track_el else item.get("data-track", "")
        pdf_url = html_url.replace("-Abstract-", "-Paper-").replace(".html", ".pdf")
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
                pdf_url=pdf_url,
                source_url=source_url,
            )
        )

    def fetcher(record: dict[str, Any]) -> dict[str, str]:
        page = soup_from_url(record["html_url"], timeout=30)
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
                next_p = heading.find_next("p")
                abstract = next_p.get_text(" ", strip=True) if next_p else ""
        return {"abstract": abstract}

    return enrich_abstracts(
        records,
        fetcher,
        enabled=fetch_detail_abstracts,
        max_workers=max_workers,
        desc=f"{job.venue} {job.year} abstracts",
    )


def crawl_pmlr(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    volume = job.args["volume"]
    source_url = f"https://proceedings.mlr.press/{volume}/"
    soup = soup_from_url(source_url)
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

    def fetcher(record: dict[str, Any]) -> dict[str, str]:
        page = soup_from_url(record["html_url"], timeout=30)
        abstract = page.select_one("#abstract") or page.select_one(".abstract")
        return {"abstract": abstract.get_text(" ", strip=True) if abstract else ""}

    return enrich_abstracts(
        records,
        fetcher,
        enabled=fetch_detail_abstracts,
        max_workers=max_workers,
        desc=f"{job.venue} {job.year} abstracts",
    )


def crawl_acl(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    del fetch_detail_abstracts, max_workers
    event = job.args["event"]
    source_url = f"https://aclanthology.org/events/{event}/"
    soup = soup_from_url(source_url, timeout=120)
    records: list[dict[str, Any]] = []
    year_pattern = str(job.year)
    seen_urls: set[str] = set()

    for block in soup.select("div.d-sm-flex.align-items-stretch.mb-3"):
        title_link = block.select_one("strong a[href]")
        if not title_link:
            continue
        href = title_link["href"]
        if not re.fullmatch(rf"/{year_pattern}\.[A-Za-z0-9_.-]+\.\d+/", href):
            continue
        if href.endswith(".0/"):
            continue
        html_url = urljoin(source_url, href)
        if html_url in seen_urls:
            continue
        seen_urls.add(html_url)
        pdf_link = block.select_one("a[aria-label='Open PDF'][href]")
        author_links = [
            a.get_text(" ", strip=True)
            for a in block.select("span.d-block > a[href*='/people/']")
        ]
        abstract = ""
        next_sibling = block.find_next_sibling("div", class_=lambda cls: cls and "abstract-collapse" in cls)
        if next_sibling:
            abstract = next_sibling.get_text(" ", strip=True)
        volume_section = block.find_parent("div")
        track = ""
        if volume_section:
            heading = volume_section.find_previous("h4")
            track = heading.get_text(" ", strip=True) if heading else ""
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


def crawl_ijcai(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    source_url = f"https://www.ijcai.org/proceedings/{job.year}/"
    soup = soup_from_url(source_url)
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

    def fetcher(record: dict[str, Any]) -> dict[str, str]:
        page = soup_from_url(record["html_url"], timeout=30)
        abstract_el = page.select_one(".proceedings-detail .row .col-md-12")
        doi_meta = page.select_one("meta[name='citation_doi']")
        return {
            "abstract": abstract_el.get_text(" ", strip=True) if abstract_el else "",
            "doi": doi_meta["content"] if doi_meta and doi_meta.get("content") else "",
        }

    return enrich_abstracts(
        records,
        fetcher,
        enabled=fetch_detail_abstracts,
        max_workers=max_workers,
        desc=f"{job.venue} {job.year} abstracts",
    )


def aaai_issue_links(year: int) -> list[str]:
    yy = str(year)[-2:]
    links: set[str] = set()
    for page_num in range(1, 8):
        url = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
        if page_num > 1:
            url = f"{url}/{page_num}"
        soup = soup_from_url(url, timeout=60)
        for link in soup.select("a[href*='/issue/view/']"):
            label = link.get_text(" ", strip=True)
            if f"AAAI-{yy} Technical Tracks" in label:
                links.add(link["href"])
    return sorted(links)


def crawl_aaai(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    issue_urls = aaai_issue_links(job.year)
    records: list[dict[str, Any]] = []

    for issue_url in tqdm(issue_urls, desc=f"{job.venue} {job.year} issues"):
        soup = soup_from_url(issue_url, timeout=60)
        title = soup.select_one("h1")
        track = title.get_text(" ", strip=True) if title else ""
        for article in soup.select(".obj_article_summary"):
            title_link = article.select_one(".title a[href]")
            if not title_link:
                continue
            pdf_link = article.find("a", string=re.compile("PDF", re.I))
            author_text = ""
            authors_el = article.select_one(".authors")
            if authors_el:
                author_text = authors_el.get_text(" ", strip=True)
            records.append(
                make_record(
                    domain=job.domain,
                    venue=job.venue,
                    year=job.year,
                    source=job.source,
                    status="accepted",
                    title=title_link.get_text(" ", strip=True),
                    authors=split_authors(author_text),
                    track=track,
                    html_url=title_link["href"],
                    pdf_url=pdf_link["href"] if pdf_link else "",
                    source_url=issue_url,
                )
            )

    def fetcher(record: dict[str, Any]) -> dict[str, str]:
        page = soup_from_url(record["html_url"], timeout=30)
        abstract_el = page.select_one(".item.abstract") or page.select_one("section.item.abstract")
        return {"abstract": abstract_el.get_text(" ", strip=True).removeprefix("Abstract ") if abstract_el else ""}

    return enrich_abstracts(
        records,
        fetcher,
        enabled=fetch_detail_abstracts,
        max_workers=max_workers,
        desc=f"{job.venue} {job.year} abstracts",
    )


def crawl_iclr(job: CrawlJob, *, fetch_detail_abstracts: bool, max_workers: int) -> list[dict[str, Any]]:
    del fetch_detail_abstracts, max_workers
    conference_id = job.args["conference_id"]
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    invitation = f"{conference_id}/-/Submission"
    records: list[dict[str, Any]] = []

    for note in tqdm(client.get_all_notes(invitation=invitation), desc=f"{job.venue} {job.year} submissions"):
        content = note.content

        def value(name: str, default: Any = "") -> Any:
            field = content.get(name, default)
            if isinstance(field, dict) and "value" in field:
                return field["value"]
            return field

        title = value("title")
        authors = value("authors", [])
        keywords = value("keywords", [])
        pdf_field = value("pdf", "")
        forum_url = f"https://openreview.net/forum?id={note.id}"
        pdf_url = urljoin("https://openreview.net", pdf_field) if isinstance(pdf_field, str) else ""
        venue = value("venue", "")
        venueid = value("venueid", "")
        records.append(
            make_record(
                domain=job.domain,
                venue=job.venue,
                year=job.year,
                source=job.source,
                status=venue or "public_submission",
                title=title,
                authors=authors,
                abstract=value("abstract"),
                keywords=keywords,
                track=value("primary_area", "") or value("subject_areas", ""),
                paper_id=note.id,
                html_url=forum_url,
                pdf_url=pdf_url,
                source_url=f"https://openreview.net/group?id={conference_id}",
                openreview_url=forum_url,
            )
        )
        if venueid:
            records[-1]["venueid"] = venueid
    return records


CRAWLERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "cvf": crawl_cvf,
    "ecva": crawl_ecva,
    "neurips": crawl_neurips,
    "pmlr": crawl_pmlr,
    "acl": crawl_acl,
    "ijcai": crawl_ijcai,
    "aaai": crawl_aaai,
    "iclr": crawl_iclr,
}


DEFAULT_JOBS: list[CrawlJob] = [
    *(CrawlJob("cv", "CVPR", year, "CVF Open Access", "cvf", {"conf": "CVPR"}) for year in [2023, 2024, 2025]),
    *(CrawlJob("cv", "ICCV", year, "CVF Open Access", "cvf", {"conf": "ICCV"}) for year in [2021, 2023, 2025]),
    *(CrawlJob("cv", "ECCV", year, "ECVA", "ecva", {}) for year in [2020, 2022, 2024]),
    *(CrawlJob("ml", "ICLR", year, "OpenReview", "iclr", {"conference_id": f"ICLR.cc/{year}/Conference"}) for year in [2024, 2025, 2026]),
    *(CrawlJob("ml", "NeurIPS", year, "NeurIPS Proceedings", "neurips", {}) for year in [2023, 2024, 2025]),
    CrawlJob("ml", "ICML", 2023, "PMLR", "pmlr", {"volume": "v202"}),
    CrawlJob("ml", "ICML", 2024, "PMLR", "pmlr", {"volume": "v235"}),
    CrawlJob("ml", "ICML", 2025, "PMLR", "pmlr", {"volume": "v267"}),
    *(CrawlJob("nlp", "ACL", year, "ACL Anthology", "acl", {"event": f"acl-{year}"}) for year in [2023, 2024, 2025]),
    *(CrawlJob("nlp", "EMNLP", year, "ACL Anthology", "acl", {"event": f"emnlp-{year}"}) for year in [2023, 2024, 2025]),
    CrawlJob("nlp", "NAACL", 2022, "ACL Anthology", "acl", {"event": "naacl-2022"}),
    CrawlJob("nlp", "NAACL", 2024, "ACL Anthology", "acl", {"event": "naacl-2024"}),
    CrawlJob("nlp", "NAACL", 2025, "ACL Anthology", "acl", {"event": "naacl-2025"}),
    *(CrawlJob("general_ai", "AAAI", year, "AAAI OJS", "aaai", {}) for year in [2024, 2025, 2026]),
    *(CrawlJob("general_ai", "IJCAI", year, "IJCAI Proceedings", "ijcai", {}) for year in [2023, 2024, 2025]),
]


def filter_jobs(jobs: list[CrawlJob], venues: set[str] | None) -> list[CrawlJob]:
    if not venues:
        return jobs
    normalized = {venue.upper() for venue in venues}
    return [job for job in jobs if job.venue.upper() in normalized]


def write_outputs(records: list[dict[str, Any]], output_dir: Path, run_name: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{run_name}.jsonl"
    csv_path = output_dir / f"{run_name}.csv"
    summary_path = output_dir / f"{run_name}_summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if records:
        fieldnames = sorted({key for record in records for key in record.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = record.copy()
                row["authors"] = "; ".join(record.get("authors") or [])
                row["keywords"] = "; ".join(record.get("keywords") or [])
                writer.writerow(row)

    summary: dict[tuple[str, int], int] = {}
    for record in records:
        key = (record["venue"], int(record["year"]))
        summary[key] = summary.get(key, 0) + 1
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["venue", "year", "count"])
        for (venue, year), count in sorted(summary.items()):
            writer.writerow([venue, year, count])

    return {"jsonl": jsonl_path, "csv": csv_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl public recent AI conference paper metadata.")
    parser.add_argument("--output-dir", default="data/recent_conferences", help="Directory for JSONL/CSV outputs.")
    parser.add_argument("--run-name", default="recent_ai_conference_papers", help="Output file stem.")
    parser.add_argument("--venues", nargs="*", help="Optional venue filter, e.g. ICLR CVPR ACL.")
    parser.add_argument("--fetch-detail-abstracts", action="store_true", help="Fetch per-paper pages for missing abstracts.")
    parser.add_argument("--max-workers", type=int, default=12, help="Workers for per-paper detail page requests.")
    args = parser.parse_args()

    jobs = filter_jobs(DEFAULT_JOBS, set(args.venues) if args.venues else None)
    all_records: list[dict[str, Any]] = []

    for job in jobs:
        started = time.time()
        crawler = CRAWLERS[job.crawler]
        try:
            records = crawler(job, fetch_detail_abstracts=args.fetch_detail_abstracts, max_workers=args.max_workers)
        except Exception as exc:
            print(f"[ERROR] {job.venue} {job.year}: {exc}")
            continue
        all_records.extend(records)
        print(f"[OK] {job.venue} {job.year}: {len(records)} records in {time.time() - started:.1f}s")

    paths = write_outputs(all_records, Path(args.output_dir), args.run_name)
    print(f"\nTotal records: {len(all_records)}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
