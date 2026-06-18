import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import openreview.api
import pandas as pd
import requests
from tqdm import tqdm

from src.utils import build_raw_output_path, ensure_directories, extract_content_value

DECISION_PATTERN = re.compile(r"/(Meta_Review|Decision)$")
REVIEW_PATTERN = re.compile(r"/Official_Review$")
OPENREVIEW_API_BASE = "https://api2.openreview.net"
OPENREVIEW_TIMEOUT = 90


def _get_note_field(note: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(note, dict):
        return note.get(field_name, default)
    return getattr(note, field_name, default)


def _get_note_content(note: Any) -> Dict[str, Any]:
    return _get_note_field(note, "content", {}) or {}


def _get_note_invitations(note: Any) -> List[str]:
    invitations = _get_note_field(note, "invitations", []) or []
    return list(invitations)


def _get_note_replies(note: Any) -> Tuple[List[Any], bool]:
    details = _get_note_field(note, "details")
    if isinstance(details, dict) and "replies" in details:
        return details.get("replies") or [], True
    if isinstance(details, dict) and "directReplies" in details:
        return details.get("directReplies") or [], True
    return [], False


def _normalize_list_field(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_rating_value(rating_value: Any) -> int | None:
    if isinstance(rating_value, str):
        match = re.search(r"^\d+", rating_value.strip())
        if match:
            return int(match.group(0))
        return None
    if isinstance(rating_value, (int, float)):
        return int(rating_value)
    return None


def _extract_review_details(related_notes: List[Any]) -> Dict[str, Any]:
    ratings: List[int] = []
    decision = "N/A"

    for note in related_notes:
        invitations = _get_note_invitations(note)
        content = _get_note_content(note)

        if any(DECISION_PATTERN.search(invitation) for invitation in invitations):
            decision_value = extract_content_value(content, "decision")
            if decision_value:
                decision = str(decision_value)

        if any(REVIEW_PATTERN.search(invitation) for invitation in invitations):
            parsed_rating = _parse_rating_value(extract_content_value(content, "rating"))
            if parsed_rating is not None:
                ratings.append(parsed_rating)

    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
    return {
        "decision": decision,
        "review_ratings": ratings,
        "avg_rating": avg_rating,
    }


def _openreview_get(params: Dict[str, Any], retries: int = 5) -> Dict[str, Any]:
    """Fetch OpenReview notes without custom headers; custom User-Agent now gets 403."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                f"{OPENREVIEW_API_BASE}/notes",
                params=params,
                timeout=OPENREVIEW_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"OpenReview raw API request failed: {last_error}")


def _get_all_notes_raw(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    notes: List[Dict[str, Any]] = []
    offset = 0
    limit = 1000
    while True:
        page_params = {**params, "limit": limit, "offset": offset}
        batch = _openreview_get(page_params).get("notes", [])
        if not batch:
            break
        notes.extend(batch)
        if len(batch) < limit:
            break
        offset += len(batch)
    return notes


def _extract_paper_data_from_iterator(
    notes_iterator: Iterator[Any], include_replies: bool = False
) -> Tuple[List[Dict[str, Any]], bool]:
    """Extracts paper fields and optionally review metadata from an OpenReview note iterator."""
    papers_list: List[Dict[str, Any]] = []
    embedded_replies_found = False

    for note in notes_iterator:
        content = _get_note_content(note)
        paper = {
            "id": _get_note_field(note, "id"),
            "title": extract_content_value(content, "title", "N/A"),
            "abstract": extract_content_value(content, "abstract", "N/A"),
            "keywords": _normalize_list_field(extract_content_value(content, "keywords", [])),
            "authors": _normalize_list_field(extract_content_value(content, "authors", [])),
        }

        if include_replies:
            replies, has_embedded_replies = _get_note_replies(note)
            embedded_replies_found = embedded_replies_found or has_embedded_replies
            paper.update(_extract_review_details(replies))

        papers_list.append(paper)

    return papers_list, embedded_replies_found


def get_rich_paper_details(
    client: openreview.api.OpenReviewClient, submissions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fallback path that enriches paper data with review ratings and decisions paper by paper."""
    for paper in tqdm(submissions, desc="Fetching detailed review information"):
        try:
            related_notes = client.get_notes(forum=paper["id"])
        except Exception as exc:
            print(f"Warning: OpenReview client detail fetch failed for {paper['id']}. Falling back to raw API. Error: {exc}")
            try:
                related_notes = _openreview_get({"forum": paper["id"], "limit": 1000}).get("notes", [])
            except Exception as raw_exc:
                print(f"Warning: Could not fetch details for paper {paper['id']}. Error: {raw_exc}")
                paper.update({"decision": "N/A", "review_ratings": [], "avg_rating": None})
                continue

        paper.update(_extract_review_details(related_notes))
    return submissions


def get_all_papers(
    client: openreview.api.OpenReviewClient,
    conference_id: str,
    include_replies: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetches all papers for a conference, trying multiple methods."""
    details = "replies" if include_replies else None
    invitation_id = f"{conference_id}/-/Submission"
    print(f"--> Attempting with Invitation ID: '{invitation_id}'")

    try:
        notes_iterator = client.get_all_notes(invitation=invitation_id, details=details)
        papers, embedded_replies_found = _extract_paper_data_from_iterator(
            tqdm(notes_iterator, desc="Processing papers (Invitation)"),
            include_replies=include_replies,
        )
        if papers:
            return papers, embedded_replies_found
    except Exception as exc:
        print(f"    - Invitation ID method failed: {exc}")

    try:
        print("    - Falling back to raw OpenReview API for Invitation ID")
        params = {"invitation": invitation_id}
        if details:
            params["details"] = details
        raw_notes = _get_all_notes_raw(params)
        papers, embedded_replies_found = _extract_paper_data_from_iterator(
            tqdm(raw_notes, desc="Processing papers (Raw Invitation)"),
            include_replies=include_replies,
        )
        if papers:
            return papers, embedded_replies_found
    except Exception as exc:
        print(f"    - Raw Invitation ID method failed: {exc}")

    print(f"\n--> Falling back to Venue ID: '{conference_id}'")
    try:
        notes_iterator = client.get_all_notes(content={"venueid": conference_id}, details=details)
        papers, embedded_replies_found = _extract_paper_data_from_iterator(
            tqdm(notes_iterator, desc="Processing papers (Venue ID)"),
            include_replies=include_replies,
        )
        if papers:
            return papers, embedded_replies_found
    except Exception as exc:
        print(f"    - Venue ID method failed: {exc}")

    try:
        print("    - Falling back to raw OpenReview API for Venue ID")
        params = {"content.venueid": conference_id}
        if details:
            params["details"] = details
        raw_notes = _get_all_notes_raw(params)
        papers, embedded_replies_found = _extract_paper_data_from_iterator(
            tqdm(raw_notes, desc="Processing papers (Raw Venue ID)"),
            include_replies=include_replies,
        )
        if papers:
            return papers, embedded_replies_found
    except Exception as exc:
        print(f"    - Raw Venue ID method failed: {exc}")

    return [], False


def main(config: Dict[str, Any], raw_data_dir: Path) -> Path | None:
    """Main function to be called as a module for fetching paper data."""
    conference_id = config["conference_id"]
    fetch_reviews = config.get("fetch_reviews", False)
    limit = config.get("limit")

    ensure_directories(raw_data_dir)

    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    submissions_data, embedded_replies_found = get_all_papers(
        client,
        conference_id,
        include_replies=fetch_reviews,
    )

    if not submissions_data:
        print("\nCould not fetch any paper data.")
        return None

    if limit:
        print(f"\nLimiting to the first {limit} papers as requested.")
        submissions_data = submissions_data[:limit]

    if fetch_reviews and not embedded_replies_found:
        print("\nEmbedded replies were not available. Falling back to per-paper review fetching...")
        final_data = get_rich_paper_details(client, submissions_data)
    else:
        final_data = submissions_data

    print(f"\nProcessed {len(final_data)} papers.")
    df = pd.DataFrame(final_data)

    output_path = build_raw_output_path(
        conference_id=conference_id,
        raw_data_dir=raw_data_dir,
        fetch_reviews=fetch_reviews,
        limit=limit,
    )
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Data successfully saved to: {output_path}")

    return output_path
