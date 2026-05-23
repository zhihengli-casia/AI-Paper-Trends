"""Build a stricter main-conference accepted-only analysis view.

This script normalizes the mixed crawler output into a comparable view for
topic analysis. It keeps only accepted main-conference papers for the selected
venues and annotates why each retained paper was included.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/recent_conferences/recent_ai_conference_papers_v9_with_kdd_sigir_www_chi_colm.jsonl"
DEFAULT_OUTPUT = "data/recent_conferences/main_accepted_ai_conference_papers_v1.jsonl"

CV_VENUES = {"CVPR", "ICCV", "ECCV"}
ML_VENUES = {"ICLR", "ICML", "NeurIPS"}
NLP_VENUES = {"ACL", "EMNLP", "NAACL"}
EXPANDED_MAIN_VENUES = {"KDD", "SIGIR", "WWW"}
OTHER_VENUES = {"AAAI", "ACMMM", "IJCAI"}

SELECTED_VENUES = CV_VENUES | ML_VENUES | NLP_VENUES | EXPANDED_MAIN_VENUES | OTHER_VENUES

NLP_MAIN_PREFIXES = {
    "ACL": {"acl-main", "acl-long", "acl-short"},
    "EMNLP": {"emnlp-main"},
    "NAACL": {"naacl-main", "naacl-long", "naacl-short"},
}

EXCLUDED_SIGIR_TRACK_KEYWORDS = (
    "short",
    "resource",
    "sirip",
    "reproducibility",
    "perspective",
    "doctoral",
    "demo",
    "tutorial",
    "workshop",
)

EXCLUDED_WWW_TRACK_KEYWORDS = (
    "poster",
    "companion",
    "demo",
    "doctoral",
    "tutorial",
    "workshop",
    "web4good",
    "challenge",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def acl_anthology_prefix(record: dict[str, Any]) -> str:
    url = str(record.get("paper_id") or record.get("html_url") or "")
    match = re.search(r"aclanthology\.org/\d{4}\.([^./]+)", url)
    return match.group(1).lower() if match else ""


def is_iclr_accepted(record: dict[str, Any]) -> bool:
    status = str(record.get("status") or "").lower()
    if "withdrawn" in status or "desk rejected" in status or "submitted to" in status or "submitted" in status or "reject" in status:
        return False
    return any(token in status for token in ("poster", "spotlight", "oral", "notable", "accept", "iclr 2020"))


def classify_record(record: dict[str, Any]) -> tuple[bool, str, str]:
    """Return (include, publication_type, inclusion_reason)."""

    venue = str(record.get("venue") or "")
    status = str(record.get("status") or "").lower()
    track = str(record.get("track") or "")
    track_lower = track.lower()

    if venue not in SELECTED_VENUES:
        return False, "excluded", "venue_not_selected"

    if venue in CV_VENUES:
        return status == "accepted", "main", "cvf_ecva_accepted_main"

    if venue == "AAAI":
        return status == "accepted", "main", "aaai_ojs_technical_track"

    if venue == "ACMMM":
        return status == "accepted", "main", "acmmm_proceedings_main"

    if venue == "IJCAI":
        return status == "accepted", "main", "ijcai_proceedings_main"

    if venue == "ICLR":
        return is_iclr_accepted(record), "main", "iclr_oral_spotlight_poster"

    if venue == "ICML":
        return status == "accepted", "main", "icml_pmlr_accepted"

    if venue == "NeurIPS":
        # Older NeurIPS proceedings pages in this dataset do not expose track
        # metadata. They are already accepted proceedings records, so keep them
        # instead of dropping 2020/2021.
        source = str(record.get("source") or "").lower()
        has_main_track = track == "Main Conference Track"
        is_proceedings_without_track = "neurips proceedings" in source and track_lower in {"", "none", "main", "main conference"}
        include = status == "accepted" and (has_main_track or is_proceedings_without_track)
        reason = "neurips_main_conference_track" if has_main_track else "neurips_proceedings_no_track"
        return include, "main", reason

    if venue in NLP_VENUES:
        prefix = acl_anthology_prefix(record)
        include = status == "accepted" and prefix in NLP_MAIN_PREFIXES[venue]
        return include, "main_long_short", f"nlp_main_prefix:{prefix or 'missing'}"

    if venue == "KDD":
        include = status == "accepted" and "research track" in track_lower
        return include, "main_research", "kdd_research_track"

    if venue == "SIGIR":
        include = status == "accepted" and not any(word in track_lower for word in EXCLUDED_SIGIR_TRACK_KEYWORDS)
        return include, "main_full_or_session", "sigir_exclude_short_resource_special_tracks"

    if venue == "WWW":
        include = status == "accepted" and not any(word in track_lower for word in EXCLUDED_WWW_TRACK_KEYWORDS)
        return include, "main_research", "www_research_track_excluding_special_tracks"

    return False, "excluded", "no_rule"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = defaultdict(lambda: {"count": 0, "abstracts": 0})
    for row in rows:
        key = (row.get("venue"), int(row.get("year", 0)))
        summary[key]["count"] += 1
        if str(row.get("abstract") or "").strip():
            summary[key]["abstracts"] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["venue", "year", "count", "abstracts", "abstract_rate"])
        writer.writeheader()
        for (venue, year), values in sorted(summary.items(), key=lambda item: (item[0][0], item[0][1])):
            count = values["count"]
            abstracts = values["abstracts"]
            writer.writerow(
                {
                    "venue": venue,
                    "year": year,
                    "count": count,
                    "abstracts": abstracts,
                    "abstract_rate": f"{abstracts / count:.4f}" if count else "0.0000",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build main accepted-only paper view.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_jsonl(input_path)

    included: list[dict[str, Any]] = []
    included_counts = Counter()
    excluded_counts = Counter()
    reason_counts = Counter()

    for record in records:
        include, publication_type, reason = classify_record(record)
        venue = str(record.get("venue") or "")
        if include:
            row = record.copy()
            row["decision_type"] = "accepted"
            row["publication_type"] = publication_type
            row["analysis_view"] = "main_accepted"
            row["inclusion_reason"] = reason
            row["original_status"] = record.get("status")
            row["original_track"] = record.get("track")
            included.append(row)
            included_counts[venue] += 1
        else:
            excluded_counts[venue] += 1
        reason_counts[(venue, reason)] += 1

    write_jsonl(output_path, included)
    write_summary(output_path.with_name(output_path.stem + "_summary.csv"), included)

    print(f"Input records: {len(records):,}")
    print(f"Included records: {len(included):,}")
    print(f"Output: {output_path}")
    print("\nIncluded by venue:")
    for venue, count in sorted(included_counts.items()):
        print(f"  {venue}: {count:,}")
    print("\nExcluded selected venues by venue:")
    for venue, count in sorted(excluded_counts.items()):
        if venue in SELECTED_VENUES:
            print(f"  {venue}: {count:,}")
    print("\nTop inclusion/exclusion reasons:")
    for (venue, reason), count in reason_counts.most_common(30):
        print(f"  {venue or '<missing>'} | {reason}: {count:,}")


if __name__ == "__main__":
    main()
