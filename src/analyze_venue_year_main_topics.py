"""Venue-year topic clustering for the main accepted-only paper view.

This is intentionally stricter than the older year-level run: each
conference-year is clustered independently so large venues or domains cannot
swallow nearby topics from other venues.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from analyze_yearly_main_topics import (
    DEFAULT_INPUT,
    build_topic_model,
    compute_topic_centroids,
    encode_documents,
    infer_doc_prefix,
    label_for_keywords,
    load_jsonl,
    multilabel_candidates,
    normalize_rows,
    reassign_outliers_to_nearest_topic,
    recluster_remaining_outliers,
    safe_model_name,
    topic_keywords,
)
from refine_topic_names import specific_label_cn, title_terms


DEFAULT_OUTPUT_DIR = "results/venue_year_main_accepted_topics_2020_2025_bge_m25"


def safe_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "unknown"


def group_topic_id(venue: str, year: int, topic_id: int) -> str:
    prefix = f"{safe_part(venue)}_{int(year)}"
    if int(topic_id) < 0:
        return f"{prefix}_T-001"
    return f"{prefix}_T{int(topic_id):03d}"


def dynamic_min_topic_size(group_size: int, base: int, floor: int, divisor: int) -> int:
    if group_size <= 0:
        return floor
    size = min(base, max(floor, group_size // divisor))
    if group_size < size * 2:
        size = max(2, group_size // 3)
    return max(2, size)


def force_assign_remaining_outliers(
    topics: np.ndarray,
    embeddings: np.ndarray,
    similarities: list[float | None],
    methods: list[str],
) -> tuple[np.ndarray, list[float | None], list[str]]:
    final_topics = topics.copy()
    remaining = np.flatnonzero(final_topics == -1)
    if len(remaining) == 0:
        return final_topics, similarities, methods

    centroids, topic_ids = compute_topic_centroids(final_topics, embeddings)
    if len(topic_ids) == 0:
        return final_topics, similarities, methods

    sim = cosine_similarity(embeddings[remaining], centroids)
    best_positions = sim.argmax(axis=1)
    best_scores = sim.max(axis=1)
    for row_id, position, score in zip(remaining, best_positions, best_scores, strict=True):
        final_topics[int(row_id)] = int(topic_ids[int(position)])
        similarities[int(row_id)] = float(score)
        methods[int(row_id)] = "nearest_centroid_fallback"
    return final_topics, similarities, methods


def representative_titles(
    papers: pd.DataFrame,
    embeddings: np.ndarray,
    topic_id: int,
    top_n: int,
) -> tuple[list[str], str]:
    idx = papers.index[papers["topic"].astype(int) == int(topic_id)].to_numpy()
    if len(idx) == 0:
        return [], ""
    topic_embeddings = embeddings[idx]
    centroid = topic_embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm:
        centroid = centroid / norm
    scores = topic_embeddings @ centroid
    order = np.argsort(scores)[::-1][:top_n]
    selected_titles = papers.iloc[idx[order]]["title"].fillna("").astype(str).tolist()
    return selected_titles, title_terms(selected_titles)


def summarize_group_topics(
    df_group: pd.DataFrame,
    embeddings: np.ndarray,
    venue: str,
    year: int,
    top_n_titles: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(df_group)
    for topic_id, count in Counter(df_group["topic"]).most_common():
        topic_df = df_group[df_group["topic"] == topic_id]
        keywords = topic_df["topic_keywords"].iloc[0] if len(topic_df) else ""
        parent, coarse_cn, coarse_en = label_for_keywords(keywords)
        titles, terms = representative_titles(df_group, embeddings, int(topic_id), top_n_titles)
        specific_cn, specific_parent, specific_en = specific_label_cn(keywords, titles)
        rows.append(
            {
                "venue": venue,
                "year": int(year),
                "topic": int(topic_id),
                "venue_year_topic_id": group_topic_id(venue, year, int(topic_id)),
                "count": int(count),
                "venue_year_total": int(total),
                "share": count / total if total else 0,
                "topic_keywords": keywords,
                "parent_category": parent,
                "topic_label_cn": coarse_cn,
                "topic_label_en": coarse_en,
                "specific_label_cn": specific_cn,
                "specific_parent_category": specific_parent,
                "specific_label_en": specific_en,
                "representative_titles": " || ".join(titles),
                "title_terms": terms,
                "naming_evidence": f"keywords={keywords}; title_terms={terms}",
            }
        )
    return pd.DataFrame(rows).sort_values(["venue", "year", "count"], ascending=[True, True, False])


def analyze_one_group(
    df_group: pd.DataFrame,
    venue: str,
    year: int,
    output_dir: Path,
    args: argparse.Namespace,
    doc_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    group_dir = output_dir / f"venue={safe_part(venue)}" / f"year={int(year)}"
    group_dir.mkdir(parents=True, exist_ok=True)

    docs = df_group["text_for_analysis"].tolist()
    model_safe = safe_model_name(args.model_name)
    embeddings_path = group_dir / f"embeddings_{model_safe}.npy"
    embeddings = encode_documents(docs, embeddings_path, args.model_name, args.batch_size, args.device, doc_prefix)

    min_topic_size = dynamic_min_topic_size(
        len(docs),
        args.min_topic_size,
        args.min_topic_size_floor,
        args.topic_size_divisor,
    )
    n_neighbors = min(args.n_neighbors, max(2, len(docs) - 1))
    topic_model = build_topic_model(min_topic_size, args.min_samples, n_neighbors)
    print(f"Fitting {venue} {year}: papers={len(docs):,}, min_topic_size={min_topic_size}")
    raw_topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    raw_topics_array = np.asarray(raw_topics, dtype=int)

    keyword_map = {
        topic_id: topic_keywords(topic_model, topic_id)
        for topic_id in sorted(set(int(topic) for topic in raw_topics_array if int(topic) >= 0))
    }
    final_topics, outlier_similarity, assignment_methods = reassign_outliers_to_nearest_topic(
        topic_model,
        raw_topics_array,
        embeddings,
        args.outlier_threshold,
    )
    if args.recluster_outliers:
        final_topics, assignment_methods, keyword_map = recluster_remaining_outliers(
            docs=docs,
            embeddings=embeddings,
            final_topics=final_topics,
            methods=assignment_methods,
            keyword_map=keyword_map,
            min_topic_size=args.secondary_min_topic_size,
            min_samples=args.secondary_min_samples,
            n_neighbors=args.n_neighbors,
        )
    if args.force_assign_remaining_outliers:
        final_topics, outlier_similarity, assignment_methods = force_assign_remaining_outliers(
            final_topics,
            embeddings,
            outlier_similarity,
            assignment_methods,
        )

    topic_candidates, topic_candidate_scores = multilabel_candidates(
        final_topics,
        embeddings,
        args.multi_label_top_k,
        args.multi_label_threshold,
    )

    df_out = df_group.copy()
    df_out["topic"] = final_topics
    df_out["raw_topic"] = raw_topics_array
    df_out["was_outlier"] = df_out["raw_topic"] == -1
    df_out["outlier_assignment_similarity"] = outlier_similarity
    df_out["topic_assignment_method"] = assignment_methods
    df_out["outlier_reassigned"] = df_out["was_outlier"] & (df_out["topic"] != -1)
    df_out["topic_keywords"] = df_out["topic"].map(lambda topic: keyword_map.get(int(topic), "outlier"))
    df_out["venue_year_topic_id"] = df_out["topic"].map(lambda topic: group_topic_id(venue, year, int(topic)))
    df_out["candidate_topics"] = topic_candidates
    df_out["candidate_topic_scores"] = topic_candidate_scores
    df_out["cluster_unit"] = "venue_year"

    topic_summary = summarize_group_topics(df_out, embeddings, venue, year, args.top_n_titles)
    topic_summary.to_csv(group_dir / "topic_summary.csv", index=False)
    topic_model.get_topic_info().to_csv(group_dir / "bertopic_raw_topic_info.csv", index=False)
    df_out.to_csv(group_dir / "papers_with_topics.csv", index=False)
    df_out.to_json(group_dir / "papers_with_topics.jsonl", orient="records", lines=True, force_ascii=False)

    report = {
        "venue": venue,
        "year": int(year),
        "papers": int(len(df_out)),
        "raw_outliers": int((df_out["raw_topic"] == -1).sum()),
        "final_outliers": int((df_out["topic"] == -1).sum()),
        "topics_excluding_outlier": int((topic_summary["topic"] != -1).sum()),
        "min_topic_size": int(min_topic_size),
        "model_name": args.model_name,
    }
    report["raw_outlier_rate"] = report["raw_outliers"] / report["papers"] if report["papers"] else 0
    report["final_outlier_rate"] = report["final_outliers"] / report["papers"] if report["papers"] else 0
    (group_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{venue} {year} done: raw_outliers={report['raw_outliers']:,}, "
        f"final_outliers={report['final_outliers']:,}, topics={report['topics_excluding_outlier']:,}"
    )
    return df_out, topic_summary, report


def write_outputs(
    output_dir: Path,
    all_papers: pd.DataFrame,
    all_topics: pd.DataFrame,
    reports: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    all_papers.to_csv(output_dir / "papers_with_venue_year_topics.csv", index=False)
    all_papers.to_json(output_dir / "papers_with_venue_year_topics.jsonl", orient="records", lines=True, force_ascii=False)
    all_topics.to_csv(output_dir / "topic_summary_by_venue_year.csv", index=False)

    report_df = pd.DataFrame(reports).sort_values(["year", "venue"])
    report_df.to_csv(output_dir / "run_summary_by_venue_year.csv", index=False)

    top_rows = []
    for (venue, year), group in all_topics.groupby(["venue", "year"], sort=True):
        for rank, (_, row) in enumerate(group.sort_values("count", ascending=False).head(args.top_k).iterrows(), 1):
            top_rows.append(
                {
                    "venue": venue,
                    "year": int(year),
                    "rank": rank,
                    "venue_year_topic_id": row["venue_year_topic_id"],
                    "count": int(row["count"]),
                    "venue_year_total": int(row["venue_year_total"]),
                    "share_pct": round(float(row["share"]) * 100, 2),
                    "specific_label_cn": row["specific_label_cn"],
                    "specific_parent_category": row["specific_parent_category"],
                    "short_keywords": ", ".join(str(row.get("topic_keywords", "")).split(", ")[:5]),
                    "representative_titles": row.get("representative_titles", ""),
                }
            )
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(output_dir / f"top{args.top_k}_topics_by_venue_year.csv", index=False)

    label_trend = (
        all_topics.groupby(["venue", "year", "specific_parent_category", "specific_label_cn"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    totals = all_papers.groupby(["venue", "year"], dropna=False).size().reset_index(name="venue_year_total")
    label_trend = label_trend.merge(totals, on=["venue", "year"], how="left")
    label_trend["share"] = label_trend["count"] / label_trend["venue_year_total"]
    label_trend.sort_values(["venue", "year", "count"], ascending=[True, True, False]).to_csv(
        output_dir / "label_trend_by_venue_year.csv",
        index=False,
    )

    lines = [
        "# Venue-Year Main Accepted Topic Analysis",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Input: `{args.input}`",
        f"- Papers analyzed: {len(all_papers):,}",
        f"- Embedding model: `{args.model_name}`",
        "- Clustering unit: independent conference-year BERTopic runs",
        f"- Outlier nearest-topic threshold: {args.outlier_threshold}",
        f"- Force assign remaining outliers: {args.force_assign_remaining_outliers}",
        "",
        "## Run Summary",
        "",
    ]
    display = report_df[
        [
            "venue",
            "year",
            "papers",
            "topics_excluding_outlier",
            "raw_outliers",
            "raw_outlier_rate",
            "final_outliers",
            "final_outlier_rate",
        ]
    ].copy()
    display["raw_outlier_rate"] = display["raw_outlier_rate"].map(lambda value: f"{value:.2%}")
    display["final_outlier_rate"] = display["final_outlier_rate"].map(lambda value: f"{value:.2%}")
    lines.append(display.to_markdown(index=False))
    lines.extend(["", f"## Top {args.top_k} Topics By Venue-Year", ""])
    for (venue, year), group in top_df.groupby(["venue", "year"], sort=True):
        lines.append(f"### {venue} {int(year)}")
        lines.append("")
        out = group[["rank", "specific_label_cn", "count", "venue_year_total", "share_pct", "short_keywords"]].copy()
        out.rename(
            columns={
                "rank": "排名",
                "specific_label_cn": "细主题名",
                "count": "篇数",
                "venue_year_total": "当年会议篇数",
                "share_pct": "占比%",
                "short_keywords": "关键词",
            },
            inplace=True,
        )
        lines.append(out.to_markdown(index=False))
        lines.append("")
    (output_dir / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conference-year main accepted topic clustering.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--doc-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--min-topic-size", type=int, default=25)
    parser.add_argument("--min-topic-size-floor", type=int, default=8)
    parser.add_argument("--topic-size-divisor", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--outlier-threshold", type=float, default=0.60)
    parser.add_argument("--recluster-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-assign-remaining-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--secondary-min-topic-size", type=int, default=8)
    parser.add_argument("--secondary-min-samples", type=int, default=3)
    parser.add_argument("--multi-label-top-k", type=int, default=3)
    parser.add_argument("--multi-label-threshold", type=float, default=0.42)
    parser.add_argument("--top-n-titles", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--years", nargs="*", type=int, default=[])
    parser.add_argument("--venues", nargs="*", default=[])
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--min-group-size", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.input))
    df = normalize_rows(records)
    if args.years:
        df = df[df["year"].isin(args.years)].copy()
    if args.venues:
        wanted = {venue.upper() for venue in args.venues}
        df = df[df["venue"].astype(str).str.upper().isin(wanted)].copy()
    if args.limit_per_group:
        df = df.groupby(["venue", "year"], group_keys=False).head(args.limit_per_group).copy()

    doc_prefix = infer_doc_prefix(args.model_name, args.doc_prefix)

    all_papers = []
    all_topic_summaries = []
    reports = []
    groups = df.groupby(["year", "venue"], sort=True)
    for (year, venue), df_group in groups:
        df_group = df_group.copy().reset_index(drop=True)
        if len(df_group) < args.min_group_size:
            print(f"Skipping {venue} {year}: only {len(df_group)} papers")
            continue
        group_papers, group_topics, report = analyze_one_group(
            df_group,
            str(venue),
            int(year),
            output_dir,
            args,
            doc_prefix,
        )
        all_papers.append(group_papers)
        all_topic_summaries.append(group_topics)
        reports.append(report)

    combined_papers = pd.concat(all_papers, ignore_index=True) if all_papers else pd.DataFrame()
    combined_topics = pd.concat(all_topic_summaries, ignore_index=True) if all_topic_summaries else pd.DataFrame()
    write_outputs(output_dir, combined_papers, combined_topics, reports, args)
    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
