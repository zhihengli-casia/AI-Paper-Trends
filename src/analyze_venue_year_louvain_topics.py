"""Venue-year topic clustering with kNN graph + Louvain communities.

This is the graph-community alternative to the BERTopic/HDBSCAN pipeline. It
keeps the same input/output shape where practical so downstream summaries can
be compared directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from analyze_venue_year_main_topics import (
    dynamic_min_topic_size,
    group_topic_id,
    representative_titles,
    safe_part,
)
from analyze_yearly_main_topics import (
    DEFAULT_STOP_WORDS,
    encode_documents,
    infer_doc_prefix,
    label_for_keywords,
    load_jsonl,
    multilabel_candidates,
    normalize_rows,
    safe_model_name,
)
from refine_topic_names import specific_label_cn


DEFAULT_INPUT = "data/recent_conferences/main_accepted_ai_conference_papers_2020_2026_v1.jsonl"
DEFAULT_OUTPUT_DIR = "results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25"


def ctfidf_keywords(docs: list[str], topics: np.ndarray, top_n: int = 10) -> dict[int, str]:
    """Return c-TF-IDF-like keywords for each topic."""

    topic_ids = sorted(int(topic) for topic in set(topics) if int(topic) >= 0)
    if not topic_ids:
        return {}

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.85,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b",
    )
    stop_words = set(vectorizer.get_stop_words() or [])
    vectorizer.stop_words = list(stop_words.union(DEFAULT_STOP_WORDS))
    doc_term = vectorizer.fit_transform(docs)
    terms = np.asarray(vectorizer.get_feature_names_out())

    topic_rows = []
    for topic_id in topic_ids:
        idx = np.flatnonzero(topics == topic_id)
        topic_rows.append(sparse.csr_matrix(doc_term[idx].sum(axis=0)))
    topic_term = sparse.vstack(topic_rows).astype("float64")

    row_sums = np.asarray(topic_term.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    tf = topic_term.multiply(1.0 / row_sums[:, None])

    term_totals = np.asarray(topic_term.sum(axis=0)).ravel()
    avg_topic_len = float(row_sums.mean()) if len(row_sums) else 1.0
    idf = np.log((avg_topic_len + 1.0) / (term_totals + 1.0)) + 1.0
    scores = tf.multiply(idf)

    keyword_map: dict[int, str] = {}
    for row_idx, topic_id in enumerate(topic_ids):
        row = scores.getrow(row_idx)
        if row.nnz == 0:
            keyword_map[topic_id] = ""
            continue
        order = np.argsort(row.data)[::-1][:top_n]
        keyword_map[topic_id] = ", ".join(terms[row.indices[order]].tolist())
    return keyword_map


def build_knn_graph(embeddings: np.ndarray, n_neighbors: int, min_similarity: float) -> nx.Graph:
    n = len(embeddings)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    if n <= 1:
        return graph

    k = min(max(2, n_neighbors + 1), n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    for i in range(n):
        for distance, j in zip(distances[i][1:], indices[i][1:], strict=False):
            similarity = 1.0 - float(distance)
            if similarity < min_similarity:
                continue
            if i == int(j):
                continue
            old_weight = graph[i].get(int(j), {}).get("weight", 0.0)
            if similarity > old_weight:
                graph.add_edge(i, int(j), weight=similarity)
    return graph


def merge_small_communities(
    topics: np.ndarray,
    embeddings: np.ndarray,
    min_topic_size: int,
) -> tuple[np.ndarray, dict[int, str]]:
    """Merge communities smaller than min_topic_size into nearest large topic."""

    merged = topics.copy()
    counts = Counter(int(t) for t in merged)
    large = sorted(topic for topic, count in counts.items() if count >= min_topic_size)
    if not large:
        return np.zeros_like(merged), {int(t): "single_topic_fallback" for t in set(merged)}

    large_centroids = []
    for topic in large:
        centroid = embeddings[merged == topic].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        large_centroids.append(centroid)
    large_matrix = np.vstack(large_centroids)

    methods: dict[int, str] = {topic: "louvain" for topic in large}
    for topic, count in counts.items():
        if topic in large:
            continue
        idx = np.flatnonzero(merged == topic)
        centroid = embeddings[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        sims = large_matrix @ centroid
        target = large[int(np.argmax(sims))]
        merged[idx] = target
        methods[int(topic)] = f"merged_small_to_{target}"

    # Renumber by descending size for stable readable topic IDs.
    ordered = [topic for topic, _ in Counter(int(t) for t in merged).most_common()]
    mapping = {old: new for new, old in enumerate(ordered)}
    renumbered = np.asarray([mapping[int(t)] for t in merged], dtype=int)
    return renumbered, methods


def louvain_topics(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_similarity: float,
    resolution: float,
    seed: int,
    min_topic_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    graph = build_knn_graph(embeddings, n_neighbors, min_similarity)
    if graph.number_of_edges() == 0:
        topics = np.zeros(len(embeddings), dtype=int)
        return topics, {"graph_nodes": graph.number_of_nodes(), "graph_edges": 0, "raw_communities": 1}

    communities = nx.algorithms.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )
    raw_topics = np.full(len(embeddings), -1, dtype=int)
    for topic_id, community in enumerate(sorted(communities, key=len, reverse=True)):
        for node in community:
            raw_topics[int(node)] = int(topic_id)
    # Isolated nodes are assigned to nearest raw community before small merges.
    if np.any(raw_topics < 0):
        valid = sorted(topic for topic in set(raw_topics) if int(topic) >= 0)
        centroids = []
        for topic in valid:
            centroid = embeddings[raw_topics == topic].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm:
                centroid = centroid / norm
            centroids.append(centroid)
        matrix = np.vstack(centroids)
        missing = np.flatnonzero(raw_topics < 0)
        sims = embeddings[missing] @ matrix.T
        for row, position in zip(missing, sims.argmax(axis=1), strict=True):
            raw_topics[int(row)] = int(valid[int(position)])

    final_topics, merge_methods = merge_small_communities(raw_topics, embeddings, min_topic_size)
    metadata = {
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "raw_communities": len(communities),
        "merge_methods": merge_methods,
    }
    return final_topics, metadata


def same_group_order(current: pd.DataFrame, previous_path: Path) -> bool:
    if not previous_path.exists():
        return False
    previous = pd.read_csv(previous_path, usecols=lambda c: c in {"paper_id", "title"})
    if len(previous) != len(current):
        return False
    if "paper_id" in previous.columns and "paper_id" in current.columns:
        return previous["paper_id"].fillna("").astype(str).tolist() == current["paper_id"].fillna("").astype(str).tolist()
    return previous["title"].fillna("").astype(str).tolist() == current["title"].fillna("").astype(str).tolist()


def load_or_encode_embeddings(
    df_group: pd.DataFrame,
    docs: list[str],
    group_dir: Path,
    args: argparse.Namespace,
    doc_prefix: str,
) -> np.ndarray:
    model_safe = safe_model_name(args.model_name)
    embeddings_path = group_dir / f"embeddings_{model_safe}.npy"
    if embeddings_path.exists():
        return np.load(embeddings_path)

    if args.reuse_embeddings_dir:
        previous_group = Path(args.reuse_embeddings_dir) / group_dir.relative_to(Path(args.output_dir))
        previous_embeddings = previous_group / f"embeddings_{model_safe}.npy"
        previous_papers = previous_group / "papers_with_topics.csv"
        if previous_embeddings.exists() and same_group_order(df_group, previous_papers):
            group_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous_embeddings, embeddings_path)
            print(f"Reused embeddings: {previous_embeddings}")
            return np.load(embeddings_path)

    return encode_documents(docs, embeddings_path, args.model_name, args.batch_size, args.device, doc_prefix)


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
    embeddings = load_or_encode_embeddings(df_group, docs, group_dir, args, doc_prefix)
    min_topic_size = dynamic_min_topic_size(
        len(docs),
        args.min_topic_size,
        args.min_topic_size_floor,
        args.topic_size_divisor,
    )
    topics, graph_meta = louvain_topics(
        embeddings,
        n_neighbors=min(args.n_neighbors, max(2, len(docs) - 1)),
        min_similarity=args.min_similarity,
        resolution=args.resolution,
        seed=args.seed,
        min_topic_size=min_topic_size,
    )
    keyword_map = ctfidf_keywords(docs, topics, top_n=args.top_n_words)
    topic_candidates, topic_candidate_scores = multilabel_candidates(
        topics,
        embeddings,
        args.multi_label_top_k,
        args.multi_label_threshold,
    )

    df_out = df_group.copy()
    df_out["topic"] = topics
    df_out["raw_topic"] = topics
    df_out["was_outlier"] = False
    df_out["outlier_assignment_similarity"] = None
    df_out["topic_assignment_method"] = "louvain"
    df_out["outlier_reassigned"] = False
    df_out["topic_keywords"] = df_out["topic"].map(lambda topic: keyword_map.get(int(topic), ""))
    df_out["venue_year_topic_id"] = df_out["topic"].map(lambda topic: group_topic_id(venue, year, int(topic)))
    df_out["candidate_topics"] = topic_candidates
    df_out["candidate_topic_scores"] = topic_candidate_scores
    df_out["cluster_unit"] = "venue_year"
    df_out["cluster_algorithm"] = "knn_louvain"

    topic_summary = summarize_group_topics(df_out, embeddings, venue, year, args.top_n_titles)
    topic_summary.to_csv(group_dir / "topic_summary.csv", index=False)
    df_out.to_csv(group_dir / "papers_with_topics.csv", index=False)
    df_out.to_json(group_dir / "papers_with_topics.jsonl", orient="records", lines=True, force_ascii=False)

    report = {
        "venue": venue,
        "year": int(year),
        "papers": int(len(df_out)),
        "raw_outliers": 0,
        "final_outliers": 0,
        "topics_excluding_outlier": int(topic_summary["topic"].nunique()),
        "min_topic_size": int(min_topic_size),
        "model_name": args.model_name,
        "algorithm": "knn_louvain",
        "n_neighbors": int(args.n_neighbors),
        "min_similarity": float(args.min_similarity),
        "resolution": float(args.resolution),
        **graph_meta,
    }
    report["raw_outlier_rate"] = 0.0
    report["final_outlier_rate"] = 0.0
    (group_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{venue} {year} done: papers={report['papers']:,}, "
        f"topics={report['topics_excluding_outlier']:,}, graph_edges={report['graph_edges']:,}"
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

    report_df = pd.DataFrame(reports).sort_values(["venue", "year"])
    report_df.to_csv(output_dir / "run_summary_by_venue_year.csv", index=False)

    top_rows = []
    for (venue, year), group in all_topics.groupby(["venue", "year"], sort=True):
        total = int(group["venue_year_total"].iloc[0]) if len(group) else 0
        for rank, (_, row) in enumerate(group.sort_values("count", ascending=False).head(args.top_k).iterrows(), 1):
            top_rows.append(
                {
                    "venue": venue,
                    "year": int(year),
                    "rank": int(rank),
                    "venue_year_topic_id": row["venue_year_topic_id"],
                    "count": int(row["count"]),
                    "venue_year_total": total,
                    "share_pct": round(float(row["share"]) * 100, 2),
                    "specific_label_cn": row["specific_label_cn"],
                    "specific_parent_category": row["specific_parent_category"],
                    "short_keywords": ", ".join(str(row["topic_keywords"]).split(", ")[:5]),
                    "representative_titles": row["representative_titles"],
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
        "# Venue-Year Main Accepted Topic Analysis (kNN Louvain)",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Input: `{args.input}`",
        f"- Papers analyzed: {len(all_papers):,}",
        f"- Embedding model: `{args.model_name}`",
        "- Clustering unit: independent venue-year kNN graph + Louvain communities",
        f"- n_neighbors={args.n_neighbors}, min_similarity={args.min_similarity}, resolution={args.resolution}",
        "",
        "## Run Summary",
        "",
        report_df[["venue", "year", "papers", "topics_excluding_outlier", "graph_edges", "raw_communities"]].to_markdown(index=False),
        "",
        "## Top Topics",
        "",
    ]
    for (venue, year), group in top_df.groupby(["venue", "year"], sort=True):
        display = group[["rank", "specific_label_cn", "count", "venue_year_total", "share_pct", "short_keywords"]].copy()
        display.rename(
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
        lines.append(f"### {venue} {int(year)}")
        lines.append("")
        lines.append(display.to_markdown(index=False))
        lines.append("")
    (output_dir / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run venue-year kNN Louvain topic clustering.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-embeddings-dir", default="")
    parser.add_argument("--model-name", default="models/bge-base-en-v1.5")
    parser.add_argument("--doc-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--min-topic-size", type=int, default=25)
    parser.add_argument("--min-topic-size-floor", type=int, default=8)
    parser.add_argument("--topic-size-divisor", type=int, default=12)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.25)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--top-n-titles", type=int, default=5)
    parser.add_argument("--multi-label-top-k", type=int, default=3)
    parser.add_argument("--multi-label-threshold", type=float, default=0.42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--years", nargs="*", type=int, default=[])
    parser.add_argument("--venues", nargs="*", default=[])
    parser.add_argument("--limit-per-group", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.input))
    df = normalize_rows(records)
    if args.years:
        df = df[df["year"].isin(args.years)].copy()
    if args.venues:
        allowed = {venue.upper() for venue in args.venues}
        df = df[df["venue"].astype(str).str.upper().isin(allowed)].copy()
    if args.limit_per_group:
        df = df.groupby(["venue", "year"], group_keys=False).head(args.limit_per_group).copy()
    doc_prefix = infer_doc_prefix(args.model_name, args.doc_prefix)

    all_papers = []
    all_topics = []
    reports = []
    for (venue, year), df_group in df.groupby(["venue", "year"], sort=True):
        group_papers, group_topics, report = analyze_one_group(
            df_group.copy().reset_index(drop=True),
            str(venue),
            int(year),
            output_dir,
            args,
            doc_prefix,
        )
        all_papers.append(group_papers)
        all_topics.append(group_topics)
        reports.append(report)

    combined_papers = pd.concat(all_papers, ignore_index=True) if all_papers else pd.DataFrame()
    combined_topics = pd.concat(all_topics, ignore_index=True) if all_topics else pd.DataFrame()
    write_outputs(output_dir, combined_papers, combined_topics, reports, args)
    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
