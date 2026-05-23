"""Run topic clustering for AI conference papers.

The script is intentionally standalone so the large recent-conference dataset can
be analyzed without changing the older OpenReview-only pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_proxy_env() -> None:
    """Avoid BERTopic/litellm importing httpx with an unsupported socks5h proxy."""

    for key in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)


normalize_proxy_env()

from bertopic import BERTopic  # noqa: E402
from hdbscan import HDBSCAN  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from umap import UMAP  # noqa: E402


LATEST_YEAR_BY_VENUE = {
    "CVPR": 2025,
    "ICCV": 2025,
    "ECCV": 2024,
    "ACL": 2025,
    "EMNLP": 2025,
    "NAACL": 2025,
    "ICLR": 2026,
    "ICML": 2025,
    "NeurIPS": 2025,
    "AAAI": 2026,
    "IJCAI": 2025,
}


TIMELINE = {
    ("ECCV", 2024): (0, "2024-09 ECCV 2024"),
    ("NAACL", 2025): (1, "2025-04 NAACL 2025"),
    ("CVPR", 2025): (2, "2025-06 CVPR 2025"),
    ("ICML", 2025): (3, "2025-07 ICML 2025"),
    ("ACL", 2025): (4, "2025-07 ACL 2025"),
    ("IJCAI", 2025): (5, "2025-08 IJCAI 2025"),
    ("ICCV", 2025): (6, "2025-10 ICCV 2025"),
    ("EMNLP", 2025): (7, "2025-11 EMNLP 2025"),
    ("NeurIPS", 2025): (8, "2025-12 NeurIPS 2025"),
    ("AAAI", 2026): (9, "2026-01 AAAI 2026"),
    ("ICLR", 2026): (10, "2026-04 ICLR 2026"),
}

VENUE_MONTH = {
    "AAAI": 1,
    "NAACL": 4,
    "ICLR": 4,
    "CVPR": 6,
    "ACL": 7,
    "ICML": 7,
    "IJCAI": 8,
    "ECCV": 9,
    "ICCV": 10,
    "EMNLP": 11,
    "NeurIPS": 12,
}


DEFAULT_STOP_WORDS = [
    "paper",
    "papers",
    "model",
    "models",
    "method",
    "methods",
    "approach",
    "approaches",
    "task",
    "tasks",
    "dataset",
    "datasets",
    "data",
    "learning",
    "neural",
    "network",
    "networks",
    "performance",
    "proposed",
    "propose",
    "based",
    "using",
    "show",
    "shows",
    "demonstrate",
    "results",
    "state",
    "art",
    "large",
    "new",
    "effective",
]


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def text_for_analysis(record: dict[str, Any]) -> str:
    keywords = record.get("keywords") or []
    if isinstance(keywords, list):
        keywords_text = " ".join(str(item) for item in keywords)
    else:
        keywords_text = str(keywords)
    return clean_text(
        f"{record.get('title', '')}. {keywords_text}. {record.get('abstract', '')}"
    )


def select_latest_cycle(records: list[dict[str, Any]]) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for record in records:
        venue = record.get("venue")
        year = int(record.get("year", 0))
        if LATEST_YEAR_BY_VENUE.get(venue) != year:
            continue
        order, label = TIMELINE.get((venue, year), (999, f"{year} {venue}"))
        row = record.copy()
        row["timeline_order"] = order
        row["timeline_label"] = label
        row["text_for_analysis"] = text_for_analysis(record)
        selected.append(row)

    df = pd.DataFrame(selected)
    df = df[df["text_for_analysis"].str.len() > 40].copy()
    df.sort_values(["timeline_order", "venue", "title"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def timeline_for_record(record: dict[str, Any]) -> tuple[int, str]:
    venue = str(record.get("venue") or "")
    year = int(record.get("year", 0))
    if (venue, year) in TIMELINE:
        return TIMELINE[(venue, year)]
    month = VENUE_MONTH.get(venue, 12)
    return year * 100 + month, f"{year}-{month:02d} {venue} {year}"


def select_all_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for record in records:
        order, label = timeline_for_record(record)
        row = record.copy()
        row["timeline_order"] = order
        row["timeline_label"] = label
        row["text_for_analysis"] = text_for_analysis(record)
        selected.append(row)

    df = pd.DataFrame(selected)
    df = df[df["text_for_analysis"].str.len() > 40].copy()
    df.sort_values(["timeline_order", "venue", "title"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def select_scope(records: list[dict[str, Any]], scope: str) -> pd.DataFrame:
    if scope == "latest":
        return select_latest_cycle(records)
    if scope == "all":
        return select_all_records(records)
    raise ValueError(f"Unsupported scope: {scope}")


def encode_documents(
    docs: list[str],
    embeddings_path: Path,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    if embeddings_path.exists():
        print(f"Using cached embeddings: {embeddings_path}")
        return np.load(embeddings_path)

    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name, device=device)
    print(f"Encoding {len(docs)} documents with batch_size={batch_size}, device={embedding_model.device}")
    embeddings = embedding_model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved: {embeddings_path}")
    return embeddings


def build_topic_model(min_topic_size: int, min_samples: int, n_neighbors: int) -> BERTopic:
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.65,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b",
    )
    stop_words = set(vectorizer_model.get_stop_words() or [])
    vectorizer_model.stop_words = list(stop_words.union(DEFAULT_STOP_WORDS))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    return BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=False,
        verbose=True,
    )


def topic_keywords(topic_model: BERTopic, topic_id: int, top_n: int = 8) -> str:
    if topic_id == -1:
        return "outlier"
    words = topic_model.get_topic(topic_id) or []
    return ", ".join(word for word, _ in words[:top_n])


def add_topic_columns(df: pd.DataFrame, topic_model: BERTopic, topics: list[int]) -> pd.DataFrame:
    result = df.copy()
    result["topic"] = topics
    keyword_map = {topic: topic_keywords(topic_model, int(topic)) for topic in sorted(set(topics))}
    result["topic_keywords"] = result["topic"].map(keyword_map)
    result["topic_label"] = result.apply(
        lambda row: f"T{int(row['topic']):03d}: {row['topic_keywords']}" if int(row["topic"]) >= 0 else "T-001: outlier",
        axis=1,
    )
    return result


def reduce_outliers_with_embeddings(
    topic_model: BERTopic,
    topics: list[int],
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[int], list[float | None]]:
    """Reassign HDBSCAN outliers to their nearest topic if cosine similarity is high enough."""

    topic_array = np.asarray(topics, dtype=int)
    new_topics = topic_array.copy()
    assignment_similarity: list[float | None] = [None] * len(topics)
    outlier_ids = np.flatnonzero(topic_array == -1)
    if len(outlier_ids) == 0:
        return new_topics.tolist(), assignment_similarity

    # This mirrors BERTopic.reduce_outliers(strategy="embeddings"), but also keeps
    # the similarity score so downstream tables can audit forced assignments.
    topic_embeddings = topic_model.topic_embeddings_[topic_model._outliers :]
    similarity = cosine_similarity(embeddings[outlier_ids], topic_embeddings)
    best_topics = similarity.argmax(axis=1)
    best_scores = similarity.max(axis=1)

    for row_id, topic_id, score in zip(outlier_ids, best_topics, best_scores, strict=True):
        assignment_similarity[int(row_id)] = float(score)
        if score >= threshold:
            new_topics[int(row_id)] = int(topic_id)

    return new_topics.tolist(), assignment_similarity


def summarize_topics(df_topics: pd.DataFrame, topic_model: BERTopic) -> pd.DataFrame:
    rows = []
    total = len(df_topics)
    for topic_id, count in Counter(df_topics["topic"]).most_common():
        topic_df = df_topics[df_topics["topic"] == topic_id]
        venue_counts = topic_df["venue"].value_counts().head(5)
        domain_counts = topic_df["domain"].value_counts().head(5)
        status_counts = topic_df["status"].value_counts().head(5)
        rows.append(
            {
                "topic": int(topic_id),
                "count": int(count),
                "share": count / total,
                "keywords": topic_keywords(topic_model, int(topic_id), top_n=12),
                "top_venues": "; ".join(f"{k}:{v}" for k, v in venue_counts.items()),
                "top_domains": "; ".join(f"{k}:{v}" for k, v in domain_counts.items()),
                "top_statuses": "; ".join(f"{k}:{v}" for k, v in status_counts.items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["topic"])


def topic_by_group(df_topics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df_topics.groupby(group_cols + ["topic", "topic_keywords"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = df_topics.groupby(group_cols, dropna=False).size().reset_index(name="group_total")
    grouped = grouped.merge(totals, on=group_cols, how="left")
    grouped["share"] = grouped["count"] / grouped["group_total"]
    return grouped.sort_values(group_cols + ["count"], ascending=[True] * len(group_cols) + [False])


def is_accepted_status(status: Any) -> bool:
    text = str(status or "").lower()
    if text == "accepted":
        return True
    return any(token in text for token in (" poster", "spotlight", " oral"))


def summarize_topic_subset(df_topics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df_topics)
    for topic_id, count in Counter(df_topics["topic"]).most_common():
        topic_df = df_topics[df_topics["topic"] == topic_id]
        venue_counts = topic_df["venue"].value_counts().head(8)
        rows.append(
            {
                "topic": int(topic_id),
                "count": int(count),
                "share": count / total if total else 0,
                "topic_keywords": topic_df["topic_keywords"].iloc[0] if len(topic_df) else "",
                "top_venues": "; ".join(f"{k}:{v}" for k, v in venue_counts.items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "topic"], ascending=[False, True])


def write_report(output_dir: Path, df_topics: pd.DataFrame, topic_summary: pd.DataFrame, scope: str) -> None:
    lines: list[str] = []
    title = "All AI Conference Topic Analysis" if scope == "all" else "Recent AI Conference Cycle Topic Analysis"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Papers analyzed: {len(df_topics):,}")
    lines.append(f"- Non-outlier papers: {(df_topics['topic'] != -1).sum():,}")
    lines.append(f"- Topics excluding outlier: {(topic_summary['topic'] != -1).sum():,}")
    lines.append("")
    lines.append("## Timeline Coverage")
    lines.append("")
    timeline_counts = (
        df_topics.groupby(["timeline_order", "timeline_label", "venue", "year"])
        .size()
        .reset_index(name="count")
        .sort_values("timeline_order")
    )
    if scope == "all":
        year_counts = df_topics.groupby("year").size().reset_index(name="count").sort_values("year")
        lines.append(year_counts.to_markdown(index=False))
    else:
        lines.append(timeline_counts[["timeline_label", "count"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Top Topics Overall")
    lines.append("")
    top_topics = topic_summary[topic_summary["topic"] != -1].sort_values("count", ascending=False).head(40)
    lines.append(top_topics[["topic", "count", "share", "keywords", "top_venues"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Top Topics by Timeline")
    lines.append("")
    by_time = topic_by_group(df_topics[df_topics["topic"] != -1], ["timeline_order", "timeline_label"])
    max_groups = None if scope == "latest" else 12
    for _, group in timeline_counts.head(max_groups).iterrows():
        label = group["timeline_label"]
        sub = by_time[by_time["timeline_label"] == label].head(12)
        lines.append(f"### {label}")
        lines.append("")
        if len(sub):
            lines.append(sub[["topic", "count", "share", "topic_keywords"]].to_markdown(index=False))
        else:
            lines.append("_No topics._")
        lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_chinese_report(
    output_dir: Path,
    df_topics: pd.DataFrame,
    topic_summary: pd.DataFrame,
    scope: str,
    input_path: str,
    min_topic_size: int,
    min_samples: int,
    n_neighbors: int,
) -> None:
    lines: list[str] = []
    title = "全量 AI 顶会主题聚类分析 - 验收报告" if scope == "all" else "最近一轮 AI 顶会主题聚类分析 - 验收报告"
    lines.append(f"# {title}")
    lines.append("")
    non_outlier = int((df_topics["topic"] != -1).sum())
    outlier = int((df_topics["topic"] == -1).sum())
    lines.append("## 运行配置")
    lines.append("")
    lines.append(f"- 输入数据：`{input_path}`")
    lines.append(f"- 分析论文数：{len(df_topics):,}")
    lines.append("- embedding：`sentence-transformers/all-MiniLM-L6-v2`，384 维")
    lines.append(
        f"- 聚类：BERTopic + UMAP + HDBSCAN，`min_topic_size={min_topic_size}`, "
        f"`min_samples={min_samples}`, `n_neighbors={n_neighbors}`"
    )
    lines.append(f"- 非离群主题数：{int((topic_summary['topic'] != -1).sum()):,}")
    lines.append(f"- 离群论文：{outlier:,} / {len(df_topics):,} = {outlier / len(df_topics):.2%}")
    if "raw_topic" in df_topics.columns:
        raw_outlier = int((df_topics["raw_topic"] == -1).sum())
        reassigned = int(df_topics["outlier_reassigned"].sum())
        lines.append(f"- 原始 HDBSCAN 离群论文：{raw_outlier:,} / {len(df_topics):,} = {raw_outlier / len(df_topics):.2%}")
        lines.append(f"- 二次归并论文：{reassigned:,}")
    lines.append("")

    lines.append("## 年份覆盖")
    lines.append("")
    year_counts = df_topics.groupby("year").size().reset_index(name="count").sort_values("year")
    lines.append(year_counts.to_markdown(index=False))
    lines.append("")

    lines.append("## 会议覆盖")
    lines.append("")
    venue_counts = df_topics.groupby("venue").size().reset_index(name="count").sort_values("count", ascending=False)
    lines.append(venue_counts.to_markdown(index=False))
    lines.append("")

    lines.append("## 全局 Top 50 主题")
    lines.append("")
    top_topics = topic_summary[topic_summary["topic"] != -1].sort_values("count", ascending=False).head(50).copy()
    top_topics["share"] = top_topics["share"].map(lambda value: f"{value:.2%}")
    lines.append(top_topics[["topic", "count", "share", "keywords", "top_venues"]].to_markdown(index=False))
    lines.append("")

    lines.append("## Accepted-only Top 30 主题")
    lines.append("")
    accepted = df_topics[df_topics["is_accepted"] & (df_topics["topic"] != -1)]
    accepted_summary = summarize_topic_subset(accepted).head(30).copy()
    if len(accepted_summary):
        accepted_summary["share"] = accepted_summary["share"].map(lambda value: f"{value:.2%}")
        lines.append(accepted_summary[["topic", "count", "share", "topic_keywords", "top_venues"]].to_markdown(index=False))
    else:
        lines.append("_No accepted papers matched._")
    lines.append("")

    lines.append("## 按年份 Top 12 主题")
    lines.append("")
    by_year = topic_by_group(df_topics[df_topics["topic"] != -1], ["year"])
    for year in sorted(df_topics["year"].dropna().unique()):
        sub = by_year[by_year["year"] == year].head(12).copy()
        sub["share"] = sub["share"].map(lambda value: f"{value:.2%}")
        lines.append(f"### {int(year)}")
        lines.append("")
        lines.append(sub[["topic", "count", "share", "topic_keywords"]].to_markdown(index=False))
        lines.append("")

    lines.append("## 数据口径说明")
    lines.append("")
    if scope == "all":
        lines.append("- 这里使用当前已爬取的全量数据：CV/NLP/ML 三大会 + AAAI + IJCAI 的公开论文元信息。")
        lines.append("- ECCV/ICCV/NAACL 因会议年份节奏不同，覆盖年份不是连续自然年。")
    else:
        lines.append("- 这里的“最近一轮”采用各会议最新可得届次。")
    lines.append("- ICLR accepted-only 口径包含 poster / spotlight / oral；submitted / withdrawn / desk reject 不算 accepted。")
    lines.append("- 主题名来自 BERTopic 的关键词，尚未做人手或 LLM 二次命名。")
    lines.append("- 聚类输入为 title + keywords + abstract，未使用 introduction 全文。")
    lines.append("")

    lines.append("## 主要输出文件")
    lines.append("")
    for filename in [
        "papers_with_topics.csv",
        "topic_summary.csv",
        "topic_by_year.csv",
        "topic_by_venue.csv",
        "topic_by_venue_year.csv",
        "topic_by_domain.csv",
        "topic_by_status.csv",
        "accepted_only_topic_summary.csv",
        "representative_papers_with_metadata.csv",
    ]:
        lines.append(f"- `{filename}`")

    (output_dir / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AI conference topics with BERTopic.")
    parser.add_argument("--input", default="data/recent_conferences/recent_ai_conference_papers_v7.jsonl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--scope", choices=["latest", "all"], default="latest")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--min-topic-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument(
        "--reduce-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reassign HDBSCAN outliers to the nearest topic when similarity clears the threshold.",
    )
    parser.add_argument("--outlier-threshold", type=float, default=0.65)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit after timeline sorting.")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"recent_cycle_topics_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records: {input_path}")
    records = load_jsonl(input_path)
    df = select_scope(records, args.scope)
    if args.limit:
        df = df.head(args.limit).copy()
    df["is_accepted"] = df["status"].map(is_accepted_status)
    print(f"Papers selected for scope={args.scope}: {len(df)}")
    df.to_json(output_dir / "analyzed_papers.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_csv(output_dir / "analyzed_papers.csv", index=False)

    docs = df["text_for_analysis"].tolist()
    embeddings_path = output_dir / "embeddings_all_minilm_l6_v2.npy"
    embeddings = encode_documents(
        docs=docs,
        embeddings_path=embeddings_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )

    topic_model = build_topic_model(
        min_topic_size=args.min_topic_size,
        min_samples=args.min_samples,
        n_neighbors=args.n_neighbors,
    )
    print("Fitting BERTopic...")
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    print("BERTopic fit complete.")
    raw_topics = [int(topic) for topic in topics]
    print(f"Raw outliers: {sum(topic == -1 for topic in raw_topics):,} / {len(raw_topics):,}")

    if args.reduce_outliers:
        print(f"Reducing outliers with embedding similarity threshold={args.outlier_threshold}...")
        final_topics, assignment_similarity = reduce_outliers_with_embeddings(
            topic_model=topic_model,
            topics=raw_topics,
            embeddings=embeddings,
            threshold=args.outlier_threshold,
        )
        print(f"Final outliers: {sum(topic == -1 for topic in final_topics):,} / {len(final_topics):,}")
    else:
        final_topics = raw_topics
        assignment_similarity = [None] * len(raw_topics)

    df_topics = add_topic_columns(df, topic_model, final_topics)
    df_topics["raw_topic"] = raw_topics
    df_topics["was_outlier"] = df_topics["raw_topic"] == -1
    df_topics["outlier_assignment_similarity"] = assignment_similarity
    df_topics["outlier_reassigned"] = df_topics["was_outlier"] & (df_topics["topic"] != -1)
    df_topics.to_csv(output_dir / "papers_with_topics.csv", index=False)
    df_topics.to_json(output_dir / "papers_with_topics.jsonl", orient="records", lines=True, force_ascii=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_dir / "bertopic_raw_topic_info.csv", index=False)

    topic_summary = summarize_topics(df_topics, topic_model)
    topic_summary.to_csv(output_dir / "topic_summary.csv", index=False)

    topic_by_group(df_topics, ["year"]).to_csv(output_dir / "topic_by_year.csv", index=False)
    topic_by_group(df_topics, ["venue"]).to_csv(output_dir / "topic_by_venue.csv", index=False)
    topic_by_group(df_topics, ["domain"]).to_csv(output_dir / "topic_by_domain.csv", index=False)
    topic_by_group(df_topics, ["venue", "year"]).to_csv(output_dir / "topic_by_venue_year.csv", index=False)
    topic_by_group(df_topics, ["timeline_order", "timeline_label"]).to_csv(
        output_dir / "topic_by_timeline.csv", index=False
    )
    topic_by_group(df_topics, ["status"]).to_csv(output_dir / "topic_by_status.csv", index=False)

    accepted_topic_summary = summarize_topic_subset(df_topics[df_topics["is_accepted"]])
    accepted_topic_summary.to_csv(output_dir / "accepted_only_topic_summary.csv", index=False)

    representative_rows = []
    representative_docs = topic_model.get_representative_docs()
    doc_lookup: dict[str, list[int]] = {}
    for idx, text in enumerate(df_topics["text_for_analysis"].tolist()):
        doc_lookup.setdefault(text, []).append(idx)
    for topic_id, docs_for_topic in representative_docs.items():
        for rank, doc in enumerate(docs_for_topic[:8], start=1):
            representative_rows.append({"topic": int(topic_id), "rank": rank, "representative_doc": doc})
    pd.DataFrame(representative_rows).to_csv(output_dir / "representative_docs.csv", index=False)

    representative_metadata_rows = []
    for row in representative_rows:
        matches = doc_lookup.get(row["representative_doc"], [])
        if not matches:
            continue
        paper = df_topics.iloc[matches[0]]
        representative_metadata_rows.append(
            {
                "topic": row["topic"],
                "rank": row["rank"],
                "topic_keywords": paper["topic_keywords"],
                "venue": paper.get("venue"),
                "year": paper.get("year"),
                "status": paper.get("status"),
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "html_url": paper.get("html_url"),
                "pdf_url": paper.get("pdf_url"),
                "openreview_url": paper.get("openreview_url"),
            }
        )
    pd.DataFrame(representative_metadata_rows).to_csv(
        output_dir / "representative_papers_with_metadata.csv", index=False
    )

    write_report(output_dir, df_topics, topic_summary, args.scope)
    write_chinese_report(
        output_dir,
        df_topics,
        topic_summary,
        args.scope,
        args.input,
        args.min_topic_size,
        args.min_samples,
        args.n_neighbors,
    )

    if args.save_model:
        model_path = output_dir / "bertopic_model"
        topic_model.save(str(model_path), serialization="safetensors")
        print(f"Model saved: {model_path}")

    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
