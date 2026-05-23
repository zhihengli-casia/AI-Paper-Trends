"""Year-by-year topic clustering for the main accepted-only paper view.

Compared with the earlier all-years BERTopic run, this script:
- clusters each year independently;
- defaults to a stronger BGE embedding model;
- keeps top-k topic candidates for multi-label analysis;
- uses high-confidence outlier reassignment plus optional outlier re-clustering;
- writes deterministic Chinese topic labels and label-level yearly trends.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)


normalize_proxy_env()

from bertopic import BERTopic  # noqa: E402
from hdbscan import HDBSCAN  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from umap import UMAP  # noqa: E402


DEFAULT_INPUT = "data/recent_conferences/main_accepted_ai_conference_papers_v1.jsonl"
DEFAULT_OUTPUT_DIR = "results/yearly_main_accepted_topics_bge_m30"

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

TOPIC_LABEL_RULES = [
    (("rag", "retrievalaugmented", "dense retrieval", "retrieval", "reranking"), "LLM / IR", "检索增强生成与信息检索"),
    (("agent", "agents", "multiagent", "workflow"), "LLM Agents", "智能体与多智能体系统"),
    (("reasoning", "chainofthought", "cot", "mathematical"), "LLM Reasoning", "大模型推理与思维链"),
    (("pretraining", "instruction", "finetuning", "alignment", "sft"), "LLM Training", "大模型训练与指令调优"),
    (("preference", "rlhf", "dpo", "reward"), "Alignment", "偏好优化与人类反馈对齐"),
    (("hallucination", "factual", "truthfulness"), "Trustworthy AI", "幻觉检测与事实性"),
    (("multimodal", "visionlanguage", "mllm", "vqa", "vlm"), "Multimodal AI", "多模态理解与视觉语言模型"),
    (("texttoimage", "image generation", "diffusion", "flow matching", "score distillation"), "Generative AI", "图像生成与扩散模型"),
    (("video generation", "texttovideo", "video diffusion"), "Generative AI", "视频生成与编辑"),
    (("textto3d", "nerf", "gaussian", "splatting", "view synthesis"), "3D Vision", "三维生成与新视角合成"),
    (("segmentation", "object detection", "detector", "openvocabulary"), "Computer Vision", "目标检测与图像分割"),
    (("point cloud", "lidar", "autonomous driving", "trajectory"), "Embodied / Autonomous", "点云、自动驾驶与轨迹预测"),
    (("medical", "clinical", "patient", "healthcare", "biomedical"), "AI for Health", "医疗健康与生物医学 AI"),
    (("molecular", "protein", "drug", "chemical", "biology"), "AI for Science", "AI for Science 与药物/蛋白"),
    (("pde", "differential equations", "physicsinformed", "operator"), "AI for Science", "科学计算与物理方程建模"),
    (("gnn", "gnns", "graph", "node", "link prediction"), "Graph Learning", "图学习与图神经网络"),
    (("recommendation", "recommender", "ctr", "collaborative filtering"), "Recommendation", "推荐系统与点击率预测"),
    (("time series", "forecasting", "timeseries"), "Time Series", "时间序列预测"),
    (("causal", "treatment", "causal discovery"), "Causal ML", "因果推断与因果发现"),
    (("federated", "privacy", "attack", "adversarial", "watermark", "backdoor"), "Security / Privacy", "隐私安全、攻击与鲁棒性"),
    (("optimization", "gradient descent", "sgd", "adam", "convex"), "Optimization", "优化算法与训练动力学"),
    (("bandit", "bandits", "regret"), "Reinforcement Learning", "老虎机与在线决策"),
    (("offline reinforcement", "policy", "mdp", "reward"), "Reinforcement Learning", "强化学习与离线策略优化"),
    (("translation", "multilingual", "crosslingual", "lowresource"), "NLP", "机器翻译与多语言 NLP"),
    (("dialogue", "conversation", "conversational"), "NLP", "对话系统"),
    (("summarization", "summary"), "NLP", "文本摘要"),
    (("code generation", "software", "programming"), "Code Intelligence", "代码生成与软件工程"),
    (("participants", "hci", "user", "users", "virtual reality"), "HCI", "人机交互与用户研究"),
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
    return clean_text(f"{record.get('title', '')}. {keywords_text}. {record.get('abstract', '')}")


def infer_doc_prefix(model_name: str, explicit_prefix: str | None) -> str:
    if explicit_prefix is not None:
        return explicit_prefix
    if "e5-" in model_name.lower() or "/e5" in model_name.lower():
        return "passage: "
    return ""


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", model_name)


def encode_documents(
    docs: list[str],
    embeddings_path: Path,
    model_name: str,
    batch_size: int,
    device: str | None,
    doc_prefix: str,
) -> np.ndarray:
    if embeddings_path.exists():
        print(f"Using cached embeddings: {embeddings_path}")
        return np.load(embeddings_path)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    prefixed_docs = [f"{doc_prefix}{doc}" for doc in docs] if doc_prefix else docs
    print(f"Encoding {len(docs):,} documents with batch_size={batch_size}, device={model.device}")
    embeddings = model.encode(
        prefixed_docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    return embeddings


def build_topic_model(min_topic_size: int, min_samples: int, n_neighbors: int) -> BERTopic:
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.85,
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


def topic_keywords(topic_model: BERTopic, topic_id: int, top_n: int = 10) -> str:
    if topic_id == -1:
        return "outlier"
    words = topic_model.get_topic(topic_id) or []
    return ", ".join(word for word, _ in words[:top_n])


def label_for_keywords(keywords: str) -> tuple[str, str, str]:
    lower = keywords.lower()
    for needles, parent, label_cn in TOPIC_LABEL_RULES:
        if any(needle in lower for needle in needles):
            label_en = " / ".join(keywords.split(", ")[:3])
            return parent, label_cn, label_en
    top_terms = [term for term in keywords.split(", ")[:3] if term]
    label_en = " / ".join(top_terms) if top_terms else "misc"
    return "Other", f"其他主题：{' / '.join(top_terms)}", label_en


def normalize_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = record.copy()
        row["text_for_analysis"] = text_for_analysis(record)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df["text_for_analysis"].str.len() > 40].copy()
    df.sort_values(["year", "venue", "title"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def topic_embedding_matrix(topic_model: BERTopic) -> tuple[np.ndarray, list[int]]:
    matrix = topic_model.topic_embeddings_[topic_model._outliers :]
    topic_ids = list(range(matrix.shape[0]))
    return matrix, topic_ids


def reassign_outliers_to_nearest_topic(
    topic_model: BERTopic,
    topics: np.ndarray,
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[float | None], list[str]]:
    final_topics = topics.copy()
    similarities: list[float | None] = [None] * len(topics)
    methods = ["hdbscan"] * len(topics)
    outlier_ids = np.flatnonzero(topics == -1)
    if len(outlier_ids) == 0:
        return final_topics, similarities, methods

    topic_embeddings, topic_ids = topic_embedding_matrix(topic_model)
    if len(topic_ids) == 0:
        return final_topics, similarities, methods

    similarity = cosine_similarity(embeddings[outlier_ids], topic_embeddings)
    best_positions = similarity.argmax(axis=1)
    best_scores = similarity.max(axis=1)
    for row_id, position, score in zip(outlier_ids, best_positions, best_scores, strict=True):
        similarities[int(row_id)] = float(score)
        if score >= threshold:
            final_topics[int(row_id)] = int(topic_ids[int(position)])
            methods[int(row_id)] = "nearest_topic_similarity"
        else:
            methods[int(row_id)] = "outlier_low_similarity"
    return final_topics, similarities, methods


def recluster_remaining_outliers(
    docs: list[str],
    embeddings: np.ndarray,
    final_topics: np.ndarray,
    methods: list[str],
    keyword_map: dict[int, str],
    min_topic_size: int,
    min_samples: int,
    n_neighbors: int,
) -> tuple[np.ndarray, list[str], dict[int, str]]:
    remaining = np.flatnonzero(final_topics == -1)
    if len(remaining) < max(min_topic_size * 2, 50):
        return final_topics, methods, keyword_map

    secondary_neighbors = min(n_neighbors, max(2, len(remaining) - 1))
    secondary_model = build_topic_model(
        min_topic_size=min_topic_size,
        min_samples=min_samples,
        n_neighbors=secondary_neighbors,
    )
    print(f"Re-clustering {len(remaining):,} remaining outliers with min_topic_size={min_topic_size}")
    secondary_docs = [docs[int(i)] for i in remaining]
    secondary_embeddings = embeddings[remaining]
    secondary_topics, _ = secondary_model.fit_transform(secondary_docs, embeddings=secondary_embeddings)

    existing = [int(topic) for topic in final_topics if int(topic) >= 0]
    offset = max(existing) + 1 if existing else 0
    for local_id, secondary_topic in zip(remaining, secondary_topics, strict=True):
        if int(secondary_topic) == -1:
            methods[int(local_id)] = "outlier_after_recluster"
            continue
        new_topic = offset + int(secondary_topic)
        final_topics[int(local_id)] = new_topic
        methods[int(local_id)] = "outlier_reclustered"

    for secondary_topic in sorted(set(int(t) for t in secondary_topics if int(t) >= 0)):
        keyword_map[offset + secondary_topic] = topic_keywords(secondary_model, secondary_topic)

    return final_topics, methods, keyword_map


def compute_topic_centroids(topics: np.ndarray, embeddings: np.ndarray) -> tuple[np.ndarray, list[int]]:
    topic_ids = sorted(int(topic) for topic in set(topics) if int(topic) >= 0)
    centroids = []
    for topic_id in topic_ids:
        topic_embeddings = embeddings[topics == topic_id]
        centroid = topic_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        centroids.append(centroid)
    if not centroids:
        return np.empty((0, embeddings.shape[1])), []
    return np.vstack(centroids), topic_ids


def multilabel_candidates(
    topics: np.ndarray,
    embeddings: np.ndarray,
    top_k: int,
    threshold: float,
) -> tuple[list[str], list[str]]:
    centroids, topic_ids = compute_topic_centroids(topics, embeddings)
    if len(topic_ids) == 0:
        return [""] * len(topics), [""] * len(topics)
    similarity = cosine_similarity(embeddings, centroids)
    candidate_topics: list[str] = []
    candidate_scores: list[str] = []
    for row_idx, row in enumerate(similarity):
        order = row.argsort()[::-1][:top_k]
        pairs = [(topic_ids[int(i)], float(row[int(i)])) for i in order if row[int(i)] >= threshold]
        primary = int(topics[row_idx])
        if primary >= 0 and all(topic_id != primary for topic_id, _ in pairs):
            primary_pos = topic_ids.index(primary)
            pairs.append((primary, float(row[primary_pos])))
        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]
        candidate_topics.append(";".join(str(topic_id) for topic_id, _ in pairs))
        candidate_scores.append(";".join(f"{score:.4f}" for _, score in pairs))
    return candidate_topics, candidate_scores


def summarize_year_topics(df_year: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    total = len(df_year)
    for topic_id, count in Counter(df_year["topic"]).most_common():
        topic_df = df_year[df_year["topic"] == topic_id]
        venues = topic_df["venue"].value_counts().head(8)
        rows.append(
            {
                "year": int(year),
                "topic": int(topic_id),
                "year_topic_id": f"{int(year)}_T{int(topic_id):03d}" if int(topic_id) >= 0 else f"{int(year)}_T-001",
                "count": int(count),
                "share": count / total if total else 0,
                "topic_keywords": topic_df["topic_keywords"].iloc[0] if len(topic_df) else "",
                "parent_category": topic_df["parent_category"].iloc[0] if len(topic_df) else "",
                "topic_label_cn": topic_df["topic_label_cn"].iloc[0] if len(topic_df) else "",
                "topic_label_en": topic_df["topic_label_en"].iloc[0] if len(topic_df) else "",
                "top_venues": "; ".join(f"{venue}:{n}" for venue, n in venues.items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "count"], ascending=[True, False])


def analyze_one_year(
    df_year: pd.DataFrame,
    year: int,
    output_dir: Path,
    args: argparse.Namespace,
    doc_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    year_dir = output_dir / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    docs = df_year["text_for_analysis"].tolist()
    model_safe = safe_model_name(args.model_name)
    embeddings_path = year_dir / f"embeddings_{model_safe}.npy"
    embeddings = encode_documents(docs, embeddings_path, args.model_name, args.batch_size, args.device, doc_prefix)

    min_topic_size = min(args.min_topic_size, max(args.min_topic_size_floor, len(docs) // 8))
    n_neighbors = min(args.n_neighbors, max(2, len(docs) - 1))
    topic_model = build_topic_model(min_topic_size, args.min_samples, n_neighbors)
    print(f"Fitting BERTopic for {year}: papers={len(docs):,}, min_topic_size={min_topic_size}")
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

    topic_candidates, topic_candidate_scores = multilabel_candidates(
        final_topics,
        embeddings,
        args.multi_label_top_k,
        args.multi_label_threshold,
    )

    df_out = df_year.copy()
    df_out["topic"] = final_topics
    df_out["raw_topic"] = raw_topics_array
    df_out["was_outlier"] = df_out["raw_topic"] == -1
    df_out["outlier_assignment_similarity"] = outlier_similarity
    df_out["topic_assignment_method"] = assignment_methods
    df_out["outlier_reassigned"] = df_out["was_outlier"] & (df_out["topic"] != -1)
    df_out["topic_keywords"] = df_out["topic"].map(lambda topic: keyword_map.get(int(topic), "outlier"))
    labels = df_out["topic_keywords"].map(label_for_keywords)
    df_out["parent_category"] = labels.map(lambda item: item[0])
    df_out["topic_label_cn"] = labels.map(lambda item: item[1])
    df_out["topic_label_en"] = labels.map(lambda item: item[2])
    df_out["year_topic_id"] = df_out["topic"].map(
        lambda topic: f"{int(year)}_T{int(topic):03d}" if int(topic) >= 0 else f"{int(year)}_T-001"
    )
    df_out["candidate_topics"] = topic_candidates
    df_out["candidate_topic_scores"] = topic_candidate_scores

    topic_summary = summarize_year_topics(df_out, year)
    topic_summary.to_csv(year_dir / "topic_summary.csv", index=False)
    topic_model.get_topic_info().to_csv(year_dir / "bertopic_raw_topic_info.csv", index=False)
    df_out.to_csv(year_dir / "papers_with_topics.csv", index=False)
    df_out.to_json(year_dir / "papers_with_topics.jsonl", orient="records", lines=True, force_ascii=False)

    report = {
        "year": int(year),
        "papers": int(len(df_out)),
        "raw_outliers": int((df_out["raw_topic"] == -1).sum()),
        "final_outliers": int((df_out["topic"] == -1).sum()),
        "topics_excluding_outlier": int((topic_summary["topic"] != -1).sum()),
        "min_topic_size": int(min_topic_size),
        "model_name": args.model_name,
    }
    (year_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{year} done: raw_outliers={report['raw_outliers']:,}, "
        f"final_outliers={report['final_outliers']:,}, topics={report['topics_excluding_outlier']:,}"
    )
    return df_out, topic_summary, report


def write_outputs(output_dir: Path, all_papers: pd.DataFrame, all_topics: pd.DataFrame, reports: list[dict[str, Any]], args: argparse.Namespace) -> None:
    all_papers.to_csv(output_dir / "papers_with_yearly_topics.csv", index=False)
    all_papers.to_json(output_dir / "papers_with_yearly_topics.jsonl", orient="records", lines=True, force_ascii=False)
    all_topics.to_csv(output_dir / "topic_summary_by_year.csv", index=False)

    topic_by_venue = (
        all_papers.groupby(["year", "venue", "topic", "year_topic_id", "topic_keywords", "topic_label_cn"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = all_papers.groupby(["year", "venue"], dropna=False).size().reset_index(name="venue_year_total")
    topic_by_venue = topic_by_venue.merge(totals, on=["year", "venue"], how="left")
    topic_by_venue["share"] = topic_by_venue["count"] / topic_by_venue["venue_year_total"]
    topic_by_venue.sort_values(["year", "venue", "count"], ascending=[True, True, False]).to_csv(
        output_dir / "topic_by_venue_yearly.csv", index=False
    )

    label_trend = (
        all_papers[all_papers["topic"] != -1]
        .groupby(["year", "parent_category", "topic_label_cn"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    year_totals = all_papers.groupby("year").size().reset_index(name="year_total")
    label_trend = label_trend.merge(year_totals, on="year", how="left")
    label_trend["share"] = label_trend["count"] / label_trend["year_total"]
    label_trend.sort_values(["year", "count"], ascending=[True, False]).to_csv(output_dir / "label_trend_by_year.csv", index=False)

    report_lines = [
        "# Yearly Main Accepted Topic Analysis",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Input: `{args.input}`",
        f"- Papers analyzed: {len(all_papers):,}",
        f"- Embedding model: `{args.model_name}`",
        f"- Clustering unit: independent year-level BERTopic runs",
        f"- Multi-label: top_k={args.multi_label_top_k}, threshold={args.multi_label_threshold}",
        f"- Outlier nearest-topic threshold: {args.outlier_threshold}",
        "",
        "## Per-year run summary",
        "",
    ]
    report_df = pd.DataFrame(reports)
    report_df["raw_outlier_rate"] = report_df["raw_outliers"] / report_df["papers"]
    report_df["final_outlier_rate"] = report_df["final_outliers"] / report_df["papers"]
    report_df.to_csv(output_dir / "run_summary_by_year.csv", index=False)
    display_df = report_df[
        ["year", "papers", "topics_excluding_outlier", "raw_outliers", "raw_outlier_rate", "final_outliers", "final_outlier_rate"]
    ].copy()
    display_df["raw_outlier_rate"] = display_df["raw_outlier_rate"].map(lambda value: f"{value:.2%}")
    display_df["final_outlier_rate"] = display_df["final_outlier_rate"].map(lambda value: f"{value:.2%}")
    report_lines.append(display_df.to_markdown(index=False))
    report_lines.extend(["", "## Top topics by year", ""])
    for year in sorted(all_topics["year"].unique()):
        sub = all_topics[(all_topics["year"] == year) & (all_topics["topic"] != -1)].head(15).copy()
        sub["share"] = sub["share"].map(lambda value: f"{value:.2%}")
        report_lines.append(f"### {int(year)}")
        report_lines.append("")
        report_lines.append(sub[["year_topic_id", "count", "share", "topic_label_cn", "topic_keywords", "top_venues"]].to_markdown(index=False))
        report_lines.append("")
    (output_dir / "REPORT_CN.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run year-by-year main accepted topic clustering.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--doc-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--min-topic-size", type=int, default=30)
    parser.add_argument("--min-topic-size-floor", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--outlier-threshold", type=float, default=0.58)
    parser.add_argument("--recluster-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--secondary-min-topic-size", type=int, default=12)
    parser.add_argument("--secondary-min-samples", type=int, default=4)
    parser.add_argument("--multi-label-top-k", type=int, default=3)
    parser.add_argument("--multi-label-threshold", type=float, default=0.42)
    parser.add_argument("--years", nargs="*", type=int, default=[])
    parser.add_argument("--limit-per-year", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.input))
    df = normalize_rows(records)
    if args.years:
        df = df[df["year"].isin(args.years)].copy()
    if args.limit_per_year:
        df = df.groupby("year", group_keys=False).head(args.limit_per_year).copy()
    doc_prefix = infer_doc_prefix(args.model_name, args.doc_prefix)

    all_papers = []
    all_topic_summaries = []
    reports = []
    for year in sorted(df["year"].dropna().unique()):
        df_year = df[df["year"] == year].copy().reset_index(drop=True)
        year_papers, year_topics, report = analyze_one_year(df_year, int(year), output_dir, args, doc_prefix)
        all_papers.append(year_papers)
        all_topic_summaries.append(year_topics)
        reports.append(report)

    combined_papers = pd.concat(all_papers, ignore_index=True) if all_papers else pd.DataFrame()
    combined_topics = pd.concat(all_topic_summaries, ignore_index=True) if all_topic_summaries else pd.DataFrame()
    write_outputs(output_dir, combined_papers, combined_topics, reports, args)
    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
