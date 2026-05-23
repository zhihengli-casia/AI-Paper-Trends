#!/usr/bin/env python3
"""Fine-grained venue-year topic analysis from cached paper embeddings.

The original repository pipeline uses BERTopic for a single conference.
This script is intentionally separate: it reuses cached paper metadata and
embeddings, clusters each venue-year independently, and writes local results
under results/ without changing README artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize


DEFAULT_INPUT_ROOT = Path("results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25")
DEFAULT_OUTPUT_ROOT = Path("results/fine_grained_venue_year_topics_umap_hdbscan")

GENERIC_TERMS = {
    "ability",
    "accurate",
    "achieve",
    "achieves",
    "across",
    "advance",
    "advanced",
    "analysis",
    "application",
    "applications",
    "approach",
    "approaches",
    "based",
    "benchmark",
    "benchmarks",
    "better",
    "challenge",
    "challenges",
    "current",
    "data",
    "dataset",
    "datasets",
    "different",
    "effective",
    "efficient",
    "empirical",
    "evaluation",
    "existing",
    "experiment",
    "experiments",
    "extensive",
    "framework",
    "high",
    "improve",
    "improved",
    "improves",
    "large",
    "learn",
    "method",
    "methods",
    "model",
    "models",
    "new",
    "novel",
    "paper",
    "performance",
    "problem",
    "propose",
    "proposed",
    "provide",
    "results",
    "show",
    "shows",
    "state",
    "study",
    "system",
    "systems",
    "task",
    "tasks",
    "training",
    "use",
    "used",
    "using",
    "via",
    "work",
}

LATEX_TERMS = {
    "alpha",
    "beta",
    "bf",
    "boldsymbol",
    "cal",
    "cdot",
    "emph",
    "frac",
    "gamma",
    "lambda",
    "mathbf",
    "math",
    "mathbb",
    "mathcal",
    "mathrm",
    "operatorname",
    "text",
    "tilde",
}

MACRO_KEYWORDS = {
    "LLM/语言模型": {
        "llm",
        "llms",
        "language model",
        "language models",
        "large language",
        "instruction",
        "prompt",
        "in-context",
        "token",
        "reasoning",
        "chain-of-thought",
        "cot",
        "alignment",
    },
    "多模态/VLM": {
        "vision language",
        "vision-language",
        "vlm",
        "vlms",
        "multimodal",
        "multi-modal",
        "image text",
        "visual question",
        "video language",
        "mllm",
        "mllms",
    },
    "生成模型": {
        "diffusion",
        "generative",
        "generation",
        "text-to-image",
        "image generation",
        "video generation",
        "score-based",
        "gan",
        "vae",
    },
    "计算机视觉": {
        "image",
        "object detection",
        "segmentation",
        "recognition",
        "tracking",
        "pose",
        "scene",
        "vision",
        "visual",
    },
    "3D/具身/机器人": {
        "3d",
        "robot",
        "robotics",
        "embodied",
        "navigation",
        "autonomous driving",
        "driving",
        "point cloud",
        "nerf",
        "slam",
    },
    "强化学习/决策": {
        "reinforcement learning",
        "rl",
        "policy",
        "reward",
        "agent",
        "agents",
        "planning",
        "control",
        "bandit",
    },
    "推荐/检索/排序": {
        "recommendation",
        "recommender",
        "retrieval",
        "ranking",
        "search",
        "query",
        "click",
        "user",
        "item",
    },
    "图学习/数据挖掘": {
        "graph",
        "graphs",
        "node",
        "knowledge graph",
        "mining",
        "clustering",
        "community",
    },
    "NLP任务": {
        "translation",
        "summarization",
        "question answering",
        "qa",
        "dialogue",
        "dialog",
        "named entity",
        "parsing",
        "semantic",
        "text classification",
    },
    "语音/音频/音乐": {
        "speech",
        "audio",
        "music",
        "sound",
        "acoustic",
        "speaker",
        "voice",
    },
    "可信/安全/公平": {
        "robust",
        "robustness",
        "adversarial",
        "privacy",
        "fairness",
        "bias",
        "safety",
        "secure",
        "trustworthy",
        "uncertainty",
    },
    "理论/优化": {
        "theory",
        "theoretical",
        "optimization",
        "convergence",
        "generalization",
        "gradient",
        "loss landscape",
        "sample complexity",
    },
    "AI4Science/医疗": {
        "protein",
        "molecule",
        "molecular",
        "drug",
        "biology",
        "medical",
        "clinical",
        "health",
        "science",
        "physics",
    },
    "系统/效率/压缩": {
        "efficient",
        "efficiency",
        "compression",
        "quantization",
        "pruning",
        "serving",
        "hardware",
        "memory",
        "latency",
    },
    "HCI/社会计算": {
        "human",
        "interaction",
        "interface",
        "user study",
        "social media",
        "crowdsourcing",
        "collaboration",
        "accessibility",
    },
}


@dataclass(frozen=True)
class GroupPath:
    venue: str
    year: int
    papers_path: Path
    embeddings_path: Path


@dataclass
class GroupRunSummary:
    venue: str
    year: int
    papers: int
    min_cluster_size: int
    n_neighbors: int
    n_components: int
    raw_topics: int
    raw_outliers: int
    reassigned_outliers: int
    final_outliers: int
    final_topics: int
    largest_topic: int
    median_topic_size: float
    seconds: float
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--venue", action="append", help="Venue to include. Can be passed multiple times.")
    parser.add_argument("--year", action="append", type=int, help="Year to include. Can be passed multiple times.")
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute groups with existing outputs.")
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--n-components", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--outlier-similarity-threshold", type=float, default=0.65)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def iter_groups(input_root: Path, venues: set[str] | None, years: set[int] | None) -> list[GroupPath]:
    groups: list[GroupPath] = []
    for papers_path in sorted(input_root.glob("venue=*/year=*/papers_with_topics.jsonl")):
        venue = papers_path.parent.parent.name.split("=", 1)[1]
        year = int(papers_path.parent.name.split("=", 1)[1])
        if venues and venue not in venues:
            continue
        if years and year not in years:
            continue
        embedding_candidates = sorted(papers_path.parent.glob("embeddings_*.npy"))
        if not embedding_candidates:
            print(f"[skip] {venue} {year}: missing embeddings", flush=True)
            continue
        groups.append(GroupPath(venue, year, papers_path, embedding_candidates[0]))
    return sorted(groups, key=lambda group: (group.venue, group.year))


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def choose_min_cluster_size(n_papers: int) -> int:
    if n_papers < 80:
        return max(4, n_papers // 12)
    if n_papers < 250:
        return 6
    if n_papers < 2000:
        return 8
    if n_papers < 4000:
        return 10
    return 15


def clean_text(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def paper_text(record: dict) -> str:
    if record.get("text_for_analysis"):
        return str(record["text_for_analysis"])
    keywords = record.get("keywords") or []
    if isinstance(keywords, list):
        keywords_text = " ".join(str(keyword) for keyword in keywords)
    else:
        keywords_text = str(keywords)
    return f"{record.get('title', '')}. {keywords_text}. {record.get('abstract', '')}".strip()


def valid_phrase(term: str) -> bool:
    tokens = term.split()
    if not tokens:
        return False
    if len(term) < 3:
        return False
    if any(token in LATEX_TERMS for token in tokens):
        return False
    if all(token in GENERIC_TERMS for token in tokens):
        return False
    if len(tokens) == 1 and tokens[0] in GENERIC_TERMS:
        return False
    if sum(token.isdigit() for token in tokens) > 1:
        return False
    return True


def diverse_terms(scored_terms: list[tuple[str, float]], limit: int = 10) -> list[str]:
    selected: list[str] = []
    selected_tokens: set[str] = set()
    for term, _score in scored_terms:
        if not valid_phrase(term):
            continue
        tokens = set(term.split())
        if selected and tokens <= selected_tokens:
            continue
        if any(term in chosen or chosen in term for chosen in selected):
            continue
        selected.append(term)
        selected_tokens.update(tokens)
        if len(selected) >= limit:
            break
    return selected


def build_topic_terms(records: list[dict], labels: np.ndarray, topic_ids: list[int]) -> dict[int, list[str]]:
    docs_by_topic: dict[int, list[str]] = defaultdict(list)
    for record, label in zip(records, labels):
        if label == -1:
            continue
        docs_by_topic[int(label)].append(clean_text(paper_text(record)))

    topic_docs = [" ".join(docs_by_topic[topic_id]) for topic_id in topic_ids]
    min_df = 2 if len(topic_docs) >= 10 else 1
    stop_words = set(ENGLISH_STOP_WORDS) | GENERIC_TERMS | LATEX_TERMS
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        min_df=min_df,
        max_df=0.70,
        max_features=80000,
        stop_words=list(stop_words),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+\-]{1,}\b",
    )
    try:
        counts = vectorizer.fit_transform(topic_docs).astype(float)
    except ValueError:
        return {topic_id: [] for topic_id in topic_ids}

    term_names = np.array(vectorizer.get_feature_names_out())
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    tf = counts.multiply(1 / row_sums[:, None])
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((1 + len(topic_docs)) / (1 + df)) + 1
    scores = tf.multiply(idf)

    topic_terms: dict[int, list[str]] = {}
    for row_idx, topic_id in enumerate(topic_ids):
        row = scores.getrow(row_idx)
        if row.nnz == 0:
            topic_terms[topic_id] = []
            continue
        order = row.data.argsort()[::-1]
        scored = [(str(term_names[row.indices[i]]), float(row.data[i])) for i in order[:80]]
        topic_terms[topic_id] = diverse_terms(scored, limit=12)
    return topic_terms


def assign_macro_topic(terms: list[str], representative_titles: list[str]) -> str:
    text = " ".join(terms + representative_titles).lower()
    scores: dict[str, int] = {}
    for macro, keywords in MACRO_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword_in_text(keyword, text):
                score += 2 if " " in keyword or "-" in keyword else 1
        if score:
            scores[macro] = score
    if scores.get("多模态/VLM") and any(
        marker in text for marker in ("mllm", "mllms", "multimodal", "multi-modal", "vision-language")
    ):
        scores["多模态/VLM"] += 3
    if scores.get("3D/具身/机器人") and any(
        marker in text for marker in ("embodied", "robot", "robotics", "autonomous driving", "point cloud")
    ):
        scores["3D/具身/机器人"] += 2
    if not scores:
        return "其他/交叉主题"
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def keyword_in_text(keyword: str, text: str) -> bool:
    if " " in keyword or "-" in keyword:
        return keyword in text
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None


def representative_titles_for_topic(
    records: list[dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    topic_id: int,
    centroid: np.ndarray,
    limit: int = 5,
) -> list[str]:
    indices = np.flatnonzero(labels == topic_id)
    if len(indices) == 0:
        return []
    sims = embeddings[indices] @ centroid
    best_indices = indices[np.argsort(sims)[::-1][:limit]]
    return [str(records[index].get("title", "")).strip() for index in best_indices]


def remap_labels_by_size(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    counts = Counter(int(label) for label in labels if label != -1)
    old_topic_ids = [topic_id for topic_id, _count in counts.most_common()]
    mapping = {old_topic_id: new_topic_id for new_topic_id, old_topic_id in enumerate(old_topic_ids)}
    remapped = np.array([mapping.get(int(label), -1) for label in labels], dtype=int)
    return remapped, mapping


def cluster_group(group: GroupPath, output_root: Path, args: argparse.Namespace) -> GroupRunSummary | None:
    group_start = time.time()
    out_dir = output_root / f"venue={group.venue}" / f"year={group.year}"
    papers_output = out_dir / "papers_with_fine_topics.jsonl"
    topic_output = out_dir / "topic_summary.csv"
    if papers_output.exists() and topic_output.exists() and not args.force:
        print(f"[skip] {group.venue} {group.year}: outputs already exist", flush=True)
        return None

    records = read_jsonl(group.papers_path)
    embeddings = np.load(group.embeddings_path)
    if len(records) != len(embeddings):
        raise ValueError(
            f"{group.venue} {group.year} has {len(records)} records but {len(embeddings)} embeddings"
        )
    n_papers = len(records)
    if n_papers < 5:
        print(f"[skip] {group.venue} {group.year}: too few papers", flush=True)
        return None

    import hdbscan
    import umap

    min_cluster_size = choose_min_cluster_size(n_papers)
    n_neighbors = min(args.n_neighbors, max(2, n_papers - 1))
    n_components = min(args.n_components, max(2, n_papers - 2))
    normalized_embeddings = normalize(embeddings.astype("float32"))

    print(
        f"[run] {group.venue} {group.year}: n={n_papers}, "
        f"mcs={min_cluster_size}, nn={n_neighbors}, dim={n_components}",
        flush=True,
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        low_memory=True,
        random_state=args.random_state,
    )
    reduced = reducer.fit_transform(normalized_embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=False,
        core_dist_n_jobs=-1,
    )
    raw_labels = clusterer.fit_predict(reduced).astype(int)
    raw_counts = Counter(int(label) for label in raw_labels)
    raw_topics = len([label for label in raw_counts if label != -1])
    raw_outliers = raw_counts.get(-1, 0)

    final_labels = raw_labels.copy()
    assignment_method = np.array(["hdbscan"] * n_papers, dtype=object)
    assignment_similarity = np.full(n_papers, np.nan, dtype=float)

    topic_ids = sorted(label for label in raw_counts if label != -1)
    centroid_by_topic: dict[int, np.ndarray] = {}
    if topic_ids:
        centroids = []
        for topic_id in topic_ids:
            centroid = normalized_embeddings[raw_labels == topic_id].mean(axis=0)
            centroid = normalize(centroid.reshape(1, -1))[0]
            centroid_by_topic[int(topic_id)] = centroid
            centroids.append(centroid)
        centroid_matrix = np.vstack(centroids)

        outlier_indices = np.flatnonzero(raw_labels == -1)
        if len(outlier_indices):
            similarities = normalized_embeddings[outlier_indices] @ centroid_matrix.T
            best_positions = similarities.argmax(axis=1)
            best_scores = similarities[np.arange(len(outlier_indices)), best_positions]
            best_topic_ids = np.array(topic_ids, dtype=int)[best_positions]
            should_reassign = best_scores >= args.outlier_similarity_threshold
            final_labels[outlier_indices[should_reassign]] = best_topic_ids[should_reassign]
            assignment_method[outlier_indices[should_reassign]] = "nearest_centroid"
            assignment_similarity[outlier_indices] = best_scores

    final_labels, remap = remap_labels_by_size(final_labels)
    raw_to_new = {old: new for old, new in remap.items()}
    remapped_raw_labels = np.array([raw_to_new.get(int(label), -1) for label in raw_labels], dtype=int)

    final_counts = Counter(int(label) for label in final_labels)
    final_topic_ids = [topic_id for topic_id in sorted(final_counts) if topic_id != -1]
    final_centroids: dict[int, np.ndarray] = {}
    for topic_id in final_topic_ids:
        centroid = normalized_embeddings[final_labels == topic_id].mean(axis=0)
        final_centroids[topic_id] = normalize(centroid.reshape(1, -1))[0]

    topic_terms = build_topic_terms(records, final_labels, final_topic_ids)

    topic_rows = []
    for topic_id in final_topic_ids:
        titles = representative_titles_for_topic(
            records,
            normalized_embeddings,
            final_labels,
            topic_id,
            final_centroids[topic_id],
            limit=5,
        )
        terms = topic_terms.get(topic_id, [])
        label = " / ".join(terms[:4]) if terms else f"Topic {topic_id}"
        topic_rows.append(
            {
                "venue": group.venue,
                "year": group.year,
                "topic_id": topic_id,
                "topic_label": label,
                "macro_topic": assign_macro_topic(terms, titles),
                "paper_count": final_counts[topic_id],
                "paper_share": final_counts[topic_id] / n_papers,
                "keywords": "; ".join(terms[:12]),
                "representative_titles": " || ".join(titles),
            }
        )

    enriched_records = []
    for index, record in enumerate(records):
        topic_id = int(final_labels[index])
        raw_topic_id = int(remapped_raw_labels[index])
        row = dict(record)
        row["fine_topic"] = topic_id
        row["fine_raw_topic"] = raw_topic_id
        row["fine_topic_label"] = next(
            (topic["topic_label"] for topic in topic_rows if topic["topic_id"] == topic_id),
            "Outlier",
        )
        row["fine_macro_topic"] = next(
            (topic["macro_topic"] for topic in topic_rows if topic["topic_id"] == topic_id),
            "Outlier",
        )
        row["fine_topic_keywords"] = next(
            (topic["keywords"] for topic in topic_rows if topic["topic_id"] == topic_id),
            "",
        )
        row["fine_was_hdbscan_outlier"] = bool(raw_labels[index] == -1)
        row["fine_topic_assignment_method"] = str(assignment_method[index])
        row["fine_outlier_assignment_similarity"] = (
            None if math.isnan(float(assignment_similarity[index])) else float(assignment_similarity[index])
        )
        row["fine_cluster_unit"] = "venue_year"
        row["fine_cluster_algorithm"] = "umap_hdbscan_leaf_centroid_reassignment"
        enriched_records.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(papers_output, enriched_records)
    pd.DataFrame(topic_rows).sort_values("paper_count", ascending=False).to_csv(
        topic_output, index=False, encoding="utf-8-sig"
    )
    with (out_dir / "topic_terms.json").open("w", encoding="utf-8") as file:
        json.dump({str(k): v for k, v in topic_terms.items()}, file, ensure_ascii=False, indent=2)

    final_outliers = final_counts.get(-1, 0)
    topic_sizes = [final_counts[topic_id] for topic_id in final_topic_ids]
    summary = GroupRunSummary(
        venue=group.venue,
        year=group.year,
        papers=n_papers,
        min_cluster_size=min_cluster_size,
        n_neighbors=n_neighbors,
        n_components=n_components,
        raw_topics=raw_topics,
        raw_outliers=raw_outliers,
        reassigned_outliers=raw_outliers - final_outliers,
        final_outliers=final_outliers,
        final_topics=len(final_topic_ids),
        largest_topic=max(topic_sizes) if topic_sizes else 0,
        median_topic_size=float(np.median(topic_sizes)) if topic_sizes else 0.0,
        seconds=time.time() - group_start,
        output_dir=str(out_dir),
    )

    print(
        f"[done] {group.venue} {group.year}: topics={summary.final_topics}, "
        f"raw_outliers={raw_outliers}, final_outliers={final_outliers}, "
        f"largest={summary.largest_topic}, median={summary.median_topic_size:.1f}, "
        f"{summary.seconds:.1f}s",
        flush=True,
    )
    return summary


def save_global_outputs(output_root: Path, summaries: list[GroupRunSummary]) -> None:
    if not summaries:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([asdict(summary) for summary in summaries])
    summary_df.sort_values(["venue", "year"]).to_csv(output_root / "run_summary.csv", index=False)

    topic_frames = []
    for summary in summaries:
        topic_path = Path(summary.output_dir) / "topic_summary.csv"
        if topic_path.exists():
            topic_frames.append(pd.read_csv(topic_path))
    if topic_frames:
        pd.concat(topic_frames, ignore_index=True).to_csv(
            output_root / "all_topic_summary.csv", index=False, encoding="utf-8-sig"
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": [asdict(summary) for summary in summaries],
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    venues = set(args.venue) if args.venue else None
    years = set(args.year) if args.year else None
    groups = iter_groups(args.input_root, venues, years)
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]
    if not groups:
        raise SystemExit(f"No groups found under {args.input_root}")

    print(f"Input root: {args.input_root}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print(f"Groups: {len(groups)}", flush=True)

    summaries: list[GroupRunSummary] = []
    for group in groups:
        summary = cluster_group(group, args.output_root, args)
        if summary is not None:
            summaries.append(summary)
            save_global_outputs(args.output_root, summaries)
    save_global_outputs(args.output_root, summaries)
    print(f"Finished {len(summaries)} computed groups.", flush=True)


if __name__ == "__main__":
    main()
