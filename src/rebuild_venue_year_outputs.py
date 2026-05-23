"""Rebuild aggregate venue-year topic outputs from completed group folders.

Use this after adding a small number of missing conference-year runs or after
updating deterministic topic naming rules. It does not rerun embeddings or
BERTopic; it only re-labels summaries and rewrites aggregate CSV/Markdown
files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from refine_topic_names import specific_label_cn, split_keywords


DETAIL_RULES = [
    (("rag", "retrieval-augmented", "retrieval augmented"), "RAG问答"),
    (("retrieval", "search", "ranking", "query", "document"), "搜索排序"),
    (("recommendation", "recommender", "user", "item"), "推荐排序"),
    (("agent", "agents", "planning", "workflow", "tool"), "Agent规划"),
    (("dialogue", "conversation", "user"), "对话交互"),
    (("reasoning", "cot", "chain-of-thought", "question", "answering"), "推理问答"),
    (("bias", "social", "cultural", "gender", "misinformation"), "社会安全"),
    (("safety", "attack", "attacks", "adversarial", "privacy", "jailbreak"), "安全攻防"),
    (("lora", "adapter", "low-rank", "federated", "client"), "LoRA联邦微调"),
    (("multimodal", "vision-language", "vlm", "vlms", "mllm", "mllms", "clip"), "视觉语言"),
    (("cache", "token", "tokens", "attention", "inference", "quantization"), "推理效率"),
    (("video", "videos", "temporal", "action", "motion"), "视频时序"),
    (("audio", "speech", "music", "speaker"), "语音音频"),
    (("segmentation", "semantic", "mask", "panoptic"), "分割"),
    (("detection", "detector", "object"), "检测识别"),
    (("depth", "point", "camera", "pose", "lidar", "geometric"), "三维几何"),
    (("gaussian", "splatting", "rendering", "reconstruction", "novel view"), "三维重建"),
    (("restoration", "superresolution", "denoising", "enhancement"), "图像复原"),
    (("medical", "clinical", "pathology", "patient"), "医学影像"),
    (("domain", "generalization", "ood", "incremental", "few-shot"), "迁移泛化"),
    (("time series", "forecasting", "dynamics", "pde"), "时序动力学"),
    (("diffusion", "generation", "generative", "editing"), "生成编辑"),
    (("graph", "graphs", "gnn", "node", "clustering"), "图学习"),
    (("optimization", "gradient", "convergence", "convex"), "优化理论"),
    (("bandit", "regret", "policy", "reward", "reinforcement"), "决策强化"),
    (("code", "program", "software"), "代码智能"),
    (("human", "linguistic", "reading", "semantic"), "认知语言"),
    (("evaluation", "metrics", "benchmark", "legal"), "评测指标"),
]


def detail_suffix(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("topic_keywords", "") or "").lower(),
            str(row.get("title_terms", "") or "").lower(),
            str(row.get("representative_titles", "") or "").lower(),
        ]
    )
    for needles, suffix in DETAIL_RULES:
        if any(needle in text for needle in needles):
            return suffix

    keywords = [kw for kw in split_keywords(row.get("topic_keywords", "")) if kw]
    if not keywords:
        return f"T{int(row.get('topic', 0)):03d}"
    return "/".join(keyword[:16] for keyword in keywords[:2])


def disambiguate_topic_labels(topic_summary: pd.DataFrame) -> pd.DataFrame:
    """Make duplicate labels unique within each venue-year using topic evidence."""

    out = topic_summary.copy()
    duplicated = out.duplicated(["venue", "year", "specific_label_cn"], keep=False)
    if not duplicated.any():
        return out

    for (_, _, _), indexes in out[duplicated].groupby(["venue", "year", "specific_label_cn"]).groups.items():
        used: dict[str, int] = {}
        for idx in indexes:
            suffix = detail_suffix(out.loc[idx])
            used[suffix] = used.get(suffix, 0) + 1
            if used[suffix] > 1:
                suffix = f"{suffix}-{used[suffix]}"
            out.loc[idx, "specific_label_cn"] = f"{out.loc[idx, 'specific_label_cn']}（{suffix}）"
            out.loc[idx, "specific_label_en"] = f"{out.loc[idx, 'specific_label_en']} / {suffix}"
    return out


def relabel_topic_summary(topic_summary: pd.DataFrame) -> pd.DataFrame:
    out = topic_summary.copy()
    labels = out.apply(
        lambda row: specific_label_cn(
            row.get("topic_keywords", ""),
            str(row.get("representative_titles", "")).split(" || ")
            if pd.notna(row.get("representative_titles", ""))
            else [],
        ),
        axis=1,
    )
    out["specific_label_cn"] = [item[0] for item in labels]
    out["specific_parent_category"] = [item[1] for item in labels]
    out["specific_label_en"] = [item[2] for item in labels]
    out["naming_evidence"] = out.apply(
        lambda row: f"keywords={row.get('topic_keywords', '')}; title_terms={row.get('title_terms', '')}",
        axis=1,
    )
    return disambiguate_topic_labels(out)


def group_dirs(results_dir: Path) -> list[Path]:
    dirs = [path for path in results_dir.glob("venue=*/year=*") if path.is_dir()]
    return sorted(dirs, key=lambda path: (path.parent.name, path.name))


def read_group_outputs(results_dir: Path, update_group_summaries: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paper_frames: list[pd.DataFrame] = []
    topic_frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []

    for group_dir in group_dirs(results_dir):
        papers_path = group_dir / "papers_with_topics.csv"
        topics_path = group_dir / "topic_summary.csv"
        report_path = group_dir / "run_report.json"
        if not papers_path.exists() or not topics_path.exists() or not report_path.exists():
            print(f"Skipping incomplete group: {group_dir}")
            continue

        topics = relabel_topic_summary(pd.read_csv(topics_path))
        if update_group_summaries:
            topics.to_csv(topics_path, index=False)
        topic_frames.append(topics)
        paper_frames.append(pd.read_csv(papers_path))
        with report_path.open(encoding="utf-8") as f:
            reports.append(json.load(f))

    papers = pd.concat(paper_frames, ignore_index=True) if paper_frames else pd.DataFrame()
    topics = pd.concat(topic_frames, ignore_index=True) if topic_frames else pd.DataFrame()
    report_df = pd.DataFrame(reports)
    if len(report_df):
        report_df.sort_values(["year", "venue"], inplace=True)
    if len(topics):
        topics.sort_values(["year", "venue", "count"], ascending=[True, True, False], inplace=True)
    if len(papers):
        papers.sort_values(["year", "venue", "title"], inplace=True)
    return papers, topics, report_df


def write_outputs(
    results_dir: Path,
    papers: pd.DataFrame,
    topics: pd.DataFrame,
    report_df: pd.DataFrame,
    top_k: int,
    write_jsonl: bool,
) -> None:
    papers.to_csv(results_dir / "papers_with_venue_year_topics.csv", index=False)
    if write_jsonl:
        papers.to_json(
            results_dir / "papers_with_venue_year_topics.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
    topics.to_csv(results_dir / "topic_summary_by_venue_year.csv", index=False)
    report_df.to_csv(results_dir / "run_summary_by_venue_year.csv", index=False)

    top_rows = []
    for (venue, year), group in topics.groupby(["venue", "year"], sort=True):
        for rank, (_, row) in enumerate(group.sort_values("count", ascending=False).head(top_k).iterrows(), 1):
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
                    "short_keywords": ", ".join(split_keywords(row.get("topic_keywords", ""))[:5]),
                    "representative_titles": row.get("representative_titles", ""),
                }
            )
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(results_dir / f"top{top_k}_topics_by_venue_year.csv", index=False)

    label_trend = (
        topics.groupby(["venue", "year", "specific_parent_category", "specific_label_cn"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    totals = papers.groupby(["venue", "year"], dropna=False).size().reset_index(name="venue_year_total")
    label_trend = label_trend.merge(totals, on=["venue", "year"], how="left")
    label_trend["share"] = label_trend["count"] / label_trend["venue_year_total"]
    label_trend.sort_values(["venue", "year", "count"], ascending=[True, True, False]).to_csv(
        results_dir / "label_trend_by_venue_year.csv",
        index=False,
    )

    lines = [
        "# Venue-Year Main Accepted Topic Analysis",
        "",
        f"- Rebuilt at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Papers analyzed: {len(papers):,}",
        "- Clustering unit: independent conference-year kNN graph + Louvain runs",
        "- Topic names: deterministic rule-based refinement from keywords and representative titles",
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
    lines.extend(["", f"## Top {top_k} Topics By Venue-Year", ""])
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
    (results_dir / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild aggregate venue-year topic outputs.")
    parser.add_argument("--results-dir", default="results/venue_year_main_accepted_topics_2020_2025_bge_m25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--write-jsonl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update-group-summaries", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    papers, topics, report_df = read_group_outputs(results_dir, args.update_group_summaries)
    write_outputs(results_dir, papers, topics, report_df, args.top_k, args.write_jsonl)
    print(
        f"Rebuilt outputs under {results_dir}: "
        f"groups={len(report_df):,}, papers={len(papers):,}, topics={len(topics):,}"
    )


if __name__ == "__main__":
    main()
