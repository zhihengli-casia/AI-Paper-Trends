#!/usr/bin/env python3
"""Build a static topic atlas from fine-grained topic results."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote

import pandas as pd


DEFAULT_TOPIC_ROOT = Path("results/fine_grained_venue_year_topics_2020_2026_mcs_fine")
DEFAULT_OUTPUT_ROOT = Path("docs/topic-atlas")


CN_TOPIC_RULES: list[tuple[tuple[str, ...], str]] = [
    (("rlvr", "verifiable"), "可验证奖励驱动的大模型推理"),
    (("diffusion language",), "扩散语言模型与并行解码"),
    (("llm-as-a-judge",), "LLM-as-Judge 与自动评测"),
    (("mllms",), "多模态大模型与视觉语言推理"),
    (("mllm",), "多模态大模型与视觉语言推理"),
    (("vlms",), "视觉语言模型与多模态理解"),
    (("vlm",), "视觉语言模型与多模态理解"),
    (("vision-language",), "视觉语言模型与多模态理解"),
    (("rag", "retrieval"), "RAG 与检索增强生成"),
    (("chain-of-thought",), "Chain-of-Thought 与大模型推理"),
    (("cot", "reasoning"), "Chain-of-Thought 与大模型推理"),
    (("preference", "dpo"), "偏好优化、RLHF 与 DPO"),
    (("rlhf",), "人类反馈对齐与偏好优化"),
    (("lora",), "LoRA 与参数高效微调"),
    (("peft",), "参数高效微调与模型适配"),
    (("long-context",), "长上下文建模与压缩"),
    (("code generation",), "代码生成与程序理解"),
    (("programming",), "代码生成与程序理解"),
    (("multilingual",), "多语言建模与跨语言迁移"),
    (("translation",), "机器翻译与跨语言对齐"),
    (("summarization",), "文档摘要与信息压缩"),
    (("parsing",), "句法语义解析与结构化表示"),
    (("syntactic",), "句法知识与语言学分析"),
    (("clinical",), "医疗健康与临床 AI"),
    (("medical",), "医疗健康与临床 AI"),
    (("speech",), "语音/音频语言模型"),
    (("audio",), "语音/音频语言模型"),
    (("emotion", "multimodal"), "多模态情感理解"),
    (("eeg",), "脑电信号表征与解码"),
    (("graph", "gnn"), "图神经网络与图表示学习"),
    (("graph", "gnns"), "图神经网络与图表示学习"),
    (("graph", "node"), "图神经网络与节点表示学习"),
    (("knowledge graph",), "知识图谱推理与表示学习"),
    (("multi-view", "clustering"), "多视图聚类与图学习"),
    (("recommendation",), "推荐系统与用户建模"),
    (("recommender",), "推荐系统与用户建模"),
    (("ranking",), "搜索排序与相关性建模"),
    (("query",), "查询理解与检索优化"),
    (("generative retrieval",), "生成式检索"),
    (("vision-language-action",), "视觉语言动作模型与具身操作"),
    (("vla",), "视觉语言动作模型与具身操作"),
    (("text-to-image",), "文生图生成与个性化编辑"),
    (("t2i",), "文生图生成与个性化编辑"),
    (("video", "diffusion"), "视频扩散生成与运动控制"),
    (("motion", "video"), "视频动作生成与运动控制"),
    (("diffusion",), "扩散生成模型"),
    (("multimodal",), "多模态学习与跨模态理解"),
    (("gaussian", "splatting"), "3D Gaussian Splatting 与场景重建"),
    (("nerf",), "NeRF 与神经渲染"),
    (("radiance",), "NeRF 与神经渲染"),
    (("avatar",), "3D Avatar 与人脸头部建模"),
    (("depth",), "深度估计与立体匹配"),
    (("stereo",), "深度估计与立体匹配"),
    (("lidar",), "LiDAR 点云与 3D 感知"),
    (("point", "cloud"), "点云表示与 3D 感知"),
    (("autonomous",), "自动驾驶感知与世界模型"),
    (("driving",), "自动驾驶感知与世界模型"),
    (("robot",), "机器人操作与具身智能"),
    (("embodied",), "具身智能与物理交互"),
    (("segmentation",), "目标检测与图像分割"),
    (("object detection",), "目标检测与图像分割"),
    (("restoration",), "图像复原与超分辨率"),
    (("super-resolution",), "图像复原与超分辨率"),
    (("mamba",), "Mamba 与状态空间视觉模型"),
    (("ssm",), "状态空间模型与高效序列建模"),
    (("policy", "reward"), "强化学习策略与奖励建模"),
    (("reinforcement learning",), "强化学习算法与理论"),
    (("mdp",), "强化学习与 MDP 理论"),
    (("mdps",), "强化学习与 MDP 理论"),
    (("bandit",), "Bandit 与 regret 理论"),
    (("regret",), "在线学习与 regret 理论"),
    (("planning",), "规划搜索与决策推理"),
    (("agent", "gui"), "GUI 操作与计算机使用型 Agent"),
    (("multi-agent",), "多智能体协作与规划"),
    (("agents",), "LLM Agent 与工具使用"),
    (("sgd",), "随机优化与收敛理论"),
    (("convex",), "凸/非凸优化理论"),
    (("bilevel",), "双层优化与元学习"),
    (("ntk",), "神经网络理论、NTK 与宽度分析"),
    (("relu",), "神经网络理论与优化行为"),
    (("sample complexity",), "样本复杂度与统计学习理论"),
    (("conformal",), "Conformal Prediction 与不确定性校准"),
    (("adversarial",), "对抗攻击、鲁棒性与安全"),
    (("attack",), "攻击、防御与模型安全"),
    (("privacy",), "隐私保护与安全学习"),
    (("fairness",), "公平性、偏见与可信 AI"),
    (("federated",), "联邦学习与分布式训练"),
    (("time series",), "时间序列建模与预测"),
    (("forecasting",), "时间序列预测"),
    (("pde",), "PDE 神经求解器与科学计算"),
    (("molecular",), "分子表示学习与药物发现"),
    (("molecule",), "分子表示学习与药物发现"),
    (("protein",), "蛋白质建模与 AI4Science"),
    (("phishing",), "Web 安全与钓鱼网站分析"),
    (("blockchain",), "区块链生态与风险分析"),
    (("html",), "网页理解与代码生成"),
    (("social",), "社交媒体与社会计算"),
    (("news",), "新闻文本、虚假信息与安全检测"),
]

TERM_CN = {
    "llms": "大语言模型",
    "llm": "大语言模型",
    "reasoning": "推理",
    "retrieval": "检索",
    "rag": "RAG",
    "graph": "图学习",
    "gnns": "图神经网络",
    "node": "节点表示",
    "multimodal": "多模态",
    "visual": "视觉理解",
    "diffusion": "扩散模型",
    "policy": "策略优化",
    "reward": "奖励建模",
    "agents": "智能体",
    "agent": "智能体",
    "video": "视频理解/生成",
    "motion": "运动建模",
    "gaussian": "高斯表示",
    "splatting": "Splatting",
    "ranking": "排序",
    "recommendation": "推荐",
    "privacy": "隐私",
    "fairness": "公平性",
    "robust": "鲁棒性",
    "attack": "攻击防御",
    "optimization": "优化",
    "forecasting": "预测",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-root", type=Path, default=DEFAULT_TOPIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before rebuilding.")
    return parser.parse_args()


def slugify(value: object, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:80] or fallback


def table_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text


def md_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("\n", " ").strip()


def topic_name_cn(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("topic_label", "")),
            str(row.get("keywords", "")),
        ]
    ).lower()
    for required_terms, name in CN_TOPIC_RULES:
        if all(contains_term(text, term) for term in required_terms):
            return name

    terms = [term.strip().lower() for term in str(row.get("topic_label", "")).split("/") if term.strip()]
    translated = []
    for term in terms:
        translated.append(TERM_CN.get(term, term))
    if translated:
        return " / ".join(translated[:4])
    return f"{row.get('macro_topic', '主题')}细分方向"


def contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text) is not None


def external_paper_url(record: dict) -> str:
    for key in (
        "openreview_url",
        "html_url",
        "open_access_url",
        "pdf_url",
        "semantic_scholar_url",
        "dblp_url",
        "source_url",
    ):
        value = record.get(key)
        if value and str(value).startswith(("http://", "https://")):
            return str(value)
    doi = record.get("doi")
    if doi:
        doi_text = str(doi).replace("https://doi.org/", "").strip()
        return f"https://doi.org/{doi_text}"
    return ""


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def relative_link(from_path: Path, to_path: Path) -> str:
    return quote(os.path.relpath(to_path, from_path.parent).replace("\\", "/"), safe="/#.-_")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def build_topic_page(
    output_root: Path,
    year: int,
    venue: str,
    topic_row: pd.Series,
    records: list[dict],
    topic_path: Path,
) -> None:
    topic_id = int(topic_row["topic_id"])
    topic_records = [record for record in records if int(record.get("fine_topic", -999999)) == topic_id]
    cn_name = topic_name_cn(topic_row)
    lines = [
        f"# {venue} {year}: {cn_name}",
        "",
        f"- Topic ID: `{topic_id}`",
        f"- Papers: **{int(topic_row['paper_count'])}** ({float(topic_row['paper_share']) * 100:.2f}%)",
        f"- Macro topic: {md_escape(topic_row.get('macro_topic', ''))}",
        f"- English keywords: `{md_escape(topic_row.get('topic_label', ''))}`",
        f"- Keyword pool: {md_escape(topic_row.get('keywords', ''))}",
        "",
        f"[Back to {venue} {year}](README.md) | [Atlas home]({relative_link(topic_path, output_root / 'README.md')})",
        "",
        "## Representative Papers",
        "",
    ]
    representatives = [title for title in str(topic_row.get("representative_titles", "")).split(" || ") if title]
    for title in representatives[:8]:
        lines.append(f"- {md_escape(title)}")

    lines.extend(["", "## Papers", ""])
    for index, record in enumerate(topic_records, start=1):
        title = md_escape(record.get("title", "Untitled"))
        paper_id = slugify(record.get("paper_id") or record.get("doi") or title, f"paper-{index}")
        url = external_paper_url(record)
        title_text = f"[{title}]({url})" if url else title
        authors = record.get("authors") or []
        if isinstance(authors, list):
            authors_text = ", ".join(str(author) for author in authors[:6])
            if len(authors) > 6:
                authors_text += ", et al."
        else:
            authors_text = str(authors)
        method = record.get("fine_topic_assignment_method", "")
        source = record.get("source", "") or record.get("abstract_source", "")
        lines.append(f'<a id="paper-{paper_id}"></a>')
        lines.append(f"{index}. {title_text}")
        metadata = []
        if authors_text:
            metadata.append(authors_text)
        if method:
            metadata.append(f"assignment: `{method}`")
        if source:
            metadata.append(f"source: `{source}`")
        if metadata:
            lines.append(f"   - {'; '.join(metadata)}")
    write(topic_path, "\n".join(lines))


def build_venue_page(
    output_root: Path,
    year: int,
    venue: str,
    summary_row: pd.Series,
    topics: pd.DataFrame,
    venue_path: Path,
) -> None:
    lines = [
        f"# {venue} {year} Topic Atlas",
        "",
        f"- Papers: **{int(summary_row['papers'])}**",
        f"- Fine topics: **{int(summary_row['final_topics'])}**",
        f"- Raw HDBSCAN outliers: {int(summary_row['raw_outliers'])}",
        f"- Final outliers after centroid reassignment: {int(summary_row['final_outliers'])}",
        "",
        f"[Back to {year}](../README.md) | [Atlas home]({relative_link(venue_path, output_root / 'README.md')})",
        "",
        "## Topics",
        "",
        "| Topic | 中文主题名 | Papers | Share | Macro | Keywords | Representative paper |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for _, topic in topics.iterrows():
        topic_file = Path(f"topic-{int(topic['topic_id']):03d}.md")
        topic_link = f"[{int(topic['topic_id']):03d}]({quote(str(topic_file), safe='/#.-_')})"
        cn_name = topic_name_cn(topic)
        representative = str(topic.get("representative_titles", "")).split(" || ")[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    topic_link,
                    table_escape(cn_name),
                    str(int(topic["paper_count"])),
                    f"{float(topic['paper_share']) * 100:.2f}%",
                    table_escape(topic.get("macro_topic", "")),
                    f"`{table_escape(topic.get('topic_label', ''))}`",
                    table_escape(representative),
                ]
            )
            + " |"
        )
    write(venue_path, "\n".join(lines))


def build_year_page(output_root: Path, year: int, rows: pd.DataFrame, year_path: Path) -> None:
    lines = [
        f"# {year} Topic Atlas",
        "",
        f"[Atlas home]({relative_link(year_path, output_root / 'README.md')})",
        "",
        "| Venue | Papers | Fine topics | Largest topic | Median topic size |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in rows.sort_values("venue").iterrows():
        venue = row["venue"]
        lines.append(
            f"| [{venue}]({quote(str(Path(venue) / 'README.md'), safe='/#.-_')}) "
            f"| {int(row['papers'])} | {int(row['final_topics'])} "
            f"| {int(row['largest_topic'])} | {float(row['median_topic_size']):.1f} |"
        )
    write(year_path, "\n".join(lines))


def build_home_page(output_root: Path, summary: pd.DataFrame, topic_index: pd.DataFrame) -> None:
    by_venue = (
        summary.groupby("venue")
        .agg(
            years=("year", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            papers=("papers", "sum"),
            topics=("final_topics", "sum"),
            avg_topics=("final_topics", "mean"),
        )
        .sort_index()
        .reset_index()
    )
    lines = [
        "# AI Paper Topic Atlas",
        "",
        "Fine-grained topic index generated from AI conference and journal papers.",
        "",
        "Navigation pattern: **year -> venue -> topic -> paper**.",
        "",
        f"- Venue-year groups: **{len(summary)}**",
        f"- Papers: **{int(summary['papers'].sum()):,}**",
        f"- Fine topics: **{int(summary['final_topics'].sum()):,}**",
        f"- Final outliers: **{int(summary['final_outliers'].sum())}**",
        "",
        "## Years",
        "",
    ]
    for year in sorted(summary["year"].unique(), reverse=True):
        year_rows = summary[summary["year"] == year]
        lines.append(
            f"- [{year}]({quote(str(Path(str(year)) / 'README.md'), safe='/#.-_')}) "
            f"- {len(year_rows)} venues, {int(year_rows['papers'].sum()):,} papers, "
            f"{int(year_rows['final_topics'].sum()):,} topics"
        )

    lines.extend(
        [
            "",
            "## Venues",
            "",
            "| Venue | Years | Papers | Fine topics | Avg topics/year |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in by_venue.iterrows():
        lines.append(
            f"| {row['venue']} | {int(row['first_year'])}-{int(row['last_year'])} "
            f"({int(row['years'])}) | {int(row['papers']):,} | {int(row['topics']):,} "
            f"| {float(row['avg_topics']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Data Files",
            "",
            "- [venue_year_summary.csv](data/venue_year_summary.csv)",
            "- [topic_index.csv](data/topic_index.csv)",
            "",
            "Topic labels include a reproducible English keyword label and a heuristic Chinese display name. "
            "The Chinese name is designed for browsing and visualization; use representative paper titles for audit.",
        ]
    )
    write(output_root / "README.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    topic_root = args.topic_root
    output_root = args.output_root
    summary_path = topic_root / "run_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path).sort_values(["year", "venue"])
    topic_rows = []
    for _, group in summary.iterrows():
        year = int(group["year"])
        venue = str(group["venue"])
        group_dir = topic_root / f"venue={venue}" / f"year={year}"
        topic_summary_path = group_dir / "topic_summary.csv"
        papers_path = group_dir / "papers_with_fine_topics.jsonl"
        if not topic_summary_path.exists() or not papers_path.exists():
            raise SystemExit(f"Missing topic outputs for {venue} {year}")

        topics = pd.read_csv(topic_summary_path).sort_values("paper_count", ascending=False)
        records = read_jsonl(papers_path)
        venue_output_dir = output_root / str(year) / venue
        venue_page = venue_output_dir / "README.md"

        for _, topic in topics.iterrows():
            topic_file = venue_output_dir / f"topic-{int(topic['topic_id']):03d}.md"
            build_topic_page(output_root, year, venue, topic, records, topic_file)
            topic_record = topic.to_dict()
            topic_record["topic_name_cn"] = topic_name_cn(topic)
            topic_record["topic_page"] = str(topic_file.relative_to(output_root)).replace("\\", "/")
            topic_rows.append(topic_record)

        build_venue_page(output_root, year, venue, group, topics, venue_page)

    for year, rows in summary.groupby("year"):
        build_year_page(output_root, int(year), rows, output_root / str(int(year)) / "README.md")

    topic_index = pd.DataFrame(topic_rows)
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(data_dir / "venue_year_summary.csv", index=False, encoding="utf-8-sig")
    topic_index.to_csv(data_dir / "topic_index.csv", index=False, encoding="utf-8-sig")
    build_home_page(output_root, summary, topic_index)
    print(f"Wrote atlas to {output_root}")
    print(f"Venue-year groups: {len(summary)}")
    print(f"Topics: {len(topic_index)}")


if __name__ == "__main__":
    main()
