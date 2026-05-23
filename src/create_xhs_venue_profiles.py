"""Create card-style venue profile visuals for Xiaohongshu posts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25")
OUTPUT_DIR = Path("docs/visuals/xhs_venue_profiles_2020_2026")

BG = "#F8F7F3"
CARD = "#FFFFFF"
INK = "#202124"
MUTED = "#6B7280"
GRID = "#E2E0D8"
BLUE = "#3B82F6"
ORANGE = "#F97316"
PURPLE = "#8B5CF6"
PINK = "#EC4899"
GREEN = "#10B981"
TEAL = "#14B8A6"
RED = "#EF4444"
GOLD = "#F59E0B"
SLATE = "#64748B"

VENUE_ORDER = [
    "AAAI",
    "IJCAI",
    "ICLR",
    "ICML",
    "NeurIPS",
    "CVPR",
    "ICCV",
    "ECCV",
    "ACL",
    "EMNLP",
    "NAACL",
    "ACMMM",
    "KDD",
    "SIGIR",
    "WWW",
]

VENUE_GROUPS = {
    "AAAI": "综合 AI",
    "IJCAI": "综合 AI",
    "ICLR": "机器学习",
    "ICML": "机器学习",
    "NeurIPS": "机器学习",
    "CVPR": "计算机视觉",
    "ICCV": "计算机视觉",
    "ECCV": "计算机视觉",
    "ACL": "自然语言处理",
    "EMNLP": "自然语言处理",
    "NAACL": "自然语言处理",
    "ACMMM": "多媒体",
    "KDD": "数据挖掘",
    "SIGIR": "信息检索",
    "WWW": "Web / 社会计算",
}

VENUE_COLORS = {
    "AAAI": PURPLE,
    "IJCAI": PURPLE,
    "ICLR": BLUE,
    "ICML": BLUE,
    "NeurIPS": BLUE,
    "CVPR": ORANGE,
    "ICCV": ORANGE,
    "ECCV": ORANGE,
    "ACL": GOLD,
    "EMNLP": GOLD,
    "NAACL": GOLD,
    "ACMMM": PINK,
    "KDD": GREEN,
    "SIGIR": TEAL,
    "WWW": SLATE,
}

FAMILY_COLORS = {
    "大模型": BLUE,
    "多模态": PURPLE,
    "生成模型": PINK,
    "视觉/三维/视频": ORANGE,
    "强化学习": GREEN,
    "优化/理论": SLATE,
    "推荐/检索/图": TEAL,
    "可信安全": RED,
    "AI4Science/健康": "#84CC16",
    "NLP/语音": GOLD,
    "其他": "#A3A3A3",
}


def configure_font() -> fm.FontProperties:
    candidates = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    plt.rcParams["axes.unicode_minus"] = False
    return fm.FontProperties()


FONT = configure_font()


def family_from_parent(parent: str, label: str) -> str:
    parent = str(parent)
    label = str(label)
    if parent in {"Security / Privacy", "Security / Vision", "Trustworthy LLM"}:
        return "可信安全"
    if parent in {"Multimodal LLM", "Multimodal AI", "Multimodal Generation", "Multimedia Retrieval"}:
        return "多模态"
    if parent in {"Generative AI", "Text Generation"}:
        return "生成模型"
    if parent in {"Computer Vision", "3D Vision", "Video Understanding", "Event Vision", "Embodied AI"}:
        return "视觉/三维/视频"
    if parent in {"Reinforcement Learning", "Agents / Game Theory"}:
        return "强化学习"
    if parent in {"Optimization", "Statistical Learning", "Probabilistic ML", "AutoML", "Causal ML"}:
        return "优化/理论"
    if parent in {
        "Recommendation",
        "Graph Learning",
        "Graph Foundation Models",
        "Knowledge Graph",
        "Information Retrieval",
        "Anomaly Detection",
    }:
        return "推荐/检索/图"
    if parent in {"AI for Science", "AI for Health", "AI for Neuroscience"}:
        return "AI4Science/健康"
    if parent in {"NLP", "Speech / Audio", "Information Extraction"}:
        return "NLP/语音"
    if "大模型" in label or "LLM" in parent or parent == "Code Intelligence":
        return "大模型"
    return "其他"


def compact_topic_label(label: str) -> str:
    label = re.sub(r"（[^）]+）", "", str(label))
    rules = [
        ("Gaussian Splatting", "3D GS/新视角"),
        ("视频扩散", "视频扩散"),
        ("文生图", "文生图/图像编辑"),
        ("多模态大模型", "多模态大模型"),
        ("多模态理解", "多模态理解"),
        ("大模型推理：RL", "RL推理"),
        ("大模型推理", "大模型推理"),
        ("高效大模型", "高效大模型"),
        ("长上下文", "长上下文"),
        ("检索增强大模型", "RAG/知识问答"),
        ("大模型训练", "大模型训练"),
        ("大模型社会安全", "LLM安全"),
        ("大模型评测", "LLM评测"),
        ("代码大模型", "代码大模型"),
        ("语言模型分析", "模型分析"),
        ("强化学习", "强化学习"),
        ("在线决策", "Bandit/在线决策"),
        ("优化理论", "优化理论"),
        ("统计学习理论", "统计学习理论"),
        ("三维视觉", "三维视觉"),
        ("开放词汇视觉", "开放词汇视觉"),
        ("自动驾驶感知", "自动驾驶感知"),
        ("底层视觉", "底层视觉"),
        ("视觉感知", "视觉感知"),
        ("视频理解", "视频理解"),
        ("具身智能", "具身智能"),
        ("多媒体检索", "多媒体检索"),
        ("多媒体安全", "多媒体安全"),
        ("多模态音视频生成", "音视频生成"),
        ("推荐系统", "推荐系统"),
        ("图基础模型", "图基础模型"),
        ("图学习", "图学习"),
        ("知识图谱", "知识图谱"),
        ("时序建模", "时间序列"),
        ("AI4Science", "AI4Science"),
        ("可信安全", "可信安全"),
        ("迁移泛化", "迁移泛化"),
        ("鲁棒泛化", "鲁棒泛化/OOD"),
        ("语音音频", "语音音频"),
        ("多语言", "多语言NLP"),
        ("知识抽取", "知识抽取"),
        ("联邦", "联邦学习"),
    ]
    for key, value in rules:
        if key in label:
            return value
    return label.split("：", 1)[0][:9]


def load_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topics = pd.read_csv(results_dir / "topic_summary_by_venue_year.csv")
    top10 = pd.read_csv(results_dir / "top10_topics_by_venue_year.csv")
    summary = pd.read_csv(results_dir / "run_summary_by_venue_year.csv")
    topics["family"] = [
        family_from_parent(parent, label)
        for parent, label in zip(topics["specific_parent_category"], topics["specific_label_cn"], strict=False)
    ]
    top10 = top10.merge(topics[["venue_year_topic_id", "family"]], on="venue_year_topic_id", how="left")
    return topics, top10, summary


def add_card(fig, x: float, y: float, w: float, h: float, edge: str = GRID, face: str = CARD) -> None:
    rect = plt.Rectangle(
        (x, y),
        w,
        h,
        transform=fig.transFigure,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        zorder=-10,
    )
    fig.patches.append(rect)


def fig_base():
    fig = plt.figure(figsize=(6, 8.4), dpi=200)
    fig.patch.set_facecolor(BG)
    return fig


def add_footer(fig) -> None:
    fig.text(
        0.06,
        0.025,
        "AI Paper Trends | 2020-2026 主会 accepted papers | 2026 仅含 ICLR + AAAI",
        fontsize=7.4,
        color=MUTED,
        fontproperties=FONT,
    )


def draw_metric(fig, x: float, y: float, w: float, title: str, value: str, color: str) -> None:
    add_card(fig, x, y, w, 0.082)
    fig.text(x + 0.018, y + 0.047, value, fontsize=17, fontweight="bold", color=color, fontproperties=FONT)
    fig.text(x + 0.018, y + 0.019, title, fontsize=7.6, color=MUTED, fontproperties=FONT)


def axis_in_card(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    left_pad: float = 0.05,
    right_pad: float = 0.035,
    bottom_pad: float = 0.05,
    top_pad: float = 0.075,
):
    add_card(fig, x, y, w, h)
    ax = fig.add_axes(
        [x + left_pad, y + bottom_pad, w - left_pad - right_pad, h - bottom_pad - top_pad],
        facecolor=CARD,
        zorder=5,
    )
    return ax


def draw_year_volume(fig, venue_summary: pd.DataFrame, x: float, y: float, w: float, h: float, color: str) -> None:
    ax = axis_in_card(fig, x, y, w, h, left_pad=0.052, right_pad=0.03, bottom_pad=0.045, top_pad=0.078)
    years = venue_summary["year"].to_numpy()
    papers = venue_summary["papers"].to_numpy()
    ax.bar(years, papers, color=color, alpha=0.86, width=0.55)
    ax.set_title("年度论文规模", loc="left", fontsize=10, fontweight="bold", fontproperties=FONT, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(y)) for y in years], rotation=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ymax = papers.max() if len(papers) else 1
    for yr, val in zip(years, papers, strict=False):
        ax.text(yr, val + ymax * 0.035, f"{int(val):,}", ha="center", fontsize=6.3, color=MUTED, fontproperties=FONT)


def draw_latest_topics(fig, latest_topics: pd.DataFrame, x: float, y: float, w: float, h: float, color: str) -> None:
    ax = axis_in_card(fig, x, y, w, h, left_pad=0.155, right_pad=0.035, bottom_pad=0.045, top_pad=0.075)
    sub = latest_topics.sort_values("rank").head(6).iloc[::-1]
    labels = [compact_topic_label(v) for v in sub["specific_label_cn"]]
    values = sub["share_pct"].to_numpy()
    colors = [FAMILY_COLORS.get(v, color) for v in sub["family"]]
    ax.barh(np.arange(len(sub)), values, color=colors, height=0.62)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(labels, fontsize=7.4, fontproperties=FONT)
    ax.set_title("最新年份 Top 6 主题", loc="left", fontsize=10, fontweight="bold", fontproperties=FONT, pad=8)
    ax.set_xlim(0, max(22, float(values.max()) * 1.25 if len(values) else 20))
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED)
    ax.tick_params(axis="y", colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for i, val in enumerate(values):
        ax.text(val + 0.25, i, f"{val:.1f}%", va="center", fontsize=6.4, color=MUTED, fontproperties=FONT)


def draw_family_dna(fig, venue_topics: pd.DataFrame, x: float, y: float, w: float, h: float) -> None:
    add_card(fig, x, y, w, h)
    fig.text(x + 0.025, y + h - 0.04, "研究风格 DNA", fontsize=10, fontweight="bold", color=INK, fontproperties=FONT)
    fam = venue_topics.groupby("family")["count"].sum().sort_values(ascending=False)
    fam = fam[fam.index != "其他"].head(4)
    total = float(fam.sum()) or 1.0
    y0 = y + h - 0.075
    spacing = min(0.037, max(0.026, (h - 0.105) / max(len(fam), 1)))
    for i, (name, count) in enumerate(fam.items()):
        yy = y0 - i * spacing
        pct = count / total
        fig.text(x + 0.025, yy, name, fontsize=7.5, color=INK, fontproperties=FONT)
        fig.text(x + w - 0.03, yy, f"{pct * 100:.0f}%", fontsize=7.2, color=MUTED, ha="right", fontproperties=FONT)
        bx = x + 0.025
        by = yy - 0.018
        bw = w - 0.055
        fig.patches.append(
            plt.Rectangle((bx, by), bw, 0.0075, transform=fig.transFigure, facecolor="#EEECE4", edgecolor="none")
        )
        fig.patches.append(
            plt.Rectangle(
                (bx, by),
                bw * pct,
                0.0075,
                transform=fig.transFigure,
                facecolor=FAMILY_COLORS.get(name, "#999999"),
                edgecolor="none",
            )
        )


def draw_yearly_champions(fig, venue_top10: pd.DataFrame, x: float, y: float, w: float, h: float, color: str) -> None:
    add_card(fig, x, y, w, h)
    fig.text(x + 0.025, y + h - 0.038, "历年第一主题", fontsize=10, fontweight="bold", color=INK, fontproperties=FONT)
    champions = venue_top10.sort_values(["year", "rank"]).groupby("year").head(1)
    n = len(champions)
    if n == 0:
        return
    col_w = (w - 0.05) / n
    base_y = y + 0.04
    top_y = y + h - 0.078
    for i, (_, row) in enumerate(champions.iterrows()):
        cx = x + 0.025 + i * col_w
        fig.patches.append(
            plt.Rectangle(
                (cx + 0.003, base_y),
                max(col_w - 0.008, 0.02),
                top_y - base_y,
                transform=fig.transFigure,
                facecolor="#FBFAF7",
                edgecolor="#ECE8DE",
                linewidth=0.7,
            )
        )
        cell_h = top_y - base_y
        fig.text(cx + col_w / 2, base_y + cell_h - 0.022, str(int(row["year"])), ha="center", fontsize=7.2, color=color, fontweight="bold", fontproperties=FONT)
        label = compact_topic_label(row["specific_label_cn"])
        if len(label) > 7:
            label = label[:7] + "…"
        fig.text(cx + col_w / 2, base_y + 0.012, label, ha="center", fontsize=6.7, color=INK, fontproperties=FONT)


def draw_family_trend(fig, venue_topics: pd.DataFrame, venue_summary: pd.DataFrame, x: float, y: float, w: float, h: float) -> None:
    ax = axis_in_card(fig, x, y, w, h, left_pad=0.055, right_pad=0.03, bottom_pad=0.052, top_pad=0.075)
    top_families = venue_topics.groupby("family")["count"].sum().sort_values(ascending=False)
    top_families = [f for f in top_families.index if f != "其他"][:3]
    yearly_total = venue_summary.set_index("year")["papers"]
    for fam in top_families:
        series = venue_topics[venue_topics["family"] == fam].groupby("year")["count"].sum()
        share = (series / yearly_total * 100).reindex(yearly_total.index, fill_value=0)
        ax.plot(share.index, share.values, marker="o", linewidth=1.8, markersize=3.5, label=fam, color=FAMILY_COLORS.get(fam))
    ax.set_title("核心风格随年份变化", loc="left", fontsize=10, fontweight="bold", fontproperties=FONT, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_xticks(yearly_total.index)
    ax.set_xticklabels([str(int(y)) for y in yearly_total.index])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    leg = ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=5.8, prop=FONT)
    for text in leg.get_texts():
        text.set_fontproperties(FONT)


def venue_tagline(venue: str, latest_top: list[str]) -> str:
    joined = "、".join(latest_top[:3])
    return f"{VENUE_GROUPS.get(venue, 'AI')} 会议画像：最近一年主要集中在 {joined}"


def draw_venue_profile(
    venue: str,
    topics: pd.DataFrame,
    top10: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    venue_summary = summary[summary["venue"] == venue].sort_values("year")
    venue_topics = topics[topics["venue"] == venue].copy()
    venue_top10 = top10[top10["venue"] == venue].copy()
    latest_year = int(venue_summary["year"].max())
    latest_summary = venue_summary[venue_summary["year"] == latest_year].iloc[0]
    latest_topics = venue_top10[venue_top10["year"] == latest_year].sort_values("rank")
    color = VENUE_COLORS.get(venue, BLUE)
    latest_labels = [compact_topic_label(v) for v in latest_topics["specific_label_cn"].head(3)]

    fig = fig_base()
    fig.text(0.055, 0.955, f"{venue} 会议画像", fontsize=25, fontweight="bold", color=INK, fontproperties=FONT)
    fig.text(0.057, 0.92, venue_tagline(venue, latest_labels), fontsize=8.6, color=MUTED, fontproperties=FONT)
    fig.text(
        0.057,
        0.892,
        f"{int(venue_summary['year'].min())}-{latest_year} | {VENUE_GROUPS.get(venue, 'AI')} | 主会 accepted-only",
        fontsize=7.8,
        color=color,
        fontproperties=FONT,
        fontweight="bold",
    )

    draw_metric(fig, 0.055, 0.795, 0.19, "覆盖年份", f"{len(venue_summary)}年", color)
    draw_metric(fig, 0.265, 0.795, 0.21, "论文总数", f"{int(venue_summary['papers'].sum()):,}", color)
    draw_metric(fig, 0.495, 0.795, 0.2, "最新年份", f"{latest_year}", color)
    draw_metric(fig, 0.715, 0.795, 0.23, "最新论文", f"{int(latest_summary['papers']):,}", color)

    draw_year_volume(fig, venue_summary, 0.055, 0.535, 0.42, 0.22, color)
    draw_family_dna(fig, venue_topics, 0.515, 0.535, 0.43, 0.22)
    draw_latest_topics(fig, latest_topics, 0.055, 0.265, 0.42, 0.235, color)
    draw_family_trend(fig, venue_topics, venue_summary, 0.515, 0.265, 0.43, 0.235)
    draw_yearly_champions(fig, venue_top10, 0.055, 0.045, 0.89, 0.18, color)

    add_footer(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{VENUE_ORDER.index(venue) + 1:02d}_{venue}_profile.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_index(topics: pd.DataFrame, top10: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    fig = fig_base()
    fig.text(0.055, 0.955, "AI 顶会风格索引", fontsize=25, fontweight="bold", color=INK, fontproperties=FONT)
    fig.text(0.057, 0.92, "每个格子是该会议最新年份 Top 3 主题，完整画像见单独会议页", fontsize=8.6, color=MUTED, fontproperties=FONT)
    latest = summary.loc[summary.groupby("venue")["year"].idxmax()][["venue", "year", "papers"]]
    latest = latest.set_index("venue").reindex([v for v in VENUE_ORDER if v in latest["venue"].values]).reset_index()
    left, top_y = 0.055, 0.835
    w, h = 0.29, 0.126
    gap_x, gap_y = 0.035, 0.024
    for i, row in latest.iterrows():
        venue = row["venue"]
        year = int(row["year"])
        x = left + (i % 3) * (w + gap_x)
        y = top_y - (i // 3) * (h + gap_y)
        color = VENUE_COLORS.get(venue, BLUE)
        add_card(fig, x, y - h, w, h)
        fig.text(x + 0.012, y - 0.026, f"{venue} {year}", fontsize=10.7, color=color, fontweight="bold", fontproperties=FONT)
        fig.text(x + w - 0.012, y - 0.026, f"{int(row['papers']):,}篇", fontsize=7, color=MUTED, ha="right", fontproperties=FONT)
        sub = top10[(top10["venue"] == venue) & (top10["year"] == year)].sort_values("rank").head(3)
        for j, (_, topic) in enumerate(sub.iterrows(), start=1):
            fig.text(
                x + 0.014,
                y - 0.042 - j * 0.023,
                f"{j}. {compact_topic_label(topic['specific_label_cn'])}",
                fontsize=7.2,
                color=INK,
                fontproperties=FONT,
            )
            fig.text(
                x + w - 0.014,
                y - 0.042 - j * 0.023,
                f"{float(topic['share_pct']):.1f}%",
                fontsize=6.5,
                color=MUTED,
                ha="right",
                fontproperties=FONT,
            )
    add_footer(fig)
    fig.savefig(out_dir / "00_all_venues_index.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    files = "\n".join(f"- `{p.name}`" for p in sorted(out_dir.glob("*.png")))
    (out_dir / "README.md").write_text(
        "# Xiaohongshu Venue Profiles\n\n"
        "Card-style vertical visuals for AI conference trend notes.\n\n"
        f"{files}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    topics, top10, summary = load_data(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old in args.output_dir.glob("*.png"):
        old.unlink()
    draw_index(topics, top10, summary, args.output_dir)
    for venue in VENUE_ORDER:
        if venue in set(summary["venue"]):
            draw_venue_profile(venue, topics, top10, summary, args.output_dir)
    write_readme(args.output_dir)
    print(f"Wrote venue profiles to {args.output_dir}")


if __name__ == "__main__":
    main()
