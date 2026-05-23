"""Create composition-atlas visuals for venue-year AI paper topics.

The previous profile cards are intentionally narrative. This script focuses on
composition: every venue-year is represented as a 100% stacked bar over topic
families, with exact topic-level composition exported as CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_xhs_venue_profiles import (
    BG,
    CARD,
    FAMILY_COLORS,
    GRID,
    INK,
    MUTED,
    OUTPUT_DIR as PROFILE_OUTPUT_DIR,
    RESULTS_DIR,
    VENUE_COLORS,
    VENUE_GROUPS,
    VENUE_ORDER,
    compact_topic_label,
    configure_font,
    family_from_parent,
)


OUTPUT_DIR = Path("docs/visuals/xhs_composition_atlas_2020_2026")
FAMILY_ORDER = [
    "大模型",
    "多模态",
    "生成模型",
    "视觉/三维/视频",
    "强化学习",
    "优化/理论",
    "推荐/检索/图",
    "可信安全",
    "AI4Science/健康",
    "NLP/语音",
    "其他",
]

FONT = configure_font()


def fig_base(height: float = 8.4):
    fig = plt.figure(figsize=(6.2, height), dpi=210)
    fig.patch.set_facecolor(BG)
    return fig


def add_card(fig, x: float, y: float, w: float, h: float) -> None:
    fig.patches.append(
        plt.Rectangle(
            (x, y),
            w,
            h,
            transform=fig.transFigure,
            facecolor=CARD,
            edgecolor=GRID,
            linewidth=1.0,
            zorder=-10,
        )
    )


def add_footer(fig) -> None:
    fig.text(
        0.055,
        0.023,
        "AI Paper Trends | 2020-2026 主会 accepted papers | 组成=主题大类 100% 堆叠；精确细主题见 CSV",
        fontsize=6.8,
        color=MUTED,
        fontproperties=FONT,
    )


def load_data(results_dir: Path):
    topics = pd.read_csv(results_dir / "topic_summary_by_venue_year.csv")
    top10 = pd.read_csv(results_dir / "top10_topics_by_venue_year.csv")
    summary = pd.read_csv(results_dir / "run_summary_by_venue_year.csv")
    topics["family"] = [
        family_from_parent(parent, label)
        for parent, label in zip(topics["specific_parent_category"], topics["specific_label_cn"], strict=False)
    ]
    topics["compact_label_cn"] = [compact_topic_label(label) for label in topics["specific_label_cn"]]
    topics["share_pct"] = topics["share"] * 100
    top10 = top10.merge(topics[["venue_year_topic_id", "family", "compact_label_cn"]], on="venue_year_topic_id", how="left")

    family = (
        topics.groupby(["venue", "year", "family"], as_index=False)
        .agg(count=("count", "sum"), venue_year_total=("venue_year_total", "first"))
    )
    family["share_pct"] = family["count"] / family["venue_year_total"] * 100
    family["family"] = pd.Categorical(family["family"], FAMILY_ORDER, ordered=True)
    family = family.sort_values(["venue", "year", "family"])
    return topics, top10, summary, family


def write_tables(out_dir: Path, topics: pd.DataFrame, family: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    family[
        ["venue", "year", "family", "count", "venue_year_total", "share_pct"]
    ].to_csv(out_dir / "venue_year_family_composition.csv", index=False)
    topic_table = topics[
        [
            "venue",
            "year",
            "venue_year_topic_id",
            "specific_label_cn",
            "compact_label_cn",
            "family",
            "count",
            "venue_year_total",
            "share_pct",
            "topic_keywords",
            "representative_titles",
        ]
    ].sort_values(["venue", "year", "count"], ascending=[True, True, False])
    topic_table.to_csv(out_dir / "venue_year_topic_composition_full.csv", index=False)


def draw_legend(fig, x: float, y: float, families: list[str], cols: int = 3) -> None:
    for i, fam in enumerate(families):
        col = i % cols
        row = i // cols
        xx = x + col * 0.29
        yy = y - row * 0.025
        fig.patches.append(
            plt.Rectangle(
                (xx, yy - 0.008),
                0.014,
                0.014,
                transform=fig.transFigure,
                facecolor=FAMILY_COLORS.get(fam, "#999999"),
                edgecolor="none",
            )
        )
        fig.text(xx + 0.019, yy - 0.001, fam, fontsize=6.9, color=MUTED, fontproperties=FONT, va="center")


def draw_stacked_bar(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    shares: dict[str, float],
    label_inside: bool = True,
) -> None:
    left = x
    for fam in FAMILY_ORDER:
        pct = float(shares.get(fam, 0.0))
        if pct <= 0:
            continue
        seg_w = w * pct / 100.0
        fig.patches.append(
            plt.Rectangle(
                (left, y),
                seg_w,
                h,
                transform=fig.transFigure,
                facecolor=FAMILY_COLORS.get(fam, "#999999"),
                edgecolor=BG,
                linewidth=0.45,
            )
        )
        if label_inside and pct >= 11:
            fig.text(
                left + seg_w / 2,
                y + h / 2,
                f"{pct:.0f}%",
                fontsize=6.4,
                color="white",
                ha="center",
                va="center",
                fontproperties=FONT,
                fontweight="bold",
            )
        left += seg_w


def top_topic_text(top10: pd.DataFrame, venue: str, year: int, n: int = 2) -> str:
    rows = top10[(top10["venue"] == venue) & (top10["year"] == year)].sort_values("rank").head(n)
    parts = []
    for _, row in rows.iterrows():
        parts.append(f"{compact_topic_label(row['specific_label_cn'])} {float(row['share_pct']):.1f}%")
    return "；".join(parts)


def draw_year_page(year: int, family: pd.DataFrame, top10: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    rows = summary[summary["year"] == year].copy()
    rows["venue_order"] = rows["venue"].map({v: i for i, v in enumerate(VENUE_ORDER)}).fillna(999)
    rows = rows.sort_values(["venue_order", "venue"])
    n = len(rows)
    fig = fig_base(height=8.4 if n > 8 else 5.6)
    fig.text(0.055, 0.955, f"{year} 顶会主题组成", fontsize=24, fontweight="bold", color=INK, fontproperties=FONT)
    fig.text(
        0.057,
        0.922,
        f"{n} 个会议，{int(rows['papers'].sum()):,} 篇主会 accepted paper；每条为 100% 主题大类构成",
        fontsize=8.0,
        color=MUTED,
        fontproperties=FONT,
    )

    top = 0.855
    bottom = 0.105
    row_h = min(0.053, (top - bottom) / max(n, 1) * 0.78)
    gap = ((top - bottom) - n * row_h) / max(n - 1, 1) if n > 1 else 0
    for i, (_, row) in enumerate(rows.iterrows()):
        venue = row["venue"]
        y = top - i * (row_h + gap) - row_h
        color = VENUE_COLORS.get(venue, INK)
        fig.text(0.045, y + row_h * 0.55, venue, fontsize=8.8, color=color, fontproperties=FONT, fontweight="bold", va="center")
        fig.text(
            0.151,
            y + row_h * 0.55,
            f"{int(row['papers']):,}篇",
            fontsize=6.7,
            color=MUTED,
            fontproperties=FONT,
            ha="right",
            va="center",
        )
        shares = {
            r["family"]: r["share_pct"]
            for _, r in family[(family["venue"] == venue) & (family["year"] == year)].iterrows()
        }
        draw_stacked_bar(fig, 0.17, y + row_h * 0.18, 0.51, row_h * 0.64, shares)
        fig.text(
            0.70,
            y + row_h * 0.55,
            top_topic_text(top10, venue, year, n=2),
            fontsize=6.5,
            color=INK,
            fontproperties=FONT,
            va="center",
        )

    add_footer(fig)
    (out_dir / "by_year").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "by_year" / f"{year}_composition_by_venue.png", dpi=210, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_venue_page(venue: str, family: pd.DataFrame, top10: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    rows = summary[summary["venue"] == venue].sort_values("year")
    group = VENUE_GROUPS.get(venue, "AI")
    color = VENUE_COLORS.get(venue, INK)
    fig = fig_base(height=6.8)
    fig.text(0.055, 0.945, f"{venue} 历年主题组成", fontsize=23, fontweight="bold", color=INK, fontproperties=FONT)
    fig.text(
        0.057,
        0.908,
        f"{group} | {int(rows['year'].min())}-{int(rows['year'].max())} | {int(rows['papers'].sum()):,} 篇论文",
        fontsize=8.2,
        color=color,
        fontproperties=FONT,
    )

    top = 0.82
    bottom = 0.21
    n = len(rows)
    row_h = 0.055
    gap = ((top - bottom) - n * row_h) / max(n - 1, 1) if n > 1 else 0
    for i, (_, row) in enumerate(rows.iterrows()):
        year = int(row["year"])
        y = top - i * (row_h + gap) - row_h
        fig.text(0.06, y + row_h * 0.55, str(year), fontsize=9.2, color=color, fontweight="bold", fontproperties=FONT, va="center")
        fig.text(0.105, y + row_h * 0.55, f"{int(row['papers']):,}篇", fontsize=6.8, color=MUTED, fontproperties=FONT, va="center")
        shares = {
            r["family"]: r["share_pct"]
            for _, r in family[(family["venue"] == venue) & (family["year"] == year)].iterrows()
        }
        draw_stacked_bar(fig, 0.19, y + row_h * 0.18, 0.56, row_h * 0.64, shares)
        fig.text(
            0.77,
            y + row_h * 0.55,
            top_topic_text(top10, venue, year, n=2),
            fontsize=6.7,
            color=INK,
            fontproperties=FONT,
            va="center",
        )

    add_card(fig, 0.055, 0.095, 0.89, 0.075)
    latest_year = int(rows["year"].max())
    latest = top10[(top10["venue"] == venue) & (top10["year"] == latest_year)].sort_values("rank").head(5)
    fig.text(0.075, 0.142, f"{latest_year} 细主题 Top 5", fontsize=8.5, color=INK, fontweight="bold", fontproperties=FONT)
    fig.text(
        0.075,
        0.113,
        " / ".join([compact_topic_label(v) for v in latest["specific_label_cn"]]),
        fontsize=7.2,
        color=MUTED,
        fontproperties=FONT,
    )
    add_footer(fig)
    (out_dir / "by_venue").mkdir(parents=True, exist_ok=True)
    idx = VENUE_ORDER.index(venue) + 1 if venue in VENUE_ORDER else 99
    fig.savefig(out_dir / "by_venue" / f"{idx:02d}_{venue}_yearly_composition.png", dpi=210, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_overview(summary: pd.DataFrame, out_dir: Path) -> None:
    fig = fig_base(height=6.2)
    fig.text(0.055, 0.935, "顶会主题组成图谱", fontsize=25, fontweight="bold", color=INK, fontproperties=FONT)
    fig.text(
        0.057,
        0.89,
        "每个会议-年份都画成 100% 组成条：不是只看 Top 3，而是看完整结构。",
        fontsize=8.5,
        color=MUTED,
        fontproperties=FONT,
    )
    metrics = [
        ("论文", f"{int(summary['papers'].sum()):,}"),
        ("会议", f"{summary['venue'].nunique()}"),
        ("会议-年份", f"{len(summary)}"),
        ("年份", f"{int(summary['year'].min())}-{int(summary['year'].max())}"),
    ]
    for i, (label, value) in enumerate(metrics):
        x = 0.06 + (i % 2) * 0.43
        y = 0.72 - (i // 2) * 0.14
        add_card(fig, x, y, 0.37, 0.095)
        fig.text(x + 0.02, y + 0.055, value, fontsize=18, fontweight="bold", color=INK, fontproperties=FONT)
        fig.text(x + 0.02, y + 0.024, label, fontsize=7.6, color=MUTED, fontproperties=FONT)

    add_card(fig, 0.06, 0.235, 0.84, 0.235)
    fig.text(0.085, 0.425, "读图方式", fontsize=11, fontweight="bold", color=INK, fontproperties=FONT)
    notes = [
        "1. 每一条横条等于一个会议某一年的全部论文。",
        "2. 颜色表示主题大类，长度表示占比。",
        "3. 右侧文字给该会议-年份最主要的细主题。",
        "4. 精确到每个细主题的完整构成已导出为 CSV。",
    ]
    for i, note in enumerate(notes):
        fig.text(0.085, 0.385 - i * 0.037, note, fontsize=8.3, color=INK, fontproperties=FONT)
    draw_legend(fig, 0.07, 0.17, FAMILY_ORDER, cols=3)
    add_footer(fig)
    fig.savefig(out_dir / "00_composition_atlas_cover.png", dpi=210, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        "# Composition Atlas\n\n"
        "This directory contains venue-year composition visuals and exact CSV tables.\n\n"
        "- `00_composition_atlas_cover.png`: how to read the atlas\n"
        "- `by_year/*.png`: one image per year, all available venues in that year\n"
        "- `by_venue/*.png`: one image per venue, all available years for that venue\n"
        "- `venue_year_family_composition.csv`: complete family-level composition\n"
        "- `venue_year_topic_composition_full.csv`: complete topic-level composition\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    topics, top10, summary, family = load_data(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for folder in (args.output_dir / "by_year", args.output_dir / "by_venue"):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.png"):
            old.unlink()
    for old in args.output_dir.glob("*.png"):
        old.unlink()

    write_tables(args.output_dir, topics, family)
    draw_overview(summary, args.output_dir)
    for year in sorted(summary["year"].unique()):
        draw_year_page(int(year), family, top10, summary, args.output_dir)
    for venue in [v for v in VENUE_ORDER if v in set(summary["venue"])]:
        draw_venue_page(venue, family, top10, summary, args.output_dir)
    write_readme(args.output_dir)
    print(f"Wrote composition atlas to {args.output_dir}")


if __name__ == "__main__":
    main()
