<div align="right"><a href="README.md">English</a> | <strong>中文</strong></div>

# AI Paper Trends

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## AI 论文数据库索引

**从这里开始浏览：[AI Paper Topic Atlas](docs/topic-atlas/README.md)**

数据库按照 **年份 -> venue -> 主题 -> 论文** 组织。目前覆盖 **155,662 篇会议与期刊论文**、**160 个 venue-year 单元**、**7,378 个细粒度主题**。

快速入口：

| 入口 | 链接 |
|---|---|
| 总图谱 | [docs/topic-atlas/README.md](docs/topic-atlas/README.md) |
| 2020-2026 会议主题快照 | [docs/clean-2020-plus/README.md](docs/clean-2020-plus/README.md) |
| 2026 年论文 | [docs/topic-atlas/2026/README.md](docs/topic-atlas/2026/README.md) |
| ICLR 2026 主题列表 | [docs/topic-atlas/2026/ICLR/README.md](docs/topic-atlas/2026/ICLR/README.md) |
| 示例主题 | [ICLR 2026：可验证奖励驱动的大模型推理](docs/topic-atlas/2026/ICLR/topic-004.md) |
| 主题 CSV 索引 | [docs/topic-atlas/data/topic_index.csv](docs/topic-atlas/data/topic_index.csv) |
| 会议-年份 CSV 汇总 | [docs/topic-atlas/data/venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) |

示例路径：

`2026 -> ICLR -> 可验证奖励驱动的大模型推理 -> Reinforcement Learning with Verifiable Rewards...`

本仓库提供覆盖主要会议与部分期刊的 AI 论文数据库和细粒度主题图谱，同时保留单会议 OpenReview 分析流程，便于进行会议级实验。

## 2020-2026 会议主题快照

[2020-2026 会议主题快照](docs/clean-2020-plus/README.md) 汇总 conference / main accepted / published 口径下的会议论文主题分布，提供逐年、逐会议、逐会议-年份和逐主题的聚合表。

快照覆盖范围：

| 指标 | 数值 |
|---|---:|
| 已聚类会议论文 | 142,799 |
| 会议 | 25 |
| 会议-年份单元 | 132 |
| 会议-年份主题总数 | 1,307 |
| 年份 | 2020-2026 |
| 最终离群点 | 0 |

CSV 表格包含每个会议-年份的完整主题汇总、每个会议最新年份的主题列表、逐年覆盖和逐会覆盖。

## 包含内容

| 内容 | 文件 |
|---|---|
| 可浏览论文数据库 | [docs/topic-atlas/](docs/topic-atlas/README.md) |
| 2020+ 会议主题快照 | [docs/clean-2020-plus/](docs/clean-2020-plus/README.md) |
| 主题索引数据 | [topic_index.csv](docs/topic-atlas/data/topic_index.csv), [venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) |
| 图谱生成脚本 | [scripts/build_topic_atlas.py](scripts/build_topic_atlas.py) |
| 细粒度聚类脚本 | [scripts/fine_grained_topic_analysis.py](scripts/fine_grained_topic_analysis.py) |
| 自动更新引擎 | [configs/auto_update.yaml](configs/auto_update.yaml), [scripts/auto_update_atlas.py](scripts/auto_update_atlas.py), [docs/auto-update/status.md](docs/auto-update/status.md) |
| 单会议 OpenReview 流程 | [main.py](main.py), [src/](src), [configs/](configs) |

## 🚀 浏览数据库

这个仓库最主要的使用方式是直接浏览静态主题图谱：

- [图谱总入口](docs/topic-atlas/README.md)
- [2026 年会议列表](docs/topic-atlas/2026/README.md)
- [ICLR 2026 主题列表](docs/topic-atlas/2026/ICLR/README.md)
- [示例主题页](docs/topic-atlas/2026/ICLR/topic-004.md)

每个主题页包含：

- 中文展示主题名，
- 可复现的英文关键词标签，
- 代表论文，
- 论文级外链，例如 OpenReview、DOI、Semantic Scholar 或来源 URL。

如果想自己做可视化，可以直接使用 CSV 索引：

- [topic_index.csv](docs/topic-atlas/data/topic_index.csv)
- [venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv)

## 🔁 重新生成主题图谱

主题图谱由缓存论文元数据和 embedding 生成。大体积 embedding 缓存存放在仓库外部。

安装依赖：

```bash
pip install -r requirements.txt
```

对缓存好的会议-年份 embedding 做细粒度聚类：

```bash
python scripts/fine_grained_topic_analysis.py \
  --input-root results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25 \
  --output-root results/fine_grained_venue_year_topics_2020_2026_mcs_fine
```

生成静态主题图谱：

```bash
python scripts/build_topic_atlas.py \
  --topic-root results/fine_grained_venue_year_topics_2020_2026_mcs_fine \
  --output-root docs/topic-atlas \
  --clean
```

当前图谱的生成逻辑是：

- 复用 BGE embedding 缓存，
- 按会议-年份独立聚类，
- UMAP 降维，
- HDBSCAN leaf 细聚类，
- 对 HDBSCAN 离群点做 centroid 回填，
- 用 c-TF-IDF 风格关键词提取生成英文标签，
- 用启发式中文命名辅助浏览。

## 🔄 自动更新

仓库包含一套 GitHub Actions 自动更新引擎：

- 每周轻量检查：在 GitHub 托管 runner 上检测哪些会议-年份卷宗已经到期或需要继续观察，并更新 [docs/auto-update/status.md](docs/auto-update/status.md)。
- 完整刷新：在带有 `ai-paper-trends` 标签的 self-hosted runner 上重建 `docs/topic-atlas`。该模式需要外部 embedding/result 缓存。把仓库变量 `AUTO_REFRESH_ATLAS` 设为 `true` 后，它会跟随每周定时任务运行。

本地运行轻量检查：

```bash
python scripts/auto_update_atlas.py check --write-report
```

在有 `results/` 缓存的机器上完整刷新：

```bash
python scripts/auto_update_atlas.py refresh
```

会议时间表和数据源备注在 [configs/auto_update.yaml](configs/auto_update.yaml) 中维护。

## 🧪 单会议流程

OpenReview + BERTopic 流程适合小规模单会议实验，可用于审稿分图、接收类型图和 OpenReview-only 分析。

### 1. 环境配置

```bash
conda create --name ai-trend-analysis python=3.10
conda activate ai-trend-analysis
pip install -r requirements.txt
```

### 2. 运行 OpenReview 单会议任务

在 `configs/` 中配置任务，然后运行：

```bash
python main.py --config configs/iclr_2025_full_analysis.yaml
```

如果需要审稿分图、接收类型图，或者快速测试某个 OpenReview 会议，可以使用这条旧流程。多会议数据库请使用上面的 atlas 脚本。

## 🤝 参与贡献

欢迎通过提交 [Issues](https://github.com/zhihengli-casia/AI-Paper-Trends/issues) 或 [Pull Requests](https://github.com/zhihengli-casia/AI-Paper-Trends/pulls) 来报告问题、提出建议或贡献代码。

## 📄 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。
