<div align="right"><a href="README.md">English</a> | <strong>中文</strong></div>

# AI Paper Trends

面向 AI 顶会论文的公开元数据爬取、主会 accepted 口径清洗、主题聚类和趋势分析工具。

这个仓库最早用于 ICLR 2025 投稿论文热点分析。现在已扩展为覆盖多个 AI 顶会的趋势分析框架，支持按“会议-年份”独立聚类，避免把不同会议、不同年份的主题混在一起。

它的目标不是再做一个 paper list，而是做一个可复现的 **AI 顶会主题组成图谱**：回答每一年、每个会议到底由哪些研究主题组成，以及这些主题如何随时间变化。

## 当前全量分析

最新一次分析使用 2020-2026 年主会 accepted 论文，目标是给出“每一年、每个会议”的完整主题组成，而不是只列几个热门方向。

| 指标 | 数值 |
|---|---:|
| 覆盖会议 | 15 |
| 会议-年份组 | 84 |
| 进入聚类论文 | 117,100 |
| 会议-年份细主题 | 763 |
| 图谱中的主题大类 | 15 |
| 最终离群论文 | 0 |
| 年份范围 | 2020-2026 |
| 2026 已纳入口径 | ICLR accepted, AAAI Technical Tracks accepted |

聚类口径：

- 论文范围：主会 accepted paper；NLP 不含 Findings / Industry 等非主会 track。
- 聚类单位：每个会议的每一年单独聚类，例如 `CVPR 2025`、`ICLR 2026` 分开跑。
- 主题算法：`BGE embedding -> kNN cosine graph -> Louvain community detection -> 小社区合并 -> 中文规则命名`。
- 2026 数据：目前只纳入已能合法确认的 `ICLR 2026 accepted` 和 `AAAI 2026 Technical Tracks accepted`。

已提交的轻量结果文件：

| 文件 | 行数 | 内容 |
|---|---:|---|
| [REPORT_CN.md](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/REPORT_CN.md) | - | 完整中文报告 |
| [run_summary_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/run_summary_by_venue_year.csv) | 84 | 每个会议-年份的论文数、图规模、主题数、离群率 |
| [top10_topics_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/top10_topics_by_venue_year.csv) | 700 | 每个会议-年份的 Top 10 主题 |
| [topic_summary_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/topic_summary_by_venue_year.csv) | 763 | 全部细主题，含命名、关键词、篇数、占比、代表论文标题 |
| [label_trend_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/label_trend_by_venue_year.csv) | 763 | 跨会议、跨年份的主题趋势表 |
| [venue_year_family_composition.csv](docs/visuals/xhs_composition_atlas_2020_2026/venue_year_family_composition.csv) | 473 | 每个会议-年份的主题大类组成 |
| [venue_year_topic_composition_full.csv](docs/visuals/xhs_composition_atlas_2020_2026/venue_year_topic_composition_full.csv) | 763 | 每个会议-年份的完整细主题组成 |

没有把完整 per-paper 结果和原始抓取数据提交到仓库，因为体积较大；这些文件应在本地 `data/` 和 `results/` 下生成。

## 主题组成图谱

为了让结果更适合阅读和传播，仓库内提交了一套轻量级组成图谱：

![AI Paper Trends 主题组成图谱](docs/visuals/xhs_composition_atlas_2020_2026/00_composition_atlas_cover.png)

- 图谱入口：[docs/visuals/xhs_composition_atlas_2020_2026/00_composition_atlas_cover.png](docs/visuals/xhs_composition_atlas_2020_2026/00_composition_atlas_cover.png)
- 按年份查看所有会议组成：[docs/visuals/xhs_composition_atlas_2020_2026/by_year/](docs/visuals/xhs_composition_atlas_2020_2026/by_year/)
- 按会议查看历年组成：[docs/visuals/xhs_composition_atlas_2020_2026/by_venue/](docs/visuals/xhs_composition_atlas_2020_2026/by_venue/)

每条横向组成条都代表某个会议某一年的全部论文，颜色表示主题大类，长度表示占比。右侧文字只展示主要细主题；精确到每个细主题的完整构成在 CSV 中。

## 按年份覆盖

| 年份 | 会议-年份组 | 论文数 | 主题数 |
|---:|---:|---:|---:|
| 2020 | 13 | 11,555 | 106 |
| 2021 | 14 | 13,334 | 117 |
| 2022 | 14 | 14,333 | 117 |
| 2023 | 13 | 17,424 | 120 |
| 2024 | 14 | 22,334 | 132 |
| 2025 | 14 | 28,619 | 144 |
| 2026 | 2 | 9,501 | 27 |

## 按领域覆盖

| 领域 | 会议 | 会议-年份组 | 论文数 | 主题数 |
|---|---|---:|---:|---:|
| 计算机视觉 | CVPR, ICCV, ECCV | 12 | 24,999 | 132 |
| 机器学习 | ICLR, ICML, NeurIPS | 19 | 46,238 | 214 |
| NLP | ACL, EMNLP, NAACL | 16 | 14,651 | 147 |
| 综合 AI | AAAI, IJCAI | 13 | 21,179 | 121 |
| 多媒体 | ACMMM | 6 | 5,006 | 57 |
| 数据挖掘 / 检索 / Web | KDD, SIGIR, WWW | 18 | 5,027 | 92 |

## 按会议覆盖

| 领域 | 会议 | 年份 | 会议-年份组 | 论文数 | 主题数 | 最新年份 Top1 主题 |
|---|---|---:|---:|---:|---:|---|
| 综合 AI | AAAI | 2020-2026 | 7 | 15,638 | 73 | 2026：三维视觉：Gaussian Splatting、新视角合成与重建（610, 14.70%） |
| 综合 AI | IJCAI | 2020-2025 | 6 | 5,541 | 48 | 2025：多模态理解：视觉语言表征与跨模态对齐（277, 21.64%） |
| 机器学习 | ICLR | 2020-2026 | 7 | 15,529 | 71 | 2026：生成模型：视频扩散生成与编辑（752, 14.05%） |
| 机器学习 | ICML | 2020-2025 | 6 | 11,268 | 64 | 2025：高效大模型：推理加速、压缩与资源优化（446, 13.39%） |
| 机器学习 | NeurIPS | 2020-2025 | 6 | 19,441 | 79 | 2025：大模型推理：RL驱动推理与奖励学习（781, 14.77%） |
| 计算机视觉 | CVPR | 2020-2025 | 6 | 13,140 | 68 | 2025：生成模型：文生图、扩散采样与图像编辑（458, 15.95%） |
| 计算机视觉 | ICCV | 2021-2025 | 3 | 6,469 | 36 | 2025：多模态大模型：视觉语言理解与跨模态推理（460, 17.03%） |
| 计算机视觉 | ECCV | 2020-2024 | 3 | 5,390 | 28 | 2024：开放词汇视觉：开放词汇检测、分割与CLIP语义（373, 15.63%） |
| NLP | ACL | 2020-2025 | 6 | 5,902 | 60 | 2025：高效大模型：长上下文、注意力与推理优化（308, 18.13%） |
| NLP | EMNLP | 2020-2025 | 6 | 6,550 | 56 | 2025：检索增强大模型：RAG、知识注入与问答（254, 14.04%） |
| NLP | NAACL | 2021-2025 | 4 | 2,199 | 31 | 2025：大模型社会安全：偏见、虚假信息与检测（126, 17.55%） |
| 多媒体 | ACMMM | 2020-2025 | 6 | 5,006 | 57 | 2025：多媒体检索：跨模态检索、语义匹配与内容理解（197, 15.77%） |
| 数据挖掘 | KDD | 2020-2025 | 6 | 1,985 | 36 | 2025：图基础模型：LLM增强图学习与节点表示（122, 22.10%） |
| 信息检索 | SIGIR | 2020-2025 | 6 | 1,077 | 23 | 2025：推荐系统：偏好建模、反馈学习与个性化排序（85, 35.56%） |
| Web / 推荐系统 | WWW | 2020-2025 | 6 | 1,965 | 33 | 2025：推荐系统：检索增强推荐、排序与个性化（40, 25.97%） |

## 最新年份主题示例

### ICLR 2026

| 排名 | 主题 | 篇数 | 占比 |
|---:|---|---:|---:|
| 1 | 生成模型：视频扩散生成与编辑 | 752 | 14.05% |
| 2 | 高效大模型：推理加速、压缩与资源优化 | 691 | 12.91% |
| 3 | 大模型推理：RL驱动推理与奖励学习 | 655 | 12.24% |
| 4 | 优化理论：随机/非凸优化与收敛率 | 645 | 12.05% |
| 5 | 多模态大模型：视觉语言理解与跨模态推理 | 625 | 11.68% |

### AAAI 2026

| 排名 | 主题 | 篇数 | 占比 |
|---:|---|---:|---:|
| 1 | 三维视觉：Gaussian Splatting、新视角合成与重建 | 610 | 14.70% |
| 2 | 大模型推理：问答、常识与思维链 | 596 | 14.36% |
| 3 | 强化学习：策略优化、奖励学习与控制 | 463 | 11.16% |
| 4 | 图学习：图异常检测、聚类与结构表示 | 391 | 9.42% |
| 5 | 高效大模型：推理加速、压缩与资源优化 | 344 | 8.29% |

### NeurIPS 2025

| 排名 | 主题 | 篇数 | 占比 |
|---:|---|---:|---:|
| 1 | 大模型推理：RL驱动推理与奖励学习 | 781 | 14.77% |
| 2 | 生成模型：视频扩散生成与编辑 | 583 | 11.03% |
| 3 | 多模态大模型：视觉语言理解与跨模态推理 | 575 | 10.88% |
| 4 | 在线决策：Bandit、后悔界与探索 | 570 | 10.78% |
| 5 | 优化理论：随机/非凸优化与收敛率 | 548 | 10.37% |

### CVPR 2025

| 排名 | 主题 | 篇数 | 占比 |
|---:|---|---:|---:|
| 1 | 生成模型：文生图、扩散采样与图像编辑 | 458 | 15.95% |
| 2 | 三维视觉：点云、深度估计与相机姿态 | 413 | 14.39% |
| 3 | 多模态大模型：视觉语言理解与跨模态推理（视觉语言） | 349 | 12.16% |
| 4 | 生成模型：视频扩散生成与编辑 | 276 | 9.61% |
| 5 | 具身智能：机器人操作、导航与视觉语言动作 | 253 | 8.81% |

## 快速上手

```bash
git clone https://github.com/zhihengli-casia/AI-Paper-Trends.git
cd AI-Paper-Trends
conda create -n ai-paper-trends python=3.10
conda activate ai-paper-trends
pip install -r requirements.txt
```

### 单会 OpenReview 分析（legacy）

旧版 ICLR 流程仍然保留，适合分析单个 OpenReview 会议并获取审稿分数/decision。

```bash
python main.py --config configs/iclr_2025_full_analysis.yaml
```

### 多会议 accepted-paper 趋势分析

下面是完整流程的核心步骤。实际大规模运行建议放到 `tmux` 中。

```bash
# 1. 抓取近年公开会议元数据
python src/crawl_recent_conferences.py \
  --output-dir data/recent_conferences \
  --run-name recent_ai_conference_papers \
  --fetch-detail-abstracts

# 2. 构建严格主会 accepted 口径
python src/build_main_accepted_view.py \
  --input data/recent_conferences/recent_ai_conference_papers.jsonl \
  --output data/recent_conferences/main_accepted_ai_conference_papers.jsonl

# 3. 按会议-年份运行 kNN + Louvain 主题聚类
python src/analyze_venue_year_louvain_topics.py \
  --input data/recent_conferences/main_accepted_ai_conference_papers.jsonl \
  --output-dir results/venue_year_main_accepted_topics_louvain \
  --model-name models/bge-base-en-v1.5 \
  --device mps
```

如果只修改主题命名规则，不需要重跑 embedding 和聚类：

```bash
python src/rebuild_venue_year_outputs.py \
  --results-dir results/venue_year_main_accepted_topics_louvain \
  --top-k 10
```

生成主题组成图谱：

```bash
python src/create_xhs_composition_atlas.py \
  --results-dir docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25 \
  --output-dir docs/visuals/xhs_composition_atlas_2020_2026
```

## 项目结构

```text
.
├── configs/                     # 单会 OpenReview 分析配置
├── data/                        # 本地原始数据和处理中间文件，默认 Git 忽略
├── docs/results/                # 可提交的轻量分析结果
├── docs/visuals/                # 主题组成图谱和传播用图片
├── main.py                      # legacy 单会 OpenReview 分析入口
├── notebooks/                   # Notebook 示例
├── src/
│   ├── crawl_recent_conferences.py          # 多会议基础爬虫
│   ├── crawl_2020_2025_backfill.py          # 2020-2025 补全爬虫
│   ├── build_main_accepted_view.py          # 主会 accepted 口径过滤
│   ├── analyze_venue_year_louvain_topics.py # kNN + Louvain 聚类
│   ├── create_xhs_composition_atlas.py      # 会议-年份主题组成图谱
│   ├── refine_topic_names.py                # 中文主题命名规则
│   ├── rebuild_venue_year_outputs.py        # 不重跑聚类的结果重建
│   └── ...                                  # legacy BERTopic / OpenReview 模块
└── requirements.txt
```

## 说明

- 本仓库只使用公开论文元数据；不同会议公开字段不同，所以 review score / rejected submissions 并非所有会议都有。
- 多会议趋势分析默认使用 accepted-only 口径，适合比较正式发表研究热点。
- 旧版 ICLR 分析可以覆盖投稿、拒稿、评分等 OpenReview 信息，但不适合直接和没有公开审稿数据的会议混合比较。

## License

MIT License.
