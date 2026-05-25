<div align="right"><a href="README.md">English</a> | <strong>中文</strong></div>

# AI Paper Trends

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI Paper Trends 是一个开源的 AI 论文与研究主题索引，覆盖主要 AI 会议与部分期刊。它把论文整理成一个可以静态浏览的图谱：

**年份 -> venue -> 主题 -> 论文**

这个项目解决的是一个很具体的问题：想看某一年、某个会议到底在发哪些方向，不用在一堆 proceedings 页面里来回翻。

## 当前快照

当前仓库中已提交的静态图谱包含：

| 指标 | 数值 |
|---|---:|
| 论文 | 155,662 |
| venue-year 单元 | 160 |
| 细粒度主题页 | 7,378 |
| 年份 | 2020-2026 |
| 回填后未分配论文 | 25 |

这是一个持续更新中的索引。新的 venue-year 会随着会议论文集和公开元数据释放逐步加入，因此最新年份在会议周期中可能是阶段性覆盖。

从这里开始浏览：[AI Paper Topic Atlas](docs/topic-atlas/README.md)

## 可以用它做什么

- 按年份、会议/期刊、细粒度主题和单篇论文浏览文献。
- 比较不同顶会、期刊和年份的研究方向变化。
- 使用 CSV 索引做二次分析和可视化。
- 基于缓存元数据和主题结果重新生成静态图谱。
- 通过自动更新配置继续扩展新的会议、期刊和年份。

## 覆盖范围

下面的数字对应当前仓库快照，会作为持续更新视图维护。随着更多会议、年份和论文集加入，这些数字会继续变化。

### 会议

| 领域 | 已收录 venue | 论文 | 细粒度主题 |
|---|---|---:|---:|
| 机器学习三大会 / 学习理论 | ICLR, ICML, NeurIPS | 46,161 | 1,724 |
| CV 三大会 | CVPR, ICCV, ECCV | 24,999 | 1,130 |
| NLP / 语言会议 | ACL, EMNLP, NAACL, COLM | 15,368 | 811 |
| 综合 AI | AAAI, IJCAI | 21,179 | 963 |
| 具身智能 / 机器人核心会议 | ICRA, IROS, RSS | 16,590 | 873 |
| 多媒体 / 图形学 / HCI | ACMMM, SIGGRAPH, SIGGRAPH-Asia, CHI | 10,473 | 591 |
| 数据挖掘 / 检索 / Web / 数据库 | KDD, SIGIR, WWW, ICDE, SIGMOD | 7,601 | 482 |
| 医疗 AI | MICCAI | 428 | 41 |

### 期刊

| 领域 | 已收录期刊 | 论文 | 细粒度主题 |
|---|---|---:|---:|
| 机器学习 / 综合 AI 期刊 | AIJ, JMLR, TNNLS | 2,768 | 183 |
| 视觉 / 图像期刊 | TPAMI, IJCV, TIP, PR | 8,009 | 456 |
| 多媒体 / 数据期刊 | TMM, TKDE | 2,086 | 124 |

### 待补充来源

候选来源会持续跟踪，实际纳入取决于公开元数据可得性和数据质量。

| 领域 | 候选会议与期刊 |
|---|---|
| 机器学习 / AI | AISTATS, UAI, COLT, JAIR, Machine Learning |
| 视觉 / 图形学 | WACV, BMVC, ACCV, 3DV, TVCG |
| NLP / 语音 | EACL, TACL, Computational Linguistics, Interspeech |
| 机器人 / 具身智能 | CoRL, RA-L, T-RO, IJRR, Autonomous Robots |
| 数据 / 检索 / Web | CIKM, WSDM, RecSys, ICDM, SDM, VLDB, EDBT, PODS |
| HCI / 系统 | UIST, CSCW, IMWUT, UbiComp |
| 医疗 AI | TMI, Medical Image Analysis, ISBI |

## 浏览入口

| 入口 | 链接 |
|---|---|
| 图谱首页 | [docs/topic-atlas/README.md](docs/topic-atlas/README.md) |
| 最新收录年份 | [docs/topic-atlas/2026/README.md](docs/topic-atlas/2026/README.md) |
| 主题 CSV 索引 | [docs/topic-atlas/data/topic_index.csv](docs/topic-atlas/data/topic_index.csv) |
| 会议-年份 CSV 汇总 | [docs/topic-atlas/data/venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) |
| 自动更新状态 | [docs/auto-update/status.md](docs/auto-update/status.md) |

每个主题页包含：

- 中文展示主题名，
- 可复现的英文关键词标签，
- 代表论文，
- 论文级外链，例如 OpenReview、DOI、Semantic Scholar 或来源 URL。

## 数据文件

| 文件 | 说明 |
|---|---|
| [docs/topic-atlas/data/topic_index.csv](docs/topic-atlas/data/topic_index.csv) | 主题级索引，包含年份、venue、topic ID、主题标签、论文数量和代表论文。 |
| [docs/topic-atlas/data/venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) | venue-year 覆盖汇总，包含论文数、主题数和聚类诊断指标。 |
| [docs/clean-2020-plus/README.md](docs/clean-2020-plus/README.md) | 2020-2026 会议论文干净版主题快照。 |
| [docs/auto-update/status.md](docs/auto-update/status.md) | 最近一次轻量自动更新检查报告。 |

大体积原始元数据、embedding 和中间聚类结果不会直接提交到仓库中，相关目录由 Git 忽略或存放在仓库外部。

## 数据口径与质量说明

- 公开图谱聚焦可公开访问元数据的论文。审稿分、拒稿论文和非公开审稿讨论不会纳入，除非会议本身公开发布这些信息。
- 不同会议、期刊和年份的覆盖完整度会受数据源开放程度影响。
- 主题名由论文元数据自动生成，并经过清洗以便浏览；它们是导航标签，不是固定学科分类。
- 论文链接会尽量指向公开页面，包括 OpenReview、DOI、Semantic Scholar、出版商页面或来源元数据 URL。

## 方法

当前图谱采用按 venue-year 独立聚类的方式生成，也就是每个会议或期刊的每一年都有自己的主题结构。

流程概览：

1. 从 OpenReview、DBLP、OpenAlex、会议论文集页面和出版商元数据等公开来源收集论文元数据。
2. 规范化 venue-year 记录，仓库展示口径聚焦公开的已发表、已接收或可公开访问的论文元数据。
3. 基于论文标题和摘要生成并缓存 BGE embedding。
4. 使用 UMAP 降维。
5. 使用 HDBSCAN leaf clustering 生成细粒度主题。
6. 对 HDBSCAN 离群点做最近主题质心回填。
7. 使用 c-TF-IDF 风格统计提取可复现的英文关键词标签。
8. 生成启发式中文展示主题名，并在同一个 venue-year 内消解重名主题。
9. 生成可浏览的 Markdown 静态图谱和可分析的 CSV 索引。

主题名主要用来帮助读者浏览图谱。每个主题页都会保留代表论文和关键词池，方便后续检查和修正。

## 重新生成

安装依赖：

```bash
pip install -r requirements.txt
```

对缓存好的 venue-year embedding 做细粒度聚类：

```bash
python scripts/fine_grained_topic_analysis.py \
  --input-root results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25 \
  --output-root results/fine_grained_venue_year_topics_2020_2026_mcs_fine
```

生成静态图谱：

```bash
python scripts/build_topic_atlas.py \
  --topic-root results/fine_grained_venue_year_topics_2020_2026_mcs_fine \
  --output-root docs/topic-atlas \
  --clean
```

## 自动更新

仓库包含 GitHub Actions 自动更新引擎：

| 模式 | 用途 | 备注 |
|---|---|---|
| 每周轻量检查 | 检查哪些 venue-year 可能已经有新的公开元数据。 | 在 GitHub 托管 runner 上运行，并更新 [docs/auto-update/status.md](docs/auto-update/status.md)。 |
| 完整图谱刷新 | 重建 `docs/topic-atlas`。 | 需要带有 `ai-paper-trends` 标签的 self-hosted runner，以及外部 embedding/result 缓存。 |

本地运行轻量检查：

```bash
python scripts/auto_update_atlas.py check --write-report
```

在有 `results/` 缓存的机器上完整刷新：

```bash
python scripts/auto_update_atlas.py refresh
```

会议时间表和数据源备注在 [configs/auto_update.yaml](configs/auto_update.yaml) 中维护。

## 项目结构

| 路径 | 用途 |
|---|---|
| [docs/topic-atlas/](docs/topic-atlas/README.md) | 可浏览的静态论文主题图谱。 |
| [docs/clean-2020-plus/](docs/clean-2020-plus/README.md) | 2020-2026 会议论文主题快照。 |
| [scripts/build_topic_atlas.py](scripts/build_topic_atlas.py) | 静态图谱生成脚本。 |
| [scripts/fine_grained_topic_analysis.py](scripts/fine_grained_topic_analysis.py) | 按 venue-year 的细粒度主题聚类脚本。 |
| [configs/auto_update.yaml](configs/auto_update.yaml) | venue 更新时间表和来源配置。 |
| [scripts/auto_update_atlas.py](scripts/auto_update_atlas.py) | 轻量检查和完整刷新入口。 |
| [main.py](main.py), [src/](src), [configs/](configs) | 原始单会议 OpenReview 分析流程。 |

## 单会议 OpenReview 流程

仓库仍保留早期 OpenReview + BERTopic 流程，适合做单个会议的审稿分分析、接收类型分析或 OpenReview-only 小实验。

创建环境：

```bash
conda create --name ai-trend-analysis python=3.10
conda activate ai-trend-analysis
pip install -r requirements.txt
```

运行一个配置好的 OpenReview 任务：

```bash
python main.py --config configs/iclr_2025_full_analysis.yaml
```

如果需要审稿分图、接收类型分析或快速测试单个 OpenReview 会议，可以使用这条流程。多会议数据库请使用 atlas 流程。

## 参与贡献

欢迎贡献：

- 修正 venue-year 元数据，
- 增加新的公开数据源，
- 改进主题命名，
- 检查代表论文，
- 改进自动更新时间表，
- 增加可视化 notebook 或分析示例。

可以通过 [Issues](https://github.com/zhihengli-casia/AI-Paper-Trends/issues) 或 [Pull Requests](https://github.com/zhihengli-casia/AI-Paper-Trends/pulls) 参与。

## 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。
