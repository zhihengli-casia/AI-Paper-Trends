<div align="right"><a href="README.md">English</a> | <strong>中文</strong></div>

# AI 学术会议热点分析框架

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## AI 论文数据库索引

**从这里开始浏览：[AI Paper Topic Atlas](docs/topic-atlas/README.md)**

数据库按照 **年份 -> 会议 -> 主题 -> 论文** 组织。目前覆盖 **117,100 篇主会接收论文**、**84 个会议-年份单元**、**5,183 个细粒度主题**。

快速入口：

| 入口 | 链接 |
|---|---|
| 总图谱 | [docs/topic-atlas/README.md](docs/topic-atlas/README.md) |
| 2026 年论文 | [docs/topic-atlas/2026/README.md](docs/topic-atlas/2026/README.md) |
| ICLR 2026 主题列表 | [docs/topic-atlas/2026/ICLR/README.md](docs/topic-atlas/2026/ICLR/README.md) |
| 示例主题 | [ICLR 2026：可验证奖励驱动的大模型推理](docs/topic-atlas/2026/ICLR/topic-004.md) |
| 主题 CSV 索引 | [docs/topic-atlas/data/topic_index.csv](docs/topic-atlas/data/topic_index.csv) |
| 会议-年份 CSV 汇总 | [docs/topic-atlas/data/venue_year_summary.csv](docs/topic-atlas/data/venue_year_summary.csv) |

示例路径：

`2026 -> ICLR -> 可验证奖励驱动的大模型推理 -> Reinforcement Learning with Verifiable Rewards...`

本框架提供自动化、可配置的流程，用于从顶级AI会议（如 ICLR, NeurIPS, ICML）论文中挖掘研究热点。通过配置文件驱动，可执行数据获取、主题建模与结果可视化。

## 📋 目录

- [AI 学术会议热点分析框架](#ai-学术会议热点分析框架)
  - [AI 论文数据库索引](#ai-论文数据库索引)
  - [📋 目录](#-目录)
  - [📂 项目结构](#-项目结构)
  - [🚀 快速上手](#-快速上手)
    - [1. 环境配置](#1-环境配置)
    - [2. 配置分析任务](#2-配置分析任务)
    - [3. 运行自动化流程](#3-运行自动化流程)
  - [🔬 探索性分析 (Jupyter Notebooks)](#-探索性分析-jupyter-notebooks)
  - [💡 高级选项](#-高级选项)
    - [分析不同会议](#分析不同会议)
  - [🤝 参与贡献](#-参与贡献)
  - [📄 许可证](#-许可证)

## 📂 项目结构

```text
.
├── configs/              # 分析任务的 YAML 配置文件
├── data/                 # (Git 忽略) 存放原始数据 (.jsonl) 和处理后数据 (.csv)
├── docs/                 # 文档及相关资源 (如 README 图片)
├── LICENSE               # 项目许可证
├── main.py               # 项目主入口脚本 (运行分析)
├── models/               # (Git 忽略) 存放下载的机器学习模型
├── notebooks/            # Jupyter Notebooks (教程, 探索性分析)
├── README_cn.md          # 项目中文介绍 (本文件)
├── README.md             # 项目英文介绍
├── requirements.txt      # Python 依赖列表
├── results/              # (Git 忽略) 存放分析结果 (图表, 表格, 模型)
├── src/                  # 核心 Python 功能模块
│   ├── analyze.py        # 分析与可视化逻辑
│   ├── get_papers.py     # 数据获取逻辑
│   ├── run_topic_modeling.py # 主题建模逻辑
│   └── utils.py          # (可选) 通用辅助函数
└── .gitignore            # 指定 Git 忽略的文件/目录
````

## 🚀 快速上手

### 1\. 环境配置

推荐使用 Conda 创建环境，并通过 `pip` 安装依赖。

```bash
# 克隆仓库
git clone [https://github.com/zhihengli-casia/AI-Paper-Trends.git](https://github.com/zhihengli-casia/AI-Paper-Trends.git)
cd AI-Paper-Trends

# 1. 创建 Conda 环境 (推荐 Python 3.10)
conda create --name ai-trend-analysis python=3.10

# 2. 激活环境
conda activate ai-trend-analysis

# 3. 安装依赖
pip install -r requirements.txt
```

### 2\. 配置分析任务

分析流程由 `configs/` 目录下的 `.yaml` 文件定义。

1.  进入 `configs/` 目录。
2.  复制现有 `.yaml` 文件或新建一个。
3.  修改文件参数以指定分析目标。

**示例 (`configs/iclr_2025_analysis.yaml`):**

```yaml
conference_id: 'ICLR.cc/2025/Conference' # 目标会议 ID
fetch_reviews: True                      # 是否获取审稿信息
limit: null                              # 处理论文数量上限 (null=无限制)

topic_modeling:
  enabled: True                          # 是否执行主题建模
  min_topic_size: 30                     # BERTopic 最小主题规模

analysis:
  enabled: True                          # 是否执行分析与可视化
  tasks:                                 # 要执行的分析任务列表
    - 'plot_paper_count'                 #   - 论文数排序图
    - 'plot_avg_rating'                  #   - 平均分排序图
    - 'plot_decision_breakdown'          #   - 决策构成图
    - 'generate_summary_table'           #   - 生成统计表格

output_folder_name: 'iclr_2025_analysis' # results/ 下的输出目录名
```

### 3\. 运行自动化流程

在项目根目录执行 `main.py`，指定配置文件。

```bash
python main.py --config configs/iclr_2025_analysis.yaml
```

脚本将按配置执行数据获取、主题建模及结果生成。产出位于 `data/` 和 `results/` 目录。

## 🔬 探索性分析 (Jupyter Notebooks)

`notebooks/` 目录提供 Jupyter 环境，用于进行更深入或定制化的探索性分析。

**使用流程**:

1.  确保已激活 Conda 环境: `conda activate ai-trend-analysis`
2.  在项目根目录启动 Jupyter Lab: `jupyter lab`
3.  在浏览器中打开 `notebooks/` 下的 `.ipynb` 文件。

## 💡 高级选项

### 分析不同会议

修改配置文件中的 `conference_id`。常见 ID 示例：

  * **ICLR**: `ICLR.cc/2025/Conference`
  * **NeurIPS**: `NeurIPS.cc/2023/Conference`
  * **ICML**: `ICML.cc/2024/Conference`

> **建议**: 在 [OpenReview](https://openreview.net/) 官网确认目标会议的准确 ID。


## 🤝 参与贡献

欢迎通过提交 [Issues](https://github.com/zhihengli-casia/AI-Paper-Trends/issues) 或 [Pull Requests](https://github.com/zhihengli-casia/AI-Paper-Trends/pulls) 来报告问题、提出建议或贡献代码。

## 📄 许可证

本项目基于 [MIT 许可证](https://www.google.com/search?q=LICENSE) 发布。
