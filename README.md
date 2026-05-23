<div align="right"><strong>English</strong> | <a href="README_cn.md">中文</a></div>

# AI Paper Trends

Tools for crawling public AI conference paper metadata, normalizing main-conference accepted papers, clustering research topics, and tracking topic trends over time.

The project started as an ICLR 2025 OpenReview analysis pipeline. It has now been expanded into a multi-conference trend-analysis workflow that clusters papers independently for each venue-year, so topic distributions are comparable across conferences and years.

This is not just a paper list. The goal is to provide a reproducible **topic composition atlas** for major AI conferences: what each venue-year is made of, and how those research themes evolve over time.

<!-- AI-PAPER-TRENDS:START -->
## Current Analysis

The latest committed run is an accepted-paper analysis for major AI venues from 2020 to 2026.

| Metric | Value |
|---|---:|
| Venues | 15 |
| Venue-year groups | 84 |
| Papers used for clustering | 117,100 |
| Venue-year topics | 763 |
| Broad topic families in the atlas | 11 |
| Final outlier papers | 0 |
| Years covered | 2020-2026 |
| 2026 accepted-paper data currently included | ICLR, AAAI |

Method:

`BGE embeddings -> cosine kNN graph -> Louvain community detection -> small-community merge -> deterministic Chinese topic naming`

Scope:

- Accepted main-conference papers only.
- NLP venues exclude Findings, Industry, SRW, and other non-main tracks.
- 2026 currently includes legally confirmed `ICLR 2026 accepted` and `AAAI 2026 Technical Tracks accepted`.
- Large raw crawls, per-paper topic assignments, embedding files, and model caches are intentionally not committed.

Committed result tables:

| File | Rows | What it contains |
|---|---:|---|
| [REPORT_CN.md](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/REPORT_CN.md) | - | Full Chinese narrative report |
| [run_summary_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/run_summary_by_venue_year.csv) | 84 | Per venue-year paper counts, graph sizes, topic counts, outlier rates |
| [top10_topics_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/top10_topics_by_venue_year.csv) | 700 | Top 10 topics for every available venue-year |
| [topic_summary_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/topic_summary_by_venue_year.csv) | 763 | Full topic summary with labels, keywords, counts, shares, representative titles |
| [label_trend_by_venue_year.csv](docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25/label_trend_by_venue_year.csv) | 763 | Topic-label trend table across years and venues |
| [venue_year_family_composition.csv](docs/visuals/xhs_composition_atlas_2020_2026/venue_year_family_composition.csv) | 473 | Broad family composition for every venue-year |
| [venue_year_topic_composition_full.csv](docs/visuals/xhs_composition_atlas_2020_2026/venue_year_topic_composition_full.csv) | 763 | Complete fine-grained topic composition for every venue-year |

## Topic Composition Atlas

The repository keeps the visual atlas as linked files instead of embedding large PNGs in the README.

- Year-by-year venue composition: [docs/visuals/xhs_composition_atlas_2020_2026/by_year/](docs/visuals/xhs_composition_atlas_2020_2026/by_year/)
- Venue-by-venue yearly composition: [docs/visuals/xhs_composition_atlas_2020_2026/by_venue/](docs/visuals/xhs_composition_atlas_2020_2026/by_venue/)

Each horizontal composition bar represents all accepted papers from one venue-year. Colors encode broad topic families and segment lengths encode shares. The right-side labels are only reading aids; the full fine-grained composition is available in CSV.

## Coverage by Year

| Year | Venue-Year Groups | Papers | Topics |
|---:|---:|---:|---:|
| 2020 | 13 | 11,555 | 106 |
| 2021 | 14 | 13,334 | 117 |
| 2022 | 14 | 14,333 | 117 |
| 2023 | 13 | 17,424 | 120 |
| 2024 | 14 | 22,334 | 132 |
| 2025 | 14 | 28,619 | 144 |
| 2026 | 2 | 9,501 | 27 |

## Coverage by Area

| Area | Venues | Venue-Year Groups | Papers | Topics |
|---|---|---:|---:|---:|
| Computer Vision | CVPR, ICCV, ECCV | 12 | 24,999 | 132 |
| Machine Learning | ICLR, ICML, NeurIPS | 19 | 46,238 | 214 |
| NLP | ACL, EMNLP, NAACL | 16 | 14,651 | 147 |
| General AI | AAAI, IJCAI | 13 | 21,179 | 121 |
| Multimedia | ACMMM | 6 | 5,006 | 57 |
| Data Mining / IR / Web | KDD, SIGIR, WWW | 18 | 5,027 | 92 |

## Coverage by Venue

| Area | Venue | Years | Venue-Year Groups | Papers | Topics | Latest #1 topic |
|---|---|---:|---:|---:|---:|---|
| General AI | AAAI | 2020-2026 | 7 | 15,638 | 73 | 2026: 三维视觉：Gaussian Splatting、新视角合成与重建 (610, 14.70%) |
| General AI | IJCAI | 2020-2025 | 6 | 5,541 | 48 | 2025: 多模态理解：视觉语言表征与跨模态对齐 (277, 21.64%) |
| Machine Learning | ICLR | 2020-2026 | 7 | 15,529 | 71 | 2026: 生成模型：视频扩散生成与编辑 (752, 14.05%) |
| Machine Learning | ICML | 2020-2025 | 6 | 11,268 | 64 | 2025: 高效大模型：推理加速、压缩与资源优化 (446, 13.39%) |
| Machine Learning | NeurIPS | 2020-2025 | 6 | 19,441 | 79 | 2025: 大模型推理：RL驱动推理与奖励学习 (781, 14.77%) |
| Computer Vision | CVPR | 2020-2025 | 6 | 13,140 | 68 | 2025: 生成模型：文生图、扩散采样与图像编辑 (458, 15.95%) |
| Computer Vision | ICCV | 2021-2025 | 3 | 6,469 | 36 | 2025: 多模态大模型：视觉语言理解与跨模态推理 (460, 17.03%) |
| Computer Vision | ECCV | 2020-2024 | 3 | 5,390 | 28 | 2024: 开放词汇视觉：开放词汇检测、分割与CLIP语义 (373, 15.63%) |
| NLP | ACL | 2020-2025 | 6 | 5,902 | 60 | 2025: 高效大模型：长上下文、注意力与推理优化 (308, 18.13%) |
| NLP | EMNLP | 2020-2025 | 6 | 6,550 | 56 | 2025: 检索增强大模型：RAG、知识注入与问答 (254, 14.04%) |
| NLP | NAACL | 2021-2025 | 4 | 2,199 | 31 | 2025: 大模型社会安全：偏见、虚假信息与检测 (126, 17.55%) |
| Multimedia | ACMMM | 2020-2025 | 6 | 5,006 | 57 | 2025: 多媒体检索：跨模态检索、语义匹配与内容理解 (197, 15.77%) |
| Data Mining | KDD | 2020-2025 | 6 | 1,985 | 36 | 2025: 图基础模型：LLM增强图学习与节点表示 (122, 22.10%) |
| Information Retrieval | SIGIR | 2020-2025 | 6 | 1,077 | 23 | 2025: 推荐系统：偏好建模、反馈学习与个性化排序 (85, 35.56%) |
| Web / Recommender Systems | WWW | 2020-2025 | 6 | 1,965 | 33 | 2025: 推荐系统：检索增强推荐、排序与个性化 (40, 25.97%) |

## Full Venue-Year Matrix

<details>
<summary><strong>Full venue-year coverage matrix</strong>: each cell is papers/topics</summary>

| Venue | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAAI | 1,601/8 | 1,642/8 | 1,315/10 | 1,572/8 | 2,331/12 | 3,028/13 | 4,149/14 |
| IJCAI | 778/9 | 722/7 | 862/7 | 851/7 | 1,048/9 | 1,280/9 | - |
| ICLR | 687/8 | 860/8 | 1,094/8 | 1,573/10 | 2,260/10 | 3,703/14 | 5,352/13 |
| ICML | 1,084/9 | 1,183/8 | 1,233/11 | 1,828/12 | 2,610/11 | 3,330/13 | - |
| NeurIPS | 1,898/11 | 2,334/12 | 2,671/13 | 3,218/14 | 4,034/14 | 5,286/15 | - |
| CVPR | 1,466/11 | 1,660/12 | 2,074/12 | 2,353/11 | 2,716/10 | 2,871/12 | - |
| ICCV | - | 1,612/13 | - | 2,156/12 | - | 2,701/11 | - |
| ECCV | 1,358/11 | - | 1,645/8 | - | 2,387/9 | - | - |
| ACL | 778/10 | 710/8 | 700/9 | 1,075/10 | 940/11 | 1,699/12 | - |
| EMNLP | 751/9 | 847/9 | 828/9 | 1,047/9 | 1,268/11 | 1,809/9 | - |
| NAACL | - | 477/8 | 442/7 | - | 562/8 | 718/8 | - |
| ACMMM | 473/7 | 542/9 | 691/9 | 902/12 | 1,149/10 | 1,249/10 | - |
| KDD | 217/5 | 239/5 | 253/6 | 313/5 | 411/6 | 552/9 | - |
| SIGIR | 147/3 | 151/5 | 161/3 | 165/4 | 214/4 | 239/4 | - |
| WWW | 317/5 | 355/5 | 364/5 | 371/6 | 404/7 | 154/5 | - |

</details>

## Latest Full Topic Lists by Venue

The README lists all topics for the latest available year of every venue, rather than a few hand-picked examples. Complete topic rows for every venue-year are in `topic_summary_by_venue_year.csv` and `venue_year_topic_composition_full.csv`.

<details>
<summary><strong>AAAI 2026</strong>: all 14 topics, 4,149 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 三维视觉：Gaussian Splatting、新视角合成与重建 | 610 | 14.70% |
| 2 | 大模型推理：问答、常识与思维链 | 596 | 14.36% |
| 3 | 强化学习：策略优化、奖励学习与控制 | 463 | 11.16% |
| 4 | 图学习：图异常检测、聚类与结构表示 | 391 | 9.42% |
| 5 | 高效大模型：推理加速、压缩与资源优化 | 344 | 8.29% |
| 6 | 可信安全：对抗攻击、后门、水印与隐私 | 328 | 7.91% |
| 7 | 多模态大模型：视觉语言理解与跨模态推理 | 295 | 7.11% |
| 8 | 多模态音视频生成：语音、音乐与情感生成 | 251 | 6.05% |
| 9 | 视频理解：动作识别、长视频与时序建模 | 216 | 5.21% |
| 10 | 推荐系统：检索增强推荐、排序与个性化 | 201 | 4.84% |
| 11 | 医疗视觉：医学影像分割、病理与临床影像 | 158 | 3.81% |
| 12 | 时序建模：时间序列预测、动力系统与基础模型 | 120 | 2.89% |
| 13 | AI4Science：蛋白结构、序列与功能建模 | 105 | 2.53% |
| 14 | 神经科学AI：脑活动建模、EEG与脉冲网络 | 71 | 1.71% |

</details>

<details>
<summary><strong>IJCAI 2025</strong>: all 9 topics, 1,280 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 多模态理解：视觉语言表征与跨模态对齐 | 277 | 21.64% |
| 2 | 强化学习：策略优化、奖励学习与控制 | 235 | 18.36% |
| 3 | 图学习：图聚类、表示学习与结构匹配 | 164 | 12.81% |
| 4 | 大模型推理：问答、常识与思维链 | 159 | 12.42% |
| 5 | 时序建模：时间序列预测、动力系统与基础模型 | 139 | 10.86% |
| 6 | 优化理论：梯度方法、收敛性与训练动力学 | 119 | 9.30% |
| 7 | 可信安全：对抗攻击、后门、水印与隐私 | 86 | 6.72% |
| 8 | 推荐系统：排序、召回与点击率预测 | 65 | 5.08% |
| 9 | 医疗视觉：医学影像分割、病理与临床影像 | 36 | 2.81% |

</details>

<details>
<summary><strong>ICLR 2026</strong>: all 13 topics, 5,352 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 生成模型：视频扩散生成与编辑 | 752 | 14.05% |
| 2 | 高效大模型：推理加速、压缩与资源优化 | 691 | 12.91% |
| 3 | 大模型推理：RL驱动推理与奖励学习 | 655 | 12.24% |
| 4 | 优化理论：随机/非凸优化与收敛率 | 645 | 12.05% |
| 5 | 多模态大模型：视觉语言理解与跨模态推理 | 625 | 11.68% |
| 6 | 视频理解：动作识别、长视频与时序建模 | 513 | 9.59% |
| 7 | 大模型评测：人类偏好、任务指标与领域评估 | 372 | 6.95% |
| 8 | 可信安全：对抗攻击、后门、水印与隐私 | 358 | 6.69% |
| 9 | 时序建模：时间序列预测、动力系统与基础模型 | 256 | 4.78% |
| 10 | 检索增强大模型：RAG、知识注入与问答 | 194 | 3.62% |
| 11 | AI4Science：蛋白结构、序列与功能建模 | 145 | 2.71% |
| 12 | 大模型对齐：偏好优化、RLHF与奖励建模 | 81 | 1.51% |
| 13 | 多模态音视频生成：语音、音乐与情感生成 | 65 | 1.21% |

</details>

<details>
<summary><strong>ICML 2025</strong>: all 13 topics, 3,330 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 高效大模型：推理加速、压缩与资源优化 | 446 | 13.39% |
| 2 | 在线决策：Bandit、后悔界与探索 | 411 | 12.34% |
| 3 | 多模态大模型：视觉语言理解与跨模态推理 | 344 | 10.33% |
| 4 | 迁移泛化：域适应、OOD泛化与鲁棒表征 | 330 | 9.91% |
| 5 | 代码大模型：代码生成、程序理解与评测 | 308 | 9.25% |
| 6 | 可信安全：对抗攻击、后门、水印与隐私 | 267 | 8.02% |
| 7 | 生成模型：文生图、扩散采样与图像编辑 | 238 | 7.15% |
| 8 | 优化理论：随机/非凸优化与收敛率 | 220 | 6.61% |
| 9 | 因果学习：因果发现、反事实与处理效应 | 181 | 5.44% |
| 10 | 图学习：GNN、节点分类与链接预测 | 172 | 5.17% |
| 11 | 时序建模：时间序列预测、动力系统与基础模型 | 154 | 4.62% |
| 12 | 生成模型：扩散模型、采样与内容生成 | 151 | 4.53% |
| 13 | 强化学习：策略优化、奖励学习与控制 | 108 | 3.24% |

</details>

<details>
<summary><strong>NeurIPS 2025</strong>: all 15 topics, 5,286 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 大模型推理：RL驱动推理与奖励学习 | 781 | 14.77% |
| 2 | 生成模型：视频扩散生成与编辑 | 583 | 11.03% |
| 3 | 多模态大模型：视觉语言理解与跨模态推理 | 575 | 10.88% |
| 4 | 在线决策：Bandit、后悔界与探索 | 570 | 10.78% |
| 5 | 优化理论：随机/非凸优化与收敛率 | 548 | 10.37% |
| 6 | 生成模型：文生图、扩散采样与图像编辑 | 538 | 10.18% |
| 7 | 高效大模型：推理加速、压缩与资源优化 | 414 | 7.83% |
| 8 | 图基础模型：LLM增强图学习与节点表示 | 235 | 4.45% |
| 9 | 可信安全：对抗攻击、后门、水印与隐私 | 226 | 4.28% |
| 10 | 神经科学AI：脑活动建模、EEG与脉冲网络 | 193 | 3.65% |
| 11 | AI4Science：蛋白结构、序列与功能建模 | 169 | 3.20% |
| 12 | 联邦大模型微调：LoRA、客户端异构与模型合并 | 154 | 2.91% |
| 13 | 时序建模：时间序列预测、动力系统与基础模型 | 127 | 2.40% |
| 14 | 因果学习：因果发现、反事实与处理效应 | 111 | 2.10% |
| 15 | 多模态音视频生成：语音、音乐与情感生成 | 62 | 1.17% |

</details>

<details>
<summary><strong>CVPR 2025</strong>: all 12 topics, 2,871 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 生成模型：文生图、扩散采样与图像编辑 | 458 | 15.95% |
| 2 | 三维视觉：点云、深度估计与相机姿态 | 413 | 14.39% |
| 3 | 多模态大模型：视觉语言理解与跨模态推理（视觉语言） | 349 | 12.16% |
| 4 | 生成模型：视频扩散生成与编辑 | 276 | 9.61% |
| 5 | 具身智能：机器人操作、导航与视觉语言动作 | 253 | 8.81% |
| 6 | 多模态音视频生成：语音、音乐与情感生成 | 197 | 6.86% |
| 7 | 多模态大模型：视觉语言理解与跨模态推理（搜索排序） | 196 | 6.83% |
| 8 | 迁移泛化：域适应、OOD泛化与鲁棒表征 | 170 | 5.92% |
| 9 | 高效大模型：推理加速、压缩与资源优化 | 169 | 5.89% |
| 10 | 三维视觉：Gaussian Splatting、新视角合成与重建 | 164 | 5.71% |
| 11 | 异常检测：图异常、欺诈检测与时序异常 | 147 | 5.12% |
| 12 | 医疗视觉：医学影像分割、病理与临床影像 | 79 | 2.75% |

</details>

<details>
<summary><strong>ICCV 2025</strong>: all 11 topics, 2,701 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 多模态大模型：视觉语言理解与跨模态推理 | 460 | 17.03% |
| 2 | 三维视觉：点云、深度估计与相机姿态 | 373 | 13.81% |
| 3 | 生成模型：视频扩散生成与编辑 | 341 | 12.62% |
| 4 | 底层视觉：超分、去噪、增强与图像复原 | 253 | 9.37% |
| 5 | 生成模型：文生图、扩散采样与图像编辑 | 252 | 9.33% |
| 6 | 异常检测：图异常、欺诈检测与时序异常 | 236 | 8.74% |
| 7 | 迁移泛化：域适应、OOD泛化与鲁棒表征 | 230 | 8.52% |
| 8 | 具身智能：机器人操作、导航与视觉语言动作 | 227 | 8.40% |
| 9 | 三维视觉：Gaussian Splatting、新视角合成与重建 | 178 | 6.59% |
| 10 | 多模态理解：视觉语言表征与跨模态对齐 | 118 | 4.37% |
| 11 | 事件视觉：事件相机、运动估计与时序感知 | 33 | 1.22% |

</details>

<details>
<summary><strong>ECCV 2024</strong>: all 9 topics, 2,387 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 开放词汇视觉：开放词汇检测、分割与CLIP语义 | 373 | 15.63% |
| 2 | 自动驾驶感知：LiDAR、轨迹与三维检测 | 363 | 15.21% |
| 3 | 三维视觉：Gaussian Splatting、新视角合成与重建 | 342 | 14.33% |
| 4 | 可信安全：对抗攻击、后门、水印与隐私 | 312 | 13.07% |
| 5 | 生成模型：文生图、扩散采样与图像编辑 | 269 | 11.27% |
| 6 | 底层视觉：超分、去噪、增强与图像复原 | 238 | 9.97% |
| 7 | 三维视觉：点云、深度估计与相机姿态 | 237 | 9.93% |
| 8 | 多模态大模型：视觉语言理解与跨模态推理 | 206 | 8.63% |
| 9 | 事件视觉：事件相机、运动估计与时序感知 | 47 | 1.97% |

</details>

<details>
<summary><strong>ACL 2025</strong>: all 12 topics, 1,699 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 高效大模型：长上下文、注意力与推理优化 | 308 | 18.13% |
| 2 | 大模型推理：问答、常识与思维链 | 206 | 12.12% |
| 3 | 大模型社会安全：偏见、虚假信息与检测 | 186 | 10.95% |
| 4 | 多模态大模型：视觉语言理解与跨模态推理 | 152 | 8.95% |
| 5 | 大模型评测：人类偏好、任务指标与领域评估 | 141 | 8.30% |
| 6 | 语音音频：ASR、说话人与音频理解 | 140 | 8.24% |
| 7 | 检索增强大模型：RAG、知识注入与问答 | 127 | 7.47% |
| 8 | 语言模型分析：认知、语言结构与可解释性 | 126 | 7.42% |
| 9 | 可信安全：对抗攻击、后门、水印与隐私 | 98 | 5.77% |
| 10 | 代码大模型：代码生成、程序理解与评测 | 87 | 5.12% |
| 11 | 高效大模型：推理加速、压缩与资源优化 | 71 | 4.18% |
| 12 | 强化学习：策略优化、奖励学习与控制 | 57 | 3.35% |

</details>

<details>
<summary><strong>EMNLP 2025</strong>: all 9 topics, 1,809 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 检索增强大模型：RAG、知识注入与问答 | 254 | 14.04% |
| 2 | 大模型训练：微调、数据配方与任务适配 | 251 | 13.88% |
| 3 | 大模型推理：问答、常识与思维链 | 244 | 13.49% |
| 4 | 语音音频：ASR、说话人与音频理解 | 239 | 13.21% |
| 5 | 大模型社会安全：偏见、虚假信息与检测 | 229 | 12.66% |
| 6 | 多模态大模型：视觉语言理解与跨模态推理 | 223 | 12.33% |
| 7 | 代码大模型：代码生成、程序理解与评测 | 170 | 9.40% |
| 8 | 可信安全：对抗攻击、后门、水印与隐私 | 166 | 9.18% |
| 9 | 医疗大模型：临床推理、医学影像与健康问答 | 33 | 1.82% |

</details>

<details>
<summary><strong>NAACL 2025</strong>: all 8 topics, 718 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 大模型社会安全：偏见、虚假信息与检测 | 126 | 17.55% |
| 2 | 高效大模型：长上下文、注意力与推理优化 | 116 | 16.16% |
| 3 | 语音音频：ASR、说话人与音频理解 | 97 | 13.51% |
| 4 | 代码大模型：代码生成、程序理解与评测 | 94 | 13.09% |
| 5 | 大模型推理：问答、常识与思维链 | 92 | 12.81% |
| 6 | 检索增强大模型：RAG、知识注入与问答 | 83 | 11.56% |
| 7 | 多模态大模型：视觉语言理解与跨模态推理 | 63 | 8.77% |
| 8 | 可信安全：对抗攻击、后门、水印与隐私 | 47 | 6.55% |

</details>

<details>
<summary><strong>ACMMM 2025</strong>: all 10 topics, 1,249 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 多媒体检索：跨模态检索、语义匹配与内容理解 | 197 | 15.77% |
| 2 | 三维视觉：点云、深度估计与相机姿态 | 163 | 13.05% |
| 3 | 多模态音视频生成：语音、音乐与情感生成 | 162 | 12.97% |
| 4 | 生成模型：视频扩散生成与编辑 | 143 | 11.45% |
| 5 | 多媒体安全：Deepfake检测、伪造识别与攻防 | 131 | 10.49% |
| 6 | 多模态大模型：视觉语言理解与跨模态推理 | 109 | 8.73% |
| 7 | 多模态理解：视觉语言表征与跨模态对齐 | 107 | 8.57% |
| 8 | 生成模型：文生图、扩散采样与图像编辑 | 95 | 7.61% |
| 9 | 图学习：图聚类、表示学习与结构匹配 | 73 | 5.84% |
| 10 | 医疗视觉：医学影像分割、病理与临床影像 | 69 | 5.52% |

</details>

<details>
<summary><strong>KDD 2025</strong>: all 9 topics, 552 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 图基础模型：LLM增强图学习与节点表示 | 122 | 22.10% |
| 2 | 大模型推理：问答、常识与思维链 | 87 | 15.76% |
| 3 | 时序建模：时间序列预测、动力系统与基础模型 | 69 | 12.50% |
| 4 | AI4Science：分子生成、药物发现与化学建模 | 65 | 11.78% |
| 5 | 推荐系统：偏好建模、反馈学习与个性化排序 | 65 | 11.78% |
| 6 | 图学习：图异常检测、聚类与结构表示 | 49 | 8.88% |
| 7 | 在线决策：Bandit、后悔界与探索 | 35 | 6.34% |
| 8 | 具身智能：机器人操作、导航与视觉语言动作 | 33 | 5.98% |
| 9 | 联邦大模型微调：LoRA、客户端异构与模型合并 | 27 | 4.89% |

</details>

<details>
<summary><strong>SIGIR 2025</strong>: all 4 topics, 239 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 推荐系统：偏好建模、反馈学习与个性化排序 | 85 | 35.56% |
| 2 | 多媒体检索：跨模态检索、语义匹配与内容理解 | 77 | 32.22% |
| 3 | 检索增强大模型：RAG、知识注入与问答 | 51 | 21.34% |
| 4 | 推荐系统：排序、召回与点击率预测 | 26 | 10.88% |

</details>

<details>
<summary><strong>WWW 2025</strong>: all 5 topics, 154 papers</summary>

| Rank | Topic label | Papers | Share |
|---:|---|---:|---:|
| 1 | 推荐系统：检索增强推荐、排序与个性化 | 40 | 25.97% |
| 2 | 图学习：图异常检测、聚类与结构表示 | 36 | 23.38% |
| 3 | 视觉感知：目标检测、识别与视觉表征 | 32 | 20.78% |
| 4 | 大模型推理：问答、常识与思维链 | 29 | 18.83% |
| 5 | 可信安全：对抗攻击、后门、水印与隐私 | 17 | 11.04% |

</details>

## Automatic README Updates

README statistics are generated from the committed CSV artifacts by `src/update_readme_data.py`. When a new venue or year is added to the result CSVs, rerun the script and the coverage tables plus latest full-topic sections will update automatically.

A GitHub Actions workflow also runs the script when result CSVs change on `main`.
<!-- AI-PAPER-TRENDS:END -->

## Quick Start

```bash
git clone https://github.com/zhihengli-casia/AI-Paper-Trends.git
cd AI-Paper-Trends
conda create -n ai-paper-trends python=3.10
conda activate ai-paper-trends
pip install -r requirements.txt
```

### Legacy Single-Venue OpenReview Analysis

The original ICLR/OpenReview workflow is still available for analyzing submissions, reviews, decisions, and reviewer scores where OpenReview exposes them.

```bash
python main.py --config configs/iclr_2025_full_analysis.yaml
```

### Multi-Conference Accepted-Paper Trend Analysis

Run the large workflow in `tmux` or another long-running session.

```bash
# 1. Crawl public metadata.
python src/crawl_recent_conferences.py \
  --output-dir data/recent_conferences \
  --run-name recent_ai_conference_papers \
  --fetch-detail-abstracts

# 2. Build the strict main-conference accepted view.
python src/build_main_accepted_view.py \
  --input data/recent_conferences/recent_ai_conference_papers.jsonl \
  --output data/recent_conferences/main_accepted_ai_conference_papers.jsonl

# 3. Cluster each venue-year independently with kNN + Louvain.
python src/analyze_venue_year_louvain_topics.py \
  --input data/recent_conferences/main_accepted_ai_conference_papers.jsonl \
  --output-dir results/venue_year_main_accepted_topics_louvain \
  --model-name models/bge-base-en-v1.5 \
  --device mps
```

If only naming rules or aggregate tables changed, rebuild outputs without rerunning embeddings/clustering:

```bash
python src/rebuild_venue_year_outputs.py \
  --results-dir results/venue_year_main_accepted_topics_louvain \
  --top-k 10
```

Generate the composition atlas:

```bash
python src/create_xhs_composition_atlas.py \
  --results-dir docs/results/venue_year_main_accepted_topics_2020_2026_louvain_bge_m25 \
  --output-dir docs/visuals/xhs_composition_atlas_2020_2026
```

Refresh README tables from the committed CSV artifacts:

```bash
python src/update_readme_data.py
```

## Project Structure

```text
.
├── .github/workflows/           # README data auto-update workflow
├── configs/                     # Legacy single-venue OpenReview configs
├── data/                        # Local crawls and intermediate files; gitignored
├── docs/results/                # Lightweight committed analysis artifacts
├── docs/visuals/                # Topic-composition atlas visuals
├── main.py                      # Legacy OpenReview entry point
├── notebooks/                   # Notebook examples
├── src/
│   ├── crawl_recent_conferences.py
│   ├── crawl_2020_2025_backfill.py
│   ├── build_main_accepted_view.py
│   ├── analyze_venue_year_louvain_topics.py
│   ├── create_xhs_composition_atlas.py
│   ├── refine_topic_names.py
│   ├── rebuild_venue_year_outputs.py
│   ├── update_readme_data.py
│   └── ...
└── requirements.txt
```

## Notes

- The multi-conference trend analysis uses public paper metadata only.
- Review scores and rejected submissions are not uniformly public across venues.
- OpenReview-based single-conference analysis can include submissions/rejections/reviews when available, but those fields should not be mixed directly with accepted-only proceedings data from other venues.

## License

MIT License.
