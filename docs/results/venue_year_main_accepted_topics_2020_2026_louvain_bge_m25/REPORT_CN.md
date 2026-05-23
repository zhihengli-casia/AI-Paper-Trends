# Venue-Year Main Accepted Topic Analysis

- Rebuilt at: 2026-05-23T12:34:08
- Papers analyzed: 117,100
- Clustering unit: independent conference-year kNN graph + Louvain runs
- Topic names: deterministic rule-based refinement from keywords and representative titles

## Run Summary

| venue   |   year |   papers |   topics_excluding_outlier |   raw_outliers | raw_outlier_rate   |   final_outliers | final_outlier_rate   |
|:--------|-------:|---------:|---------------------------:|---------------:|:-------------------|-----------------:|:---------------------|
| AAAI    |   2020 |     1601 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| ACL     |   2020 |      778 |                         10 |              0 | 0.00%              |                0 | 0.00%                |
| ACMMM   |   2020 |      473 |                          7 |              0 | 0.00%              |                0 | 0.00%                |
| CVPR    |   2020 |     1466 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| ECCV    |   2020 |     1358 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| EMNLP   |   2020 |      751 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| ICLR    |   2020 |      687 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| ICML    |   2020 |     1084 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| IJCAI   |   2020 |      778 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| KDD     |   2020 |      217 |                          5 |              0 | 0.00%              |                0 | 0.00%                |
| NeurIPS |   2020 |     1898 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| SIGIR   |   2020 |      147 |                          3 |              0 | 0.00%              |                0 | 0.00%                |
| WWW     |   2020 |      317 |                          5 |              0 | 0.00%              |                0 | 0.00%                |
| AAAI    |   2021 |     1642 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| ACL     |   2021 |      710 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| ACMMM   |   2021 |      542 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| CVPR    |   2021 |     1660 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| EMNLP   |   2021 |      847 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| ICCV    |   2021 |     1612 |                         13 |              0 | 0.00%              |                0 | 0.00%                |
| ICLR    |   2021 |      860 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| ICML    |   2021 |     1183 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| IJCAI   |   2021 |      722 |                          7 |              0 | 0.00%              |                0 | 0.00%                |
| KDD     |   2021 |      239 |                          5 |              0 | 0.00%              |                0 | 0.00%                |
| NAACL   |   2021 |      477 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| NeurIPS |   2021 |     2334 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| SIGIR   |   2021 |      151 |                          5 |              0 | 0.00%              |                0 | 0.00%                |
| WWW     |   2021 |      355 |                          5 |              0 | 0.00%              |                0 | 0.00%                |
| AAAI    |   2022 |     1315 |                         10 |              0 | 0.00%              |                0 | 0.00%                |
| ACL     |   2022 |      700 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| ACMMM   |   2022 |      691 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| CVPR    |   2022 |     2074 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| ECCV    |   2022 |     1645 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| EMNLP   |   2022 |      828 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| ICLR    |   2022 |     1094 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| ICML    |   2022 |     1233 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| IJCAI   |   2022 |      862 |                          7 |              0 | 0.00%              |                0 | 0.00%                |
| KDD     |   2022 |      253 |                          6 |              0 | 0.00%              |                0 | 0.00%                |
| NAACL   |   2022 |      442 |                          7 |              0 | 0.00%              |                0 | 0.00%                |
| NeurIPS |   2022 |     2671 |                         13 |              0 | 0.00%              |                0 | 0.00%                |
| SIGIR   |   2022 |      161 |                          3 |              0 | 0.00%              |                0 | 0.00%                |
| WWW     |   2022 |      364 |                          5 |              0 | 0.00%              |                0 | 0.00%                |
| AAAI    |   2023 |     1572 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| ACL     |   2023 |     1075 |                         10 |              0 | 0.00%              |                0 | 0.00%                |
| ACMMM   |   2023 |      902 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| CVPR    |   2023 |     2353 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| EMNLP   |   2023 |     1047 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| ICCV    |   2023 |     2156 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| ICLR    |   2023 |     1573 |                         10 |              0 | 0.00%              |                0 | 0.00%                |
| ICML    |   2023 |     1828 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| IJCAI   |   2023 |      851 |                          7 |              0 | 0.00%              |                0 | 0.00%                |
| KDD     |   2023 |      313 |                          5 |              0 | 0.00%              |                0 | 0.00%                |
| NeurIPS |   2023 |     3218 |                         14 |              0 | 0.00%              |                0 | 0.00%                |
| SIGIR   |   2023 |      165 |                          4 |              0 | 0.00%              |                0 | 0.00%                |
| WWW     |   2023 |      371 |                          6 |              0 | 0.00%              |                0 | 0.00%                |
| AAAI    |   2024 |     2331 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| ACL     |   2024 |      940 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| ACMMM   |   2024 |     1149 |                         10 |              0 | 0.00%              |                0 | 0.00%                |
| CVPR    |   2024 |     2716 |                         10 |              0 | 0.00%              |                0 | 0.00%                |
| ECCV    |   2024 |     2387 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| EMNLP   |   2024 |     1268 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| ICLR    |   2024 |     2260 |                         10 |              0 | 0.00%              |                0 | 0.00%                |
| ICML    |   2024 |     2610 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| IJCAI   |   2024 |     1048 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| KDD     |   2024 |      411 |                          6 |              0 | 0.00%              |                0 | 0.00%                |
| NAACL   |   2024 |      562 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| NeurIPS |   2024 |     4034 |                         14 |              0 | 0.00%              |                0 | 0.00%                |
| SIGIR   |   2024 |      214 |                          4 |              0 | 0.00%              |                0 | 0.00%                |
| WWW     |   2024 |      404 |                          7 |              0 | 0.00%              |                0 | 0.00%                |
| AAAI    |   2025 |     3028 |                         13 |              0 | 0.00%              |                0 | 0.00%                |
| ACL     |   2025 |     1699 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| ACMMM   |   2025 |     1249 |                         10 |              0 | 0.00%              |                0 | 0.00%                |
| CVPR    |   2025 |     2871 |                         12 |              0 | 0.00%              |                0 | 0.00%                |
| EMNLP   |   2025 |     1809 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| ICCV    |   2025 |     2701 |                         11 |              0 | 0.00%              |                0 | 0.00%                |
| ICLR    |   2025 |     3703 |                         14 |              0 | 0.00%              |                0 | 0.00%                |
| ICML    |   2025 |     3330 |                         13 |              0 | 0.00%              |                0 | 0.00%                |
| IJCAI   |   2025 |     1280 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| KDD     |   2025 |      552 |                          9 |              0 | 0.00%              |                0 | 0.00%                |
| NAACL   |   2025 |      718 |                          8 |              0 | 0.00%              |                0 | 0.00%                |
| NeurIPS |   2025 |     5286 |                         15 |              0 | 0.00%              |                0 | 0.00%                |
| SIGIR   |   2025 |      239 |                          4 |              0 | 0.00%              |                0 | 0.00%                |
| WWW     |   2025 |      154 |                          5 |              0 | 0.00%              |                0 | 0.00%                |
| AAAI    |   2026 |     4149 |                         14 |              0 | 0.00%              |                0 | 0.00%                |
| ICLR    |   2026 |     5352 |                         13 |              0 | 0.00%              |                0 | 0.00%                |

## Top 10 Topics By Venue-Year

### AAAI 2020

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                           |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------|
|      1 | 知识抽取：实体识别、关系抽取与事件理解     |    337 |           1601 |   21.05 | language, knowledge, sentence, text, training    |
|      2 | 视频理解：动作识别、长视频与时序建模       |    335 |           1601 |   20.92 | image, features, video, object, images           |
|      3 | 优化理论：梯度方法、收敛性与训练动力学     |    276 |           1601 |   17.24 | problem, algorithm, training, time, deep         |
|      4 | 推荐系统：排序、召回与点击率预测           |    201 |           1601 |   12.55 | graph, clustering, prediction, user, information |
|      5 | 强化学习：策略优化、奖励学习与控制         |    132 |           1601 |    8.24 | reinforcement, policy, agent, agents, algorithm  |
|      6 | 可信安全：对抗攻击、后门、水印与隐私       |    116 |           1601 |    7.25 | adversarial, domain, attack, target, training    |
|      7 | 多智能体：机制设计、拍卖与资源分配         |    105 |           1601 |    6.56 | agents, problem, games, fairness, algorithm      |
|      8 | 概率机器学习：变分推断、不确定性与后验建模 |     99 |           1601 |    6.18 | planning, problem, search, problems, plan        |

### AAAI 2021

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------|
|      1 | 视觉感知：语义/实例/全景分割               |    315 |           1642 |   19.18 | image, segmentation, features, object, video          |
|      2 | 生成模型：扩散模型、采样与内容生成         |    310 |           1642 |   18.88 | language, knowledge, information, generation, text    |
|      3 | 优化理论：梯度方法、收敛性与训练动力学     |    234 |           1642 |   14.25 | training, deep, knowledge, optimization, problem      |
|      4 | 概率机器学习：变分推断、不确定性与后验建模 |    203 |           1642 |   12.36 | problem, problems, agents, games, algorithm           |
|      5 | 在线决策：Bandit、后悔界与探索             |    202 |           1642 |   12.3  | planning, policy, algorithm, reinforcement, problem   |
|      6 | 推荐系统：排序、召回与点击率预测           |    189 |           1642 |   11.51 | graph, clustering, recommendation, user, time         |
|      7 | 可信安全：对抗攻击、后门、水印与隐私       |    131 |           1642 |    7.98 | adversarial, training, attacks, attack, robustness    |
|      8 | 因果学习：因果发现、反事实与处理效应       |     58 |           1642 |    3.53 | causal, covid-19, variables, counterfactual, survival |

### AAAI 2022

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                               |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------|
|      1 | 迁移泛化：域适应、OOD泛化与鲁棒表征      |    217 |           1315 |   16.5  | domain, object, segmentation, feature, detection     |
|      2 | 优化理论：随机/非凸优化与收敛率          |    179 |           1315 |   13.61 | algorithm, optimization, problem, causal, algorithms |
|      3 | 视觉感知：目标检测、识别与视觉表征       |    167 |           1315 |   12.7  | image, images, visual, features, transformer         |
|      4 | 生成模型：扩散模型、采样与内容生成       |    152 |           1315 |   11.56 | language, knowledge, text, generation, human         |
|      5 | 强化学习：策略优化、奖励学习与控制       |    139 |           1315 |   10.57 | policy, reinforcement, agent, agents, reward         |
|      6 | 语言推理：问答、常识与多跳推理           |    136 |           1315 |   10.34 | problem, search, problems, algorithm, planning       |
|      7 | 多媒体安全：Deepfake检测、伪造识别与攻防 |    116 |           1315 |    8.82 | adversarial, training, attacks, attack, detection    |
|      8 | 图学习：图聚类、表示学习与结构匹配       |     92 |           1315 |    7    | graph, clustering, graphs, gnns, node                |
|      9 | 大模型社会安全：偏见、虚假信息与检测     |     59 |           1315 |    4.49 | agents, welfare, social, study, voting               |
|     10 | 视频理解：动作识别、长视频与时序建模     |     58 |           1315 |    4.41 | video, temporal, motion, action, videos              |

### AAAI 2023

|   排名 | 细主题名                                |   篇数 |   当年会议篇数 |   占比% | 关键词                                               |
|-------:|:----------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------|
|      1 | 迁移泛化：域适应、OOD泛化与鲁棒表征     |    440 |           1572 |   27.99 | image, features, images, object, domain              |
|      2 | 优化理论：梯度方法、收敛性与训练动力学  |    237 |           1572 |   15.08 | optimization, training, problem, samples, deep       |
|      3 | 社会计算与安全NLP：偏见、仇恨与立场检测 |    200 |           1572 |   12.72 | problem, algorithm, fairness, agents, games          |
|      4 | 多语言NLP：机器翻译、跨语言与低资源     |    192 |           1572 |   12.21 | knowledge, information, language, training, dialogue |
|      5 | 图学习：图聚类、表示学习与结构匹配      |    163 |           1572 |   10.37 | graph, graphs, clustering, information, node         |
|      6 | 强化学习：离线策略、奖励建模与控制      |    162 |           1572 |   10.31 | policy, reinforcement, agent, offline, planning      |
|      7 | 因果学习：因果发现、反事实与处理效应    |    109 |           1572 |    6.93 | time, causal, graph, temporal, traffic               |
|      8 | 可信安全：对抗攻击、后门、水印与隐私    |     69 |           1572 |    4.39 | adversarial, federated, attack, attacks, robustness  |

### AAAI 2024

|   排名 | 细主题名                                      |   篇数 |   当年会议篇数 |   占比% | 关键词                                            |
|-------:|:----------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------|
|      1 | 强化学习：策略优化、奖励学习与控制            |    428 |           2331 |   18.36 | problem, agents, algorithm, policy, algorithms    |
|      2 | 视觉感知：语义/实例/全景分割                  |    358 |           2331 |   15.36 | visual, image, segmentation, semantic, features   |
|      3 | 迁移泛化：域适应、OOD泛化与鲁棒表征           |    245 |           2331 |   10.51 | domain, training, label, knowledge, classes       |
|      4 | 大模型推理：问答、常识与思维链                |    232 |           2331 |    9.95 | language, llms, knowledge, reasoning, information |
|      5 | 生成模型：文生图、扩散采样与图像编辑          |    230 |           2331 |    9.87 | image, images, diffusion, generation, existing    |
|      6 | 图学习：图聚类、表示学习与结构匹配            |    229 |           2331 |    9.82 | graph, clustering, information, graphs, node      |
|      7 | 三维视觉：点云、深度估计与相机姿态（RAG问答） |    175 |           2331 |    7.51 | point, point cloud, cloud, novel, features        |
|      8 | 可信安全：对抗攻击、后门、水印与隐私          |    154 |           2331 |    6.61 | adversarial, federated, attack, attacks, training |
|      9 | 视频理解：动作识别、长视频与时序建模          |    142 |           2331 |    6.09 | motion, audio, human, video, speech               |
|     10 | 三维视觉：点云、深度估计与相机姿态（图学习）  |     68 |           2331 |    2.92 | causal, effect, treatment, variables, estimation  |

### AAAI 2025

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                               |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------|
|      1 | 生成模型：视频扩散生成与编辑               |    380 |           3028 |   12.55 | image, diffusion, images, generation, video          |
|      2 | 大模型推理：问答、常识与思维链             |    378 |           3028 |   12.48 | llms, language, knowledge, reasoning, llm            |
|      3 | 多模态大模型：视觉语言理解与跨模态推理     |    372 |           3028 |   12.29 | visual, video, multimodal, language, information     |
|      4 | 三维视觉：点云、深度估计与相机姿态         |    342 |           3028 |   11.29 | point, object, semantic, segmentation, cloud         |
|      5 | 强化学习：策略优化、奖励学习与控制         |    304 |           3028 |   10.04 | agents, policy, problem, reinforcement, algorithms   |
|      6 | 可信安全：对抗攻击、后门、水印与隐私       |    255 |           3028 |    8.42 | attacks, federated, adversarial, detection, attack   |
|      7 | 优化理论：梯度方法、收敛性与训练动力学     |    235 |           3028 |    7.76 | optimization, problem, gradient, algorithm, training |
|      8 | 因果学习：因果发现、反事实与处理效应       |    225 |           3028 |    7.43 | clustering, multi-view, label, causal, labels        |
|      9 | AI4Science：分子生成、药物发现与化学建模   |    221 |           3028 |    7.3  | graph, node, graphs, gnns, information               |
|     10 | 时序建模：时间序列预测、动力系统与基础模型 |    105 |           3028 |    3.47 | series, time series, time, forecasting, traffic      |

### AAAI 2026

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 三维视觉：Gaussian Splatting、新视角合成与重建 |    610 |           4149 |   14.7  | image, features, existing, reconstruction, experiments |
|      2 | 大模型推理：问答、常识与思维链                 |    596 |           4149 |   14.36 | reasoning, llms, language, knowledge, framework        |
|      3 | 强化学习：策略优化、奖励学习与控制             |    463 |           4149 |   11.16 | agents, problem, policy, algorithm, algorithms         |
|      4 | 图学习：图异常检测、聚类与结构表示             |    391 |           4149 |    9.42 | graph, clustering, graphs, existing, anomaly           |
|      5 | 高效大模型：推理加速、压缩与资源优化           |    344 |           4149 |    8.29 | knowledge, visual, language, pruning, existing         |
|      6 | 可信安全：对抗攻击、后门、水印与隐私           |    328 |           4149 |    7.91 | attacks, attack, adversarial, privacy, federated       |
|      7 | 多模态大模型：视觉语言理解与跨模态推理         |    295 |           4149 |    7.11 | reasoning, multimodal, visual, video, understanding    |
|      8 | 多模态音视频生成：语音、音乐与情感生成         |    251 |           4149 |    6.05 | generation, diffusion, speech, image, existing         |
|      9 | 视频理解：动作识别、长视频与时序建模           |    216 |           4149 |    5.21 | motion, action, human, navigation, manipulation        |
|     10 | 推荐系统：检索增强推荐、排序与个性化           |    201 |           4149 |    4.84 | recommendation, user, semantic, cross-modal, retrieval |

### ACL 2020

|   排名 | 细主题名                                                            |   篇数 |   当年会议篇数 |   占比% | 关键词                                                         |
|-------:|:--------------------------------------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------------------|
|      1 | 多语言NLP：机器翻译、跨语言与低资源（translation/machine translat） |    122 |            778 |   15.68 | translation, machine translation, machine, language, languages |
|      2 | 高效大模型：推理加速、压缩与资源优化                                |    118 |            778 |   15.17 | language, attention, linguistic, syntactic, information        |
|      3 | 大模型社会安全：偏见、虚假信息与检测                                |     99 |            778 |   12.72 | nlp, bias, social, language, biases                            |
|      4 | 知识图谱：实体关系、推理与补全                                      |     90 |            778 |   11.57 | entity, ner, knowledge, extraction, entities                   |
|      5 | 生成模型：扩散模型、采样与内容生成（搜索排序）                      |     81 |            778 |   10.41 | text, generation, classification, word, language               |
|      6 | 生成模型：扩散模型、采样与内容生成（Agent规划）                     |     72 |            778 |    9.25 | dialogue, dialog, response, generation, conversations          |
|      7 | 语音音频：ASR、说话人与音频理解                                     |     53 |            778 |    6.81 | sentiment, aspect, emotion, visual, analysis                   |
|      8 | 多语言NLP：机器翻译、跨语言与低资源（搜索排序）                     |     51 |            778 |    6.56 | question, questions, reading, answering, answer                |
|      9 | 开放词汇视觉：开放词汇检测、分割与CLIP语义                          |     50 |            778 |    6.43 | parsing, semantic, discourse, semantic parsing, dependency     |
|     10 | 文本生成：摘要、事实性与可控生成                                    |     42 |            778 |    5.4  | summarization, summary, summaries, abstractive, extractive     |

### ACL 2021

|   排名 | 细主题名                                        |   篇数 |   当年会议篇数 |   占比% | 关键词                                                         |
|-------:|:------------------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------------------|
|      1 | 大模型训练：微调、数据配方与任务适配            |    138 |            710 |   19.44 | language, text, training, generation, knowledge                |
|      2 | 大模型社会安全：偏见、虚假信息与检测            |    103 |            710 |   14.51 | detection, sentiment, evidence, social, language               |
|      3 | 多语言NLP：机器翻译、跨语言与低资源（推理效率） |    100 |            710 |   14.08 | translation, machine translation, machine, language, languages |
|      4 | 多语言NLP：机器翻译、跨语言与低资源（推理问答） |     99 |            710 |   13.94 | language, word, embeddings, languages, natural                 |
|      5 | 知识抽取：实体识别、关系抽取与事件理解          |     77 |            710 |   10.85 | entity, ner, relation, entities, named                         |
|      6 | 检索增强大模型：RAG、知识注入与问答             |     70 |            710 |    9.86 | question, answering, question answering, questions, retrieval  |
|      7 | 对话系统：响应生成、情感支持与任务型对话        |     65 |            710 |    9.15 | dialogue, dialog, slot, systems, response                      |
|      8 | 事件视觉：事件相机、运动估计与时序感知          |     58 |            710 |    8.17 | event, summarization, knowledge, events, summaries             |

### ACL 2022

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                        |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------|
|      1 | 多语言语音：语音识别、翻译与手语理解       |    122 |            700 |   17.43 | translation, languages, language, multilingual, cross-lingual |
|      2 | 开放词汇视觉：开放词汇检测、分割与CLIP语义 |    117 |            700 |   16.71 | language, word, representations, semantic, information        |
|      3 | 大模型训练：微调、数据配方与任务适配       |     87 |            700 |   12.43 | language, text, training, prompt, tuning                      |
|      4 | 大模型评测：人类偏好、任务指标与领域评估   |     82 |            700 |   11.71 | language, evaluation, detection, news, metrics                |
|      5 | 大模型推理：问答、常识与思维链             |     77 |            700 |   11    | question, reasoning, questions, knowledge, answering          |
|      6 | 生成模型：扩散模型、采样与内容生成         |     71 |            700 |   10.14 | dialogue, knowledge, response, generation, systems            |
|      7 | 代码智能：程序分析、软件工程与代码检索     |     62 |            700 |    8.86 | summarization, code, contrastive, summaries, text             |
|      8 | 知识抽取：实体识别、关系抽取与事件理解     |     51 |            700 |    7.29 | entity, extraction, ner, relation, named                      |
|      9 | 多模态大模型：视觉语言理解与跨模态推理     |     31 |            700 |    4.43 | visual, language, multimodal, image, video                    |

### ACL 2023

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                                        |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------|
|      1 | 可信安全：对抗攻击、后门、水印与隐私           |    186 |           1075 |   17.3  | language, knowledge, adversarial, training, pre-trained       |
|      2 | 生成模型：扩散模型、采样与内容生成（生成编辑） |    137 |           1075 |   12.74 | language, generation, text, training, prompt                  |
|      3 | 事件视觉：事件相机、运动估计与时序感知         |    129 |           1075 |   12    | event, extraction, entity, relation, knowledge                |
|      4 | 语音音频：ASR、说话人与音频理解                |    120 |           1075 |   11.16 | translation, language, languages, speech, machine             |
|      5 | 大模型社会安全：偏见、虚假信息与检测           |    112 |           1075 |   10.42 | social, language, nlp, detection, biases                      |
|      6 | 大模型推理：问答、常识与思维链                 |     97 |           1075 |    9.02 | reasoning, language, knowledge, causal, counterfactual        |
|      7 | 信息检索：密集检索、向量召回与表示学习         |     79 |           1075 |    7.35 | question, answering, retrieval, question answering, questions |
|      8 | 多模态大模型：视觉语言理解与跨模态推理         |     75 |           1075 |    6.98 | visual, multimodal, image, cross-modal, language              |
|      9 | 生成模型：扩散模型、采样与内容生成（推荐排序） |     71 |           1075 |    6.6  | dialogue, dialogues, knowledge, generation, response          |
|     10 | 大模型评测：人类偏好、任务指标与领域评估       |     69 |           1075 |    6.42 | summarization, evaluation, metrics, summaries, human          |

### ACL 2024

|   排名 | 细主题名                                             |   篇数 |   当年会议篇数 |   占比% | 关键词                                                   |
|-------:|:-----------------------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------------|
|      1 | 大模型评测：人类偏好、任务指标与领域评估             |    150 |            940 |   15.96 | llms, language, evaluation, human, metrics               |
|      2 | 语言模型分析：认知、语言结构与可解释性（RAG问答）    |    128 |            940 |   13.62 | language, languages, llms, text, translation             |
|      3 | 检索增强大模型：RAG、知识注入与问答                  |    119 |            940 |   12.66 | llms, knowledge, retrieval, language, question           |
|      4 | 大模型推理：问答、常识与思维链                       |    112 |            940 |   11.91 | reasoning, llms, language, knowledge, event              |
|      5 | 大模型训练：微调、数据配方与任务适配（Agent规划）    |     83 |            940 |    8.83 | agents, dialogue, llms, language, planning               |
|      6 | 大模型训练：微调、数据配方与任务适配（LoRA联邦微调） |     82 |            940 |    8.72 | language, fine-tuning, memory, llms, parameters          |
|      7 | 多模态大模型：视觉语言理解与跨模态推理               |     82 |            940 |    8.72 | multimodal, visual, image, language, images              |
|      8 | 语言模型分析：认知、语言结构与可解释性（认知语言）   |     64 |            940 |    6.81 | language, in-context, llms, icl, training                |
|      9 | 可信安全：对抗攻击、后门、水印与隐私                 |     49 |            940 |    5.21 | safety, llms, attacks, language, text                    |
|     10 | 代码大模型：代码生成、程序理解与评测                 |     36 |            940 |    3.83 | code, instruction, generation, language, code generation |

### ACL 2025

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                                     |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------|
|      1 | 高效大模型：长上下文、注意力与推理优化   |    308 |           1699 |   18.13 | language, llms, training, llm, inference                   |
|      2 | 大模型推理：问答、常识与思维链           |    206 |           1699 |   12.12 | reasoning, llms, knowledge, language, llm                  |
|      3 | 大模型社会安全：偏见、虚假信息与检测     |    186 |           1699 |   10.95 | llms, language, bias, detection, social                    |
|      4 | 多模态大模型：视觉语言理解与跨模态推理   |    152 |           1699 |    8.95 | visual, multimodal, reasoning, mllms, language             |
|      5 | 大模型评测：人类偏好、任务指标与领域评估 |    141 |           1699 |    8.3  | evaluation, llms, human, legal, language                   |
|      6 | 语音音频：ASR、说话人与音频理解          |    140 |           1699 |    8.24 | language, languages, speech, multilingual, translation     |
|      7 | 检索增强大模型：RAG、知识注入与问答      |    127 |           1699 |    7.47 | retrieval, rag, generation, knowledge, retrieval-augmented |
|      8 | 语言模型分析：认知、语言结构与可解释性   |    126 |           1699 |    7.42 | agents, llms, agent, dialogue, planning                    |
|      9 | 可信安全：对抗攻击、后门、水印与隐私     |     98 |           1699 |    5.77 | safety, attacks, llms, attack, language                    |
|     10 | 代码大模型：代码生成、程序理解与评测     |     87 |           1699 |    5.12 | code, llms, language, generation, code generation          |

### ACMMM 2020

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 底层视觉：超分、去噪、增强与图像复原     |     87 |            473 |   18.39 | video, image, images, quality, streaming               |
|      2 | 多媒体安全：Deepfake检测、伪造识别与攻防 |     86 |            473 |   18.18 | adversarial, music, facial, recognition, emotion       |
|      3 | 视觉感知：目标检测、识别与视觉表征       |     83 |            473 |   17.55 | object, detection, features, person, re-identification |
|      4 | 迁移泛化：域适应、OOD泛化与鲁棒表征      |     60 |            473 |   12.68 | domain, semantic, image, visual, feature               |
|      5 | 视频理解：动作识别、长视频与时序建模     |     58 |            473 |   12.26 | video, temporal, videos, information, visual           |
|      6 | 高效大模型：推理加速、压缩与资源优化     |     58 |            473 |   12.26 | image, visual, text, information, attention            |
|      7 | 三维视觉：点云、深度估计与相机姿态       |     41 |            473 |    8.67 | pose, human, face, style, image                        |

### ACMMM 2021

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                           |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------------|
|      1 | 底层视觉：超分、去噪、增强与图像复原       |    116 |            542 |   21.4  | image, video, compression, quality, images                       |
|      2 | 多模态大模型：视觉语言理解与跨模态推理     |     97 |            542 |   17.9  | visual, text, image, language, graph                             |
|      3 | 三维视觉：点云、深度估计与相机姿态         |     61 |            542 |   11.25 | object, point, detection, features, feature                      |
|      4 | 视频理解：动作识别、长视频与时序建模       |     61 |            542 |   11.25 | video, action, temporal, motion, videos                          |
|      5 | 多模态理解：视觉语言表征与跨模态对齐       |     56 |            542 |   10.33 | music, visual, information, framework, emotion                   |
|      6 | 迁移泛化：域适应、OOD泛化与鲁棒表征        |     54 |            542 |    9.96 | domain, adaptation, segmentation, semantic, source               |
|      7 | 多媒体检索：跨模态检索、语义匹配与内容理解 |     40 |            542 |    7.38 | retrieval, image, clustering, multi-view, image retrieval        |
|      8 | 多媒体安全：Deepfake检测、伪造识别与攻防   |     32 |            542 |    5.9  | adversarial, attack, image, attacks, face                        |
|      9 | 图学习：GNN、节点分类与链接预测            |     25 |            542 |    4.61 | person, re-identification, person re-identification, reid, re-id |

### ACMMM 2022

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                            |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------------------|
|      1 | 底层视觉：超分、去噪、增强与图像复原       |    156 |            691 |   22.58 | image, video, quality, images, information                        |
|      2 | 多媒体检索：跨模态检索、语义匹配与内容理解 |    117 |            691 |   16.93 | visual, text, image, retrieval, information                       |
|      3 | 迁移泛化：小样本、类增量与域泛化           |     99 |            691 |   14.33 | domain, segmentation, target, adaptation, source                  |
|      4 | 多媒体安全：Deepfake检测、伪造识别与攻防   |     74 |            691 |   10.71 | face, image, images, generation, adversarial                      |
|      5 | 视频理解：动作识别、长视频与时序建模       |     67 |            691 |    9.7  | video, action, temporal, features, detection                      |
|      6 | 推荐系统：排序、召回与点击率预测           |     66 |            691 |    9.55 | multimodal, music, user, recognition, emotion                     |
|      7 | 三维视觉：点云、深度估计与相机姿态         |     59 |            691 |    8.54 | depth, point, human, estimation, pose                             |
|      8 | 理论机器学习：核方法、NTK与泛化分析        |     28 |            691 |    4.05 | clustering, hashing, multi-view, deep, views                      |
|      9 | 视觉感知：目标检测、识别与视觉表征         |     25 |            691 |    3.62 | person, re-identification, person re-identification, reid, images |

### ACMMM 2023

|   排名 | 细主题名                                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                                    |
|-------:|:---------------------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------------|
|      1 | 多媒体检索：跨模态检索、语义匹配与内容理解（搜索排序）   |    148 |            902 |   16.41 | text, knowledge, visual, image, retrieval                 |
|      2 | 视觉感知：目标检测、识别与视觉表征                       |    120 |            902 |   13.3  | image, images, information, features, feature             |
|      3 | 三维视觉：点云、深度估计与相机姿态                       |     82 |            902 |    9.09 | object, pose, point, objects, detection                   |
|      4 | 多模态理解：视觉语言表征与跨模态对齐                     |     81 |            902 |    8.98 | multimodal, emotion, speech, facial, audio                |
|      5 | 可信安全：对抗攻击、后门、水印与隐私                     |     80 |            902 |    8.87 | attack, adversarial, attacks, backdoor, images            |
|      6 | 底层视觉：超分、去噪、增强与图像复原                     |     73 |            902 |    8.09 | video, quality, compression, motion, videos               |
|      7 | 生成模型：文生图、扩散采样与图像编辑                     |     60 |            902 |    6.65 | image, generation, diffusion, images, style               |
|      8 | 视频理解：动作识别、长视频与时序建模                     |     58 |            902 |    6.43 | action, recognition, motion, temporal, action recognition |
|      9 | 迁移泛化：域适应、OOD泛化与鲁棒表征                      |     52 |            902 |    5.76 | domain, target, adaptation, source, domains               |
|     10 | 多媒体检索：跨模态检索、语义匹配与内容理解（搜索排序-2） |     51 |            902 |    5.65 | video, temporal, visual, retrieval, videos                |

### ACMMM 2024

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------|
|      1 | 多模态大模型：视觉语言理解与跨模态推理     |    170 |           1149 |   14.8  | visual, multimodal, text, information, language       |
|      2 | 生成模型：视频扩散生成与编辑               |    154 |           1149 |   13.4  | image, images, information, video, diffusion          |
|      3 | 三维视觉：点云、深度估计与相机姿态         |    125 |           1149 |   10.88 | point, point cloud, cloud, rendering, scene           |
|      4 | 图学习：图聚类、表示学习与结构匹配         |    116 |           1149 |   10.1  | graph, multi-view, clustering, knowledge, information |
|      5 | 视频语言理解：视频检索、时刻定位与时序推理 |    105 |           1149 |    9.14 | video, action, motion, temporal, information          |
|      6 | 多模态音视频生成：语音、音乐与情感生成     |    104 |           1149 |    9.05 | audio, speech, motion, generation, video              |
|      7 | 医疗视觉：医学影像分割、病理与临床影像     |     99 |           1149 |    8.62 | segmentation, domain, medical, images, features       |
|      8 | 底层视觉：超分、去噪、增强与图像复原       |     98 |           1149 |    8.53 | image, generation, diffusion, quality, images         |
|      9 | 多模态理解：视觉语言表征与跨模态对齐       |     93 |           1149 |    8.09 | multimodal, emotion, facial, information, modalities  |
|     10 | 多媒体安全：Deepfake检测、伪造识别与攻防   |     85 |           1149 |    7.4  | adversarial, attacks, attack, images, image           |

### ACMMM 2025

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                      |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------------|
|      1 | 多媒体检索：跨模态检索、语义匹配与内容理解 |    197 |           1249 |   15.77 | knowledge, multimodal, retrieval, semantic, framework       |
|      2 | 三维视觉：点云、深度估计与相机姿态         |    163 |           1249 |   13.05 | gaussian, scene, reconstruction, point, splatting           |
|      3 | 多模态音视频生成：语音、音乐与情感生成     |    162 |           1249 |   12.97 | motion, emotion, generation, multimodal, audio              |
|      4 | 生成模型：视频扩散生成与编辑               |    143 |           1249 |   11.45 | image, diffusion, video, fusion, feature                    |
|      5 | 多媒体安全：Deepfake检测、伪造识别与攻防   |    131 |           1249 |   10.49 | detection, anomaly, deepfake, adversarial, image            |
|      6 | 多模态大模型：视觉语言理解与跨模态推理     |    109 |           1249 |    8.73 | visual, lvlms, vision-language, language, reasoning         |
|      7 | 多模态理解：视觉语言表征与跨模态对齐       |    107 |           1249 |    8.57 | video, temporal, videos, action, event                      |
|      8 | 生成模型：文生图、扩散采样与图像编辑       |     95 |           1249 |    7.61 | generation, image, images, editing, visual                  |
|      9 | 图学习：图聚类、表示学习与结构匹配         |     73 |           1249 |    5.84 | clustering, graph, multi-view, multi-view clustering, views |
|     10 | 医疗视觉：医学影像分割、病理与临床影像     |     69 |           1249 |    5.52 | medical, segmentation, image, clinical, medical image       |

### CVPR 2020

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                                  |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------|
|      1 | 三维视觉：点云、深度估计与相机姿态（视频时序） |    234 |           1466 |   15.96 | image, images, depth, deep, estimation                  |
|      2 | 视觉感知：目标检测、识别与视觉表征（推理效率） |    198 |           1466 |   13.51 | image, person, visual, features, recognition            |
|      3 | 视觉感知：语义/实例/全景分割                   |    171 |           1466 |   11.66 | segmentation, object, detection, instance, semantic     |
|      4 | 三维视觉：点云、深度估计与相机姿态（推理问答） |    156 |           1466 |   10.64 | point, shape, object, cloud, point cloud                |
|      5 | 人体与头像生成：姿态、表情与数字人合成         |    135 |           1466 |    9.21 | image, images, face, generative, generation             |
|      6 | 自动机器学习：神经架构搜索与超参数优化         |    132 |           1466 |    9    | search, architecture, training, accuracy, architectures |
|      7 | 视频理解：动作识别、长视频与时序建模           |    110 |           1466 |    7.5  | video, action, temporal, recognition, videos            |
|      8 | 三维视觉：点云、深度估计与相机姿态（检测识别） |    107 |           1466 |    7.3  | pose, human, estimation, pose estimation, object        |
|      9 | 视觉感知：目标检测、识别与视觉表征（视觉语言） |     86 |           1466 |    5.87 | tracking, visual, prediction, object, driving           |
|     10 | 可信安全：对抗攻击、后门、水印与隐私           |     85 |           1466 |    5.8  | adversarial, attacks, attack, robustness, training      |

### CVPR 2021

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                                      |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------------|
|      1 | 三维视觉：点云、深度估计与相机姿态（检测识别） |    304 |           1660 |   18.31 | point, depth, object, cloud, point cloud                    |
|      2 | 视觉感知：语义/实例/全景分割                   |    189 |           1660 |   11.39 | segmentation, object, detection, object detection, instance |
|      3 | 底层视觉：超分、去噪、增强与图像复原           |    172 |           1660 |   10.36 | image, images, rain, deep, super-resolution                 |
|      4 | 鲁棒泛化：OOD检测、校准与噪声标签              |    167 |           1660 |   10.06 | training, classification, labels, knowledge, loss           |
|      5 | 视频理解：动作识别、长视频与时序建模           |    156 |           1660 |    9.4  | video, action, temporal, videos, recognition                |
|      6 | 生成模型：文生图、扩散采样与图像编辑           |    124 |           1660 |    7.47 | image, style, images, generative, latent                    |
|      7 | 多模态大模型：视觉语言理解与跨模态推理         |    114 |           1660 |    6.87 | text, visual, scene, reasoning, image                       |
|      8 | 三维视觉：点云、深度估计与相机姿态（视频时序） |    105 |           1660 |    6.33 | human, pose, body, motion, estimation                       |
|      9 | 图学习：GNN、节点分类与链接预测                |     99 |           1660 |    5.96 | search, architecture, graph, compression, training          |
|     10 | 多媒体安全：Deepfake检测、伪造识别与攻防       |     97 |           1660 |    5.84 | face, adversarial, attack, attacks, recognition             |

### CVPR 2022

|   排名 | 细主题名                                           |   篇数 |   当年会议篇数 |   占比% | 关键词                                              |
|-------:|:---------------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------|
|      1 | 迁移泛化：域适应、OOD泛化与鲁棒表征                |    400 |           2074 |   19.29 | segmentation, domain, object, semantic, detection   |
|      2 | 多模态大模型：视觉语言理解与跨模态推理             |    240 |           2074 |   11.57 | visual, text, image, language, video                |
|      3 | 三维视觉：点云、深度估计与相机姿态（检测识别）     |    230 |           2074 |   11.09 | point, shape, cloud, point cloud, object            |
|      4 | 三维视觉：点云、深度估计与相机姿态（视频时序）     |    193 |           2074 |    9.31 | pose, human, face, person, motion                   |
|      5 | 高效大模型：推理加速、压缩与资源优化               |    189 |           2074 |    9.11 | vision, transformer, image, transformers, training  |
|      6 | 视频理解：动作识别、长视频与时序建模（视频时序）   |    181 |           2074 |    8.73 | image, video, images, flow, motion                  |
|      7 | 三维视觉：点云、深度估计与相机姿态（三维几何）     |    170 |           2074 |    8.2  | depth, scene, rendering, stereo, novel              |
|      8 | 视频理解：动作识别、长视频与时序建模（视频时序-2） |    139 |           2074 |    6.7  | video, action, temporal, videos, recognition        |
|      9 | 具身智能：机器人操作、导航与视觉语言动作           |     93 |           2074 |    4.48 | tracking, prediction, detection, object, trajectory |
|     10 | 生成模型：文生图、扩散采样与图像编辑               |     83 |           2074 |    4    | image, images, gan, editing, generative             |

### CVPR 2023

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                              |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------|
|      1 | 视觉感知：语义/实例/全景分割                   |    291 |           2353 |   12.37 | segmentation, semantic, object, detection, training |
|      2 | 自动驾驶感知：LiDAR、轨迹与三维检测            |    282 |           2353 |   11.98 | point, object, detection, cloud, point cloud        |
|      3 | 三维视觉：点云、深度估计与相机姿态（分割）     |    275 |           2353 |   11.69 | scene, rendering, radiance, reconstruction, depth   |
|      4 | 多模态大模型：视觉语言理解与跨模态推理         |    261 |           2353 |   11.09 | visual, image, language, training, text             |
|      5 | 可信安全：对抗攻击、后门、水印与隐私           |    259 |           2353 |   11.01 | domain, adversarial, training, knowledge, attacks   |
|      6 | 三维视觉：点云、深度估计与相机姿态（视频时序） |    248 |           2353 |   10.54 | human, pose, face, facial, motion                   |
|      7 | 视频理解：动作识别、长视频与时序建模           |    201 |           2353 |    8.54 | video, action, temporal, videos, recognition        |
|      8 | 底层视觉：超分、去噪、增强与图像复原           |    188 |           2353 |    7.99 | image, images, video, motion, super-resolution      |
|      9 | 生成模型：文生图、扩散采样与图像编辑           |    157 |           2353 |    6.67 | diffusion, generation, image, images, latent        |
|     10 | 高效大模型：推理加速、压缩与资源优化           |    156 |           2353 |    6.63 | vision, image, masked, attention, transformers      |

### CVPR 2024

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                                  |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------|
|      1 | 三维视觉：点云、深度估计与相机姿态             |    373 |           2716 |   13.73 | pose, point, object, estimation, depth                  |
|      2 | 可信安全：对抗攻击、后门、水印与隐私           |    363 |           2716 |   13.37 | training, adversarial, domain, knowledge, existing      |
|      3 | 生成模型：三维生成、新视角与Gaussian Splatting |    360 |           2716 |   13.25 | diffusion, generation, human, reconstruction, rendering |
|      4 | 三维视觉：Gaussian Splatting、新视角合成与重建 |    321 |           2716 |   11.82 | image, images, reconstruction, information, real-world  |
|      5 | 多模态大模型：视觉语言理解与跨模态推理         |    319 |           2716 |   11.75 | visual, language, multimodal, image, vision-language    |
|      6 | 生成模型：文生图、扩散采样与图像编辑           |    278 |           2716 |   10.24 | diffusion, image, images, generation, text-to-image     |
|      7 | 医疗视觉：医学影像分割、病理与临床影像         |    228 |           2716 |    8.39 | segmentation, semantic, image, object, medical          |
|      8 | 具身智能：机器人操作、导航与视觉语言动作       |    169 |           2716 |    6.22 | motion, human, interaction, hand, driving               |
|      9 | 生成模型：视频扩散生成与编辑                   |    162 |           2716 |    5.96 | video, motion, generation, videos, diffusion            |
|     10 | 异常检测：图异常、欺诈检测与时序异常           |    143 |           2716 |    5.27 | video, action, detection, anomaly, videos               |

### CVPR 2025

|   排名 | 细主题名                                           |   篇数 |   当年会议篇数 |   占比% | 关键词                                                        |
|-------:|:---------------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------|
|      1 | 生成模型：文生图、扩散采样与图像编辑               |    458 |           2871 |   15.95 | image, diffusion, generation, images, training                |
|      2 | 三维视觉：点云、深度估计与相机姿态                 |    413 |           2871 |   14.39 | object, depth, estimation, point, pose                        |
|      3 | 多模态大模型：视觉语言理解与跨模态推理（视觉语言） |    349 |           2871 |   12.16 | visual, multimodal, language, image, vision-language          |
|      4 | 生成模型：视频扩散生成与编辑                       |    276 |           2871 |    9.61 | video, generation, diffusion, motion, videos                  |
|      5 | 具身智能：机器人操作、导航与视觉语言动作           |    253 |           2871 |    8.81 | motion, human, interaction, scene, hand                       |
|      6 | 多模态音视频生成：语音、音乐与情感生成             |    197 |           2871 |    6.86 | human, audio, generation, facial, head                        |
|      7 | 多模态大模型：视觉语言理解与跨模态推理（搜索排序） |    196 |           2871 |    6.83 | video, temporal, understanding, videos, benchmark             |
|      8 | 迁移泛化：域适应、OOD泛化与鲁棒表征                |    170 |           2871 |    5.92 | domain, generalization, features, knowledge, training         |
|      9 | 高效大模型：推理加速、压缩与资源优化               |    169 |           2871 |    5.89 | vision, mamba, compression, image, event                      |
|     10 | 三维视觉：Gaussian Splatting、新视角合成与重建     |    164 |           2871 |    5.71 | gaussian, rendering, splatting, gaussian splatting, gaussians |

### ECCV 2020

|   排名 | 细主题名                                         |   篇数 |   当年会议篇数 |   占比% | 关键词                                                           |
|-------:|:-------------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------------|
|      1 | 三维视觉：点云、深度估计与相机姿态（检测识别）   |    245 |           1358 |   18.04 | point, object, shape, depth, images                              |
|      2 | 迁移泛化：域适应、OOD泛化与鲁棒表征              |    190 |           1358 |   13.99 | domain, training, adaptation, domain adaptation, classes         |
|      3 | 视觉感知：语义/实例/全景分割                     |    187 |           1358 |   13.77 | object, segmentation, detection, object detection, feature       |
|      4 | 视频理解：动作识别、长视频与时序建模（视频时序） |    163 |           1358 |   12    | video, action, videos, temporal, motion                          |
|      5 | 视频理解：动作识别、长视频与时序建模（搜索排序） |    157 |           1358 |   11.56 | image, images, video, deep, flow                                 |
|      6 | 人体与头像生成：姿态、表情与数字人合成           |    109 |           1358 |    8.03 | image, images, generation, facial, generative                    |
|      7 | 多模态大模型：视觉语言理解与跨模态推理           |     74 |           1358 |    5.45 | visual, image, graph, text, scene                                |
|      8 | 自动机器学习：神经架构搜索与超参数优化           |     69 |           1358 |    5.08 | search, architecture, quantization, architecture search, pruning |
|      9 | 三维视觉：点云、深度估计与相机姿态（搜索排序）   |     68 |           1358 |    5.01 | pose, human, estimation, pose estimation, hand                   |
|     10 | 多媒体安全：Deepfake检测、伪造识别与攻防         |     68 |           1358 |    5.01 | adversarial, face, attacks, attack, deep                         |

### ECCV 2022

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                               |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------|
|      1 | 迁移泛化：域适应、OOD泛化与鲁棒表征            |    377 |           1645 |   22.92 | domain, segmentation, training, knowledge, detection |
|      2 | 三维视觉：点云、深度估计与相机姿态（搜索排序） |    361 |           1645 |   21.95 | point, depth, object, estimation, cloud              |
|      3 | 视频理解：动作识别、长视频与时序建模           |    216 |           1645 |   13.13 | video, temporal, action, videos, frames              |
|      4 | 视觉感知：目标检测、识别与视觉表征             |    183 |           1645 |   11.12 | text, vision, transformer, image, visual             |
|      5 | 三维视觉：Gaussian Splatting、新视角合成与重建 |    139 |           1645 |    8.45 | image, images, scene, rendering, radiance            |
|      6 | 生成模型：文生图、扩散采样与图像编辑           |    131 |           1645 |    7.96 | image, images, generation, style, generative         |
|      7 | 可信安全：对抗攻击、后门、水印与隐私           |    121 |           1645 |    7.36 | adversarial, training, search, pruning, accuracy     |
|      8 | 三维视觉：点云、深度估计与相机姿态（视频时序） |    117 |           1645 |    7.11 | human, pose, motion, body, poses                     |

### ECCV 2024

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                              |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------|
|      1 | 开放词汇视觉：开放词汇检测、分割与CLIP语义     |    373 |           2387 |   15.63 | segmentation, visual, image, semantic, object       |
|      2 | 自动驾驶感知：LiDAR、轨迹与三维检测            |    363 |           2387 |   15.21 | object, point, detection, driving, features         |
|      3 | 三维视觉：Gaussian Splatting、新视角合成与重建 |    342 |           2387 |   14.33 | rendering, scene, gaussian, reconstruction, novel   |
|      4 | 可信安全：对抗攻击、后门、水印与隐私           |    312 |           2387 |   13.07 | training, domain, detection, adversarial, knowledge |
|      5 | 生成模型：文生图、扩散采样与图像编辑           |    269 |           2387 |   11.27 | diffusion, image, generation, images, text          |
|      6 | 底层视觉：超分、去噪、增强与图像复原           |    238 |           2387 |    9.97 | image, vision, restoration, training, images        |
|      7 | 三维视觉：点云、深度估计与相机姿态             |    237 |           2387 |    9.93 | motion, human, video, generation, pose              |
|      8 | 多模态大模型：视觉语言理解与跨模态推理         |    206 |           2387 |    8.63 | video, action, videos, temporal, visual             |
|      9 | 事件视觉：事件相机、运动估计与时序感知         |     47 |           2387 |    1.97 | event, event-based, events, temporal, cameras       |

### EMNLP 2020

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                              |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------------|
|      1 | 开放词汇视觉：开放词汇检测、分割与CLIP语义 |    141 |            751 |   18.77 | language, word, representations, semantic, parsing                  |
|      2 | 多语言NLP：机器翻译、跨语言与低资源        |    118 |            751 |   15.71 | translation, languages, multilingual, language, machine translation |
|      3 | 事件视觉：事件相机、运动估计与时序感知     |    108 |            751 |   14.38 | knowledge, graph, entity, event, entities                           |
|      4 | 大模型推理：问答、常识与思维链             |     95 |            751 |   12.65 | question, questions, answering, question answering, reasoning       |
|      5 | 大模型社会安全：偏见、虚假信息与检测       |     74 |            751 |    9.85 | sentiment, bias, social, aspect, gender                             |
|      6 | 可信安全：对抗攻击、后门、水印与隐私       |     66 |            751 |    8.79 | generation, text, adversarial, language, story                      |
|      7 | 强化学习：离线策略、奖励建模与控制         |     61 |            751 |    8.12 | dialogue, dialog, response, responses, knowledge                    |
|      8 | 迁移泛化：域适应、OOD泛化与鲁棒表征        |     51 |            751 |    6.79 | language, bert, pre-training, domain, text                          |
|      9 | 文本生成：摘要、事实性与可控生成           |     37 |            751 |    4.93 | summarization, abstractive, summary, summaries, text                |

### EMNLP 2021

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                                                 |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------------------|
|      1 | 生成模型：扩散模型、采样与内容生成       |    168 |            847 |   19.83 | training, language, generation, transformer, text                      |
|      2 | 视觉感知：目标检测、定位与分割           |    152 |            847 |   17.95 | language, sentiment, text, information, word                           |
|      3 | 信息检索：密集检索、向量召回与表示学习   |    132 |            847 |   15.58 | question, knowledge, reasoning, retrieval, answering                   |
|      4 | 知识图谱：实体关系、推理与补全           |    106 |            847 |   12.51 | relation, entity, extraction, text, information                        |
|      5 | 多语言NLP：机器翻译、跨语言与低资源      |    101 |            847 |   11.92 | translation, multilingual, cross-lingual, machine translation, machine |
|      6 | 对话系统：响应生成、情感支持与任务型对话 |     81 |            847 |    9.56 | dialogue, dialog, knowledge, conversation, conversational              |
|      7 | 事件视觉：事件相机、运动估计与时序感知   |     44 |            847 |    5.19 | event, summarization, events, coreference, temporal                    |
|      8 | 可信安全：对抗攻击、后门、水印与隐私     |     34 |            847 |    4.01 | adversarial, attacks, attack, robustness, examples                     |
|      9 | 多模态大模型：视觉语言理解与跨模态推理   |     29 |            847 |    3.42 | visual, multimodal, image, images, grounding                           |

### EMNLP 2022

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                                              |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------------|
|      1 | 可信安全：对抗攻击、后门、水印与隐私     |    150 |            828 |   18.12 | language, training, text, adversarial, pre-trained                  |
|      2 | 检索增强大模型：RAG、知识注入与问答      |    132 |            828 |   15.94 | question, reasoning, retrieval, answering, language                 |
|      3 | 大模型社会安全：偏见、虚假信息与检测     |    123 |            828 |   14.86 | language, sentiment, bias, social, detection                        |
|      4 | 高效大模型：解码加速、KV缓存与推理优化   |    112 |            828 |   13.53 | translation, languages, language, multilingual, machine translation |
|      5 | 知识图谱：实体关系、推理与补全           |     93 |            828 |   11.23 | entity, extraction, knowledge, relation, information                |
|      6 | 大模型评测：人类偏好、任务指标与领域评估 |     81 |            828 |    9.78 | summarization, generation, text, summaries, evaluation              |
|      7 | 迁移泛化：域适应、OOD泛化与鲁棒表征      |     47 |            828 |    5.68 | language, compositional, knowledge, generalization, commonsense     |
|      8 | 鲁棒泛化：OOD检测、校准与噪声标签        |     46 |            828 |    5.56 | dialogue, dialog, knowledge, systems, ood                           |
|      9 | 多模态理解：视觉语言表征与跨模态对齐     |     44 |            828 |    5.31 | image, multimodal, video, visual, text                              |

### EMNLP 2023

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                              |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------------|
|      1 | 大模型社会安全：偏见、虚假信息与检测       |    200 |           1047 |   19.1  | language, social, llms, human, bias                                 |
|      2 | 代码大模型：代码生成、程序理解与评测       |    194 |           1047 |   18.53 | language, training, llms, code, prompt                              |
|      3 | 开放词汇视觉：开放词汇检测、分割与CLIP语义 |    138 |           1047 |   13.18 | entity, knowledge, extraction, event, semantic                      |
|      4 | 大模型评测：人类偏好、任务指标与领域评估   |    121 |           1047 |   11.56 | text, language, evaluation, summarization, generation               |
|      5 | 大模型推理：问答、常识与思维链             |    120 |           1047 |   11.46 | reasoning, language, llms, question, knowledge                      |
|      6 | 大模型训练：微调、数据配方与任务适配       |     91 |           1047 |    8.69 | translation, languages, language, multilingual, machine translation |
|      7 | 多模态大模型：视觉语言理解与跨模态推理     |     81 |           1047 |    7.74 | visual, image, language, text, multimodal                           |
|      8 | 生成模型：扩散模型、采样与内容生成         |     70 |           1047 |    6.69 | dialogue, chatgpt, generation, conversational, knowledge            |
|      9 | 信息检索：搜索排序、文档检索与重排         |     32 |           1047 |    3.06 | retrieval, document, documents, dense, query                        |

### EMNLP 2024

|   排名 | 细主题名                                         |   篇数 |   当年会议篇数 |   占比% | 关键词                                               |
|-------:|:-------------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------|
|      1 | 大模型训练：微调、数据配方与任务适配（语音音频） |    192 |           1268 |   15.14 | language, languages, text, translation, multilingual |
|      2 | 代码大模型：代码生成、程序理解与评测             |    155 |           1268 |   12.22 | llms, language, llm, code, dialogue                  |
|      3 | 检索增强大模型：RAG、知识注入与问答              |    148 |           1268 |   11.67 | knowledge, retrieval, llms, language, generation     |
|      4 | 多模态大模型：视觉语言理解与跨模态推理           |    146 |           1268 |   11.51 | visual, multimodal, image, language, images          |
|      5 | 大模型训练：微调、数据配方与任务适配（生成编辑） |    142 |           1268 |   11.2  | language, training, llms, fine-tuning, editing       |
|      6 | 大模型社会安全：偏见、虚假信息与检测             |    140 |           1268 |   11.04 | language, social, llms, bias, biases                 |
|      7 | 大模型推理：问答、常识与思维链                   |    136 |           1268 |   10.73 | reasoning, llms, language, logical, llm              |
|      8 | 可信安全：对抗攻击、后门、水印与隐私             |     70 |           1268 |    5.52 | attacks, llms, language, attack, adversarial         |
|      9 | 语言模型分析：认知、语言结构与可解释性           |     66 |           1268 |    5.21 | language, icl, in-context, context, llms             |
|     10 | 强化学习：策略优化、奖励学习与控制               |     42 |           1268 |    3.31 | preference, alignment, reward, human, preferences    |

### EMNLP 2025

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                             |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------|
|      1 | 检索增强大模型：RAG、知识注入与问答      |    254 |           1809 |   14.04 | retrieval, knowledge, rag, language, llms          |
|      2 | 大模型训练：微调、数据配方与任务适配     |    251 |           1809 |   13.88 | language, llms, training, fine-tuning, llm         |
|      3 | 大模型推理：问答、常识与思维链           |    244 |           1809 |   13.49 | reasoning, llms, language, cot, llm                |
|      4 | 语音音频：ASR、说话人与音频理解          |    239 |           1809 |   13.21 | language, languages, multilingual, speech, llms    |
|      5 | 大模型社会安全：偏见、虚假信息与检测     |    229 |           1809 |   12.66 | llms, language, bias, human, llm                   |
|      6 | 多模态大模型：视觉语言理解与跨模态推理   |    223 |           1809 |   12.33 | visual, multimodal, reasoning, video, language     |
|      7 | 代码大模型：代码生成、程序理解与评测     |    170 |           1809 |    9.4  | llms, language, code, agents, llm                  |
|      8 | 可信安全：对抗攻击、后门、水印与隐私     |    166 |           1809 |    9.18 | safety, llms, language, detection, attacks         |
|      9 | 医疗大模型：临床推理、医学影像与健康问答 |     33 |           1809 |    1.82 | medical, clinical, patient, evaluation, healthcare |

### ICCV 2021

|   排名 | 细主题名                                           |   篇数 |   当年会议篇数 |   占比% | 关键词                                                           |
|-------:|:---------------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------------|
|      1 | 三维视觉：点云、深度估计与相机姿态（检测识别）     |    241 |           1612 |   14.95 | point, depth, cloud, point cloud, object                         |
|      2 | 视频理解：动作识别、长视频与时序建模（视频时序）   |    225 |           1612 |   13.96 | image, images, video, scene, novel                               |
|      3 | 迁移泛化：小样本、类增量与域泛化                   |    184 |           1612 |   11.41 | domain, target, classes, training, adaptation                    |
|      4 | 视觉感知：语义/实例/全景分割                       |    156 |           1612 |    9.68 | segmentation, semantic, semantic segmentation, object, detection |
|      5 | 视频理解：动作识别、长视频与时序建模（视频时序-2） |    152 |           1612 |    9.43 | video, action, temporal, contrastive, self-supervised            |
|      6 | 三维视觉：点云、深度估计与相机姿态（视频时序）     |    127 |           1612 |    7.88 | pose, human, motion, estimation, body                            |
|      7 | 多模态大模型：视觉语言理解与跨模态推理             |    117 |           1612 |    7.26 | visual, graph, scene, knowledge, image                           |
|      8 | 人体与头像生成：姿态、表情与数字人合成             |    101 |           1612 |    6.27 | image, style, images, face, latent                               |
|      9 | 可信安全：对抗攻击、后门、水印与隐私               |     77 |           1612 |    4.78 | adversarial, attacks, attack, robustness, examples               |
|     10 | 具身智能：机器人操作、导航与视觉语言动作           |     65 |           1612 |    4.03 | navigation, agent, prediction, agents, trajectory                |

### ICCV 2023

|   排名 | 细主题名                                           |   篇数 |   当年会议篇数 |   占比% | 关键词                                              |
|-------:|:---------------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------|
|      1 | 迁移泛化：域适应、OOD泛化与鲁棒表征                |    380 |           2156 |   17.63 | domain, segmentation, training, detection, image    |
|      2 | 三维视觉：点云、深度估计与相机姿态（分割）         |    348 |           2156 |   16.14 | point, object, depth, detection, cloud              |
|      3 | 三维视觉：点云、深度估计与相机姿态（视频时序）     |    222 |           2156 |   10.3  | human, pose, face, image, images                    |
|      4 | 视觉感知：语义/实例/全景分割                       |    191 |           2156 |    8.86 | video, temporal, action, videos, segmentation       |
|      5 | 多模态大模型：视觉语言理解与跨模态推理（视觉语言） |    169 |           2156 |    7.84 | visual, text, language, vision-language, image      |
|      6 | 高效大模型：推理加速、压缩与资源优化               |    162 |           2156 |    7.51 | vision, accuracy, training, attention, image        |
|      7 | 三维视觉：Gaussian Splatting、新视角合成与重建     |    161 |           2156 |    7.47 | nerf, scene, radiance, rendering, fields            |
|      8 | 生成模型：文生图、扩散采样与图像编辑               |    158 |           2156 |    7.33 | diffusion, image, generation, images, text-to-image |
|      9 | 底层视觉：超分、去噪、增强与图像复原               |    139 |           2156 |    6.45 | image, images, restoration, super-resolution, noise |
|     10 | 可信安全：对抗攻击、后门、水印与隐私               |     91 |           2156 |    4.22 | adversarial, attacks, attack, federated, robustness |

### ICCV 2025

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                                             |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------------------|
|      1 | 多模态大模型：视觉语言理解与跨模态推理         |    460 |           2701 |   17.03 | visual, multimodal, reasoning, language, mllms                     |
|      2 | 三维视觉：点云、深度估计与相机姿态             |    373 |           2701 |   13.81 | point, depth, estimation, object, pose                             |
|      3 | 生成模型：视频扩散生成与编辑                   |    341 |           2701 |   12.62 | motion, video, generation, diffusion, human                        |
|      4 | 底层视觉：超分、去噪、增强与图像复原           |    253 |           2701 |    9.37 | image, diffusion, images, restoration, existing                    |
|      5 | 生成模型：文生图、扩散采样与图像编辑           |    252 |           2701 |    9.33 | image, diffusion, generation, editing, text                        |
|      6 | 异常检测：图异常、欺诈检测与时序异常           |    236 |           2701 |    8.74 | adversarial, detection, attacks, attack, anomaly                   |
|      7 | 迁移泛化：域适应、OOD泛化与鲁棒表征            |    230 |           2701 |    8.52 | segmentation, domain, semantic, object, detection                  |
|      8 | 具身智能：机器人操作、导航与视觉语言动作       |    227 |           2701 |    8.4  | driving, autonomous, prediction, navigation, autonomous driving    |
|      9 | 三维视觉：Gaussian Splatting、新视角合成与重建 |    178 |           2701 |    6.59 | gaussian, reconstruction, splatting, gaussian splatting, rendering |
|     10 | 多模态理解：视觉语言表征与跨模态对齐           |    118 |           2701 |    4.37 | video, temporal, videos, understanding, video understanding        |

### ICLR 2020

|   排名 | 细主题名                               |   篇数 |   当年会议篇数 |   占比% | 关键词                                                       |
|-------:|:---------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------------|
|      1 | 强化学习：策略优化、奖励学习与控制     |    136 |            687 |   19.8  | reinforcement, policy, agents, agent, control                |
|      2 | 迁移泛化：域适应、OOD泛化与鲁棒表征    |    123 |            687 |   17.9  | deep, gradient, training, optimization, generalization       |
|      3 | 鲁棒泛化：OOD检测、校准与噪声标签      |     83 |            687 |   12.08 | representation, information, few-shot, meta-learning, labels |
|      4 | 可信安全：对抗攻击、后门、水印与隐私   |     80 |            687 |   11.64 | adversarial, robustness, training, robust, attacks           |
|      5 | 生成模型：扩散模型、采样与内容生成     |     78 |            687 |   11.35 | language, text, natural, natural language, generation        |
|      6 | 图学习：GNN、节点分类与链接预测        |     70 |            687 |   10.19 | graph, graphs, node, embedding, knowledge                    |
|      7 | 生成模型：视频扩散生成与编辑           |     69 |            687 |   10.04 | generative, video, object, image, objects                    |
|      8 | 自动机器学习：神经架构搜索与超参数优化 |     48 |            687 |    6.99 | search, architecture, compression, nas, training             |

### ICLR 2021

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                       |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------------|
|      1 | 优化理论：随机/非凸优化与收敛率            |    198 |            860 |   23.02 | training, deep, optimization, gradient, convergence          |
|      2 | 强化学习：离线策略、奖励建模与控制         |    140 |            860 |   16.28 | reinforcement, policy, agent, control, agents                |
|      3 | 生成模型：文生图、扩散采样与图像编辑       |    115 |            860 |   13.37 | generative, image, images, latent, training                  |
|      4 | 视觉感知：目标检测、识别与视觉表征         |    114 |            860 |   13.26 | representation, classification, training, contrastive, image |
|      5 | 概率机器学习：变分推断、不确定性与后验建模 |     94 |            860 |   10.93 | language, text, training, natural, translation               |
|      6 | 科学计算：神经算子、PDE与物理建模          |     80 |            860 |    9.3  | time, memory, recurrent, continual, systems                  |
|      7 | 可信安全：对抗攻击、后门、水印与隐私       |     74 |            860 |    8.6  | adversarial, robustness, training, attacks, robust           |
|      8 | 图学习：GNN、节点分类与链接预测            |     45 |            860 |    5.23 | graph, graphs, gnns, node, gnn                               |

### ICLR 2022

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                                      |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------------|
|      1 | 优化理论：梯度方法、收敛性与训练动力学   |    224 |           1094 |   20.48 | training, optimization, deep, gradient, kernel              |
|      2 | 强化学习：离线策略、奖励建模与控制       |    199 |           1094 |   18.19 | reinforcement, policy, offline, reward, agent               |
|      3 | 生成模型：文生图、扩散采样与图像编辑     |    149 |           1094 |   13.62 | generative, latent, time, variational, inference            |
|      4 | 迁移泛化：域适应、OOD泛化与鲁棒表征      |    140 |           1094 |   12.8  | domain, training, distribution, contrastive, classification |
|      5 | 高效大模型：参数高效微调与LoRA适配       |    109 |           1094 |    9.96 | language, continual, transformer, natural, training         |
|      6 | AI4Science：分子生成、药物发现与化学建模 |     95 |           1094 |    8.68 | graph, gnns, node, graphs, gnn                              |
|      7 | 视觉感知：目标检测、识别与视觉表征       |     90 |           1094 |    8.23 | vision, image, object, detection, transformers              |
|      8 | 可信安全：对抗攻击、后门、水印与隐私     |     88 |           1094 |    8.04 | adversarial, training, robustness, attacks, federated       |

### ICLR 2023

|   排名 | 细主题名                                          |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:--------------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 视觉感知：目标检测、识别与视觉表征                |    281 |           1573 |   17.86 | image, vision, visual, representation, contrastive     |
|      2 | 强化学习：离线策略、奖励建模与控制                |    238 |           1573 |   15.13 | reinforcement, policy, offline, agents, reward         |
|      3 | 迁移泛化：域适应、OOD泛化与鲁棒表征（社会安全）   |    223 |           1573 |   14.18 | training, optimization, gradient, deep, generalization |
|      4 | 大模型推理：问答、常识与思维链                    |    167 |           1573 |   10.62 | language, reasoning, training, knowledge, generation   |
|      5 | 迁移泛化：域适应、OOD泛化与鲁棒表征（社会安全-2） |    144 |           1573 |    9.15 | domain, training, label, adaptation, distribution      |
|      6 | 生成模型：文生图、扩散采样与图像编辑              |    135 |           1573 |    8.58 | diffusion, image, generative, generation, process      |
|      7 | 时序建模：时间序列预测、动力系统与基础模型        |    126 |           1573 |    8.01 | causal, time, latent, series, time series              |
|      8 | 可信安全：对抗攻击、后门、水印与隐私              |    117 |           1573 |    7.44 | federated, adversarial, training, robustness, privacy  |
|      9 | 图学习：GNN、节点分类与链接预测                   |    101 |           1573 |    6.42 | graph, gnns, graphs, node, gnn                         |
|     10 | AI4Science：蛋白结构、序列与功能建模              |     41 |           1573 |    2.61 | protein, molecular, molecules, drug, graph             |

### ICLR 2024

|   排名 | 细主题名                                        |   篇数 |   当年会议篇数 |   占比% | 关键词                                                     |
|-------:|:------------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------|
|      1 | 大模型推理：问答、常识与思维链                  |    493 |           2260 |   21.81 | language, llms, training, reasoning, llm                   |
|      2 | 强化学习：离线策略、奖励建模与控制              |    342 |           2260 |   15.13 | reinforcement, policy, reward, offline, agents             |
|      3 | 生成模型：视频扩散生成与编辑                    |    297 |           2260 |   13.14 | diffusion, image, generation, generative, images           |
|      4 | 多模态大模型：视觉语言理解与跨模态推理          |    283 |           2260 |   12.52 | image, visual, representation, object, vision              |
|      5 | 迁移泛化：域适应、OOD泛化与鲁棒表征（检测识别） |    281 |           2260 |   12.43 | training, deep, ood, generalization, features              |
|      6 | 可信安全：对抗攻击、后门、水印与隐私            |    229 |           2260 |   10.13 | federated, adversarial, optimization, attacks, training    |
|      7 | AI4Science：蛋白结构、序列与功能建模            |    198 |           2260 |    8.76 | graph, graphs, gnns, molecular, protein                    |
|      8 | 迁移泛化：域适应、OOD泛化与鲁棒表征（迁移泛化） |     55 |           2260 |    2.43 | domain, adaptation, knowledge, continual, distillation     |
|      9 | 三维视觉：点云、深度估计与相机姿态              |     41 |           2260 |    1.81 | causal, treatment, variables, estimation, confounding      |
|     10 | 时序建模：时间序列预测、动力系统与基础模型      |     41 |           2260 |    1.81 | series, time series, time, forecasting, series forecasting |

### ICLR 2025

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 高效大模型：长上下文、注意力与推理优化     |    565 |           3703 |   15.26 | language, llms, training, llm, fine-tuning             |
|      2 | 优化理论：梯度方法、收敛性与训练动力学     |    457 |           3703 |   12.34 | optimization, training, deep, problem, gradient        |
|      3 | 多模态大模型：视觉语言理解与跨模态推理     |    395 |           3703 |   10.67 | visual, multimodal, video, language, understanding     |
|      4 | 强化学习：离线策略、奖励建模与控制         |    353 |           3703 |    9.53 | reinforcement, policy, offline, policies, environments |
|      5 | 大模型推理：问答、常识与思维链             |    334 |           3703 |    9.02 | llms, language, reasoning, llm, evaluation             |
|      6 | 生成模型：文生图、扩散采样与图像编辑       |    330 |           3703 |    8.91 | diffusion, image, generation, sampling, generative     |
|      7 | 生成模型：视频扩散生成与编辑               |    260 |           3703 |    7.02 | generation, video, motion, gaussian, diffusion         |
|      8 | 图学习：GNN、节点分类与链接预测            |    195 |           3703 |    5.27 | graph, graphs, gnns, node, gnn                         |
|      9 | 时序建模：时间序列预测、动力系统与基础模型 |    177 |           3703 |    4.78 | time, series, time series, brain, dynamics             |
|     10 | 生成模型：扩散模型、采样与内容生成         |    154 |           3703 |    4.16 | molecular, protein, design, prediction, generation     |

### ICLR 2026

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------|
|      1 | 生成模型：视频扩散生成与编辑               |    752 |           5352 |   14.05 | diffusion, generation, video, image, generative       |
|      2 | 高效大模型：推理加速、压缩与资源优化       |    691 |           5352 |   12.91 | language, attention, training, llms, quantization     |
|      3 | 大模型推理：RL驱动推理与奖励学习           |    655 |           5352 |   12.24 | reasoning, language, llms, reinforcement, reward      |
|      4 | 优化理论：随机/非凸优化与收敛率            |    645 |           5352 |   12.05 | optimization, training, framework, dynamics, gradient |
|      5 | 多模态大模型：视觉语言理解与跨模态推理     |    625 |           5352 |   11.68 | reasoning, visual, multimodal, video, language        |
|      6 | 视频理解：动作识别、长视频与时序建模       |    513 |           5352 |    9.59 | policy, reinforcement, action, policies, offline      |
|      7 | 大模型评测：人类偏好、任务指标与领域评估   |    372 |           5352 |    6.95 | agents, evaluation, llms, language, agent             |
|      8 | 可信安全：对抗攻击、后门、水印与隐私       |    358 |           5352 |    6.69 | safety, attacks, adversarial, unlearning, privacy     |
|      9 | 时序建模：时间序列预测、动力系统与基础模型 |    256 |           5352 |    4.78 | time, series, time series, causal, forecasting        |
|     10 | 检索增强大模型：RAG、知识注入与问答        |    194 |           5352 |    3.62 | graph, retrieval, graphs, rag, knowledge              |

### ICML 2020

|   排名 | 细主题名                                           |   篇数 |   当年会议篇数 |   占比% | 关键词                                                     |
|-------:|:---------------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------|
|      1 | 迁移泛化：域适应、OOD泛化与鲁棒表征                |    231 |           1084 |   21.31 | training, deep, loss, labels, generalization               |
|      2 | 优化理论：随机/非凸优化与收敛率                    |    197 |           1084 |   18.17 | optimization, algorithm, algorithms, gradient, convergence |
|      3 | 强化学习：策略优化、奖励学习与控制                 |    181 |           1084 |   16.7  | reinforcement, policy, algorithm, games, reward            |
|      4 | 生成模型：扩散模型、采样与内容生成                 |    153 |           1084 |   14.11 | inference, variational, generative, latent, distribution   |
|      5 | 高效大模型：推理加速、压缩与资源优化               |     77 |           1084 |    7.1  | language, attention, memory, transformer, modeling         |
|      6 | 可信安全：对抗攻击、后门、水印与隐私（安全攻防）   |     73 |           1084 |    6.73 | privacy, causal, fairness, private, differential privacy   |
|      7 | 在线决策：Bandit、后悔界与探索                     |     66 |           1084 |    6.09 | regret, bandits, algorithms, algorithm, online             |
|      8 | 可信安全：对抗攻击、后门、水印与隐私（安全攻防-2） |     55 |           1084 |    5.07 | adversarial, robust, attacks, robustness, perturbations    |
|      9 | AI4Science：分子生成、药物发现与化学建模           |     51 |           1084 |    4.7  | graph, graphs, node, molecular, problem                    |

### ICML 2021

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                                        |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------|
|      1 | 强化学习：策略优化、奖励学习与控制       |    231 |           1183 |   19.53 | policy, reinforcement, agents, algorithms, algorithm          |
|      2 | 优化理论：随机/非凸优化与收敛率          |    173 |           1183 |   14.62 | algorithm, algorithms, problem, clustering, optimization      |
|      3 | 迁移泛化：域适应、OOD泛化与鲁棒表征      |    166 |           1183 |   14.03 | training, deep, gradient, generalization, loss                |
|      4 | 生成模型：扩散模型、采样与内容生成       |    163 |           1183 |   13.78 | inference, variational, latent, distribution, generative      |
|      5 | 高效大模型：推理加速、压缩与资源优化     |    147 |           1183 |   12.43 | attention, training, language, representations, meta-learning |
|      6 | 可信安全：对抗攻击、后门、水印与隐私     |    140 |           1183 |   11.83 | adversarial, training, robustness, label, robust              |
|      7 | 在线决策：Bandit、后悔界与探索           |     99 |           1183 |    8.37 | regret, algorithm, problem, bandit, bandits                   |
|      8 | AI4Science：分子生成、药物发现与化学建模 |     64 |           1183 |    5.41 | graph, gnns, graphs, node, equivariant                        |

### ICML 2022

|   排名 | 细主题名                                         |   篇数 |   当年会议篇数 |   占比% | 关键词                                                    |
|-------:|:-------------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------------|
|      1 | 强化学习：离线策略、奖励建模与控制               |    210 |           1233 |   17.03 | policy, reinforcement, offline, agents, algorithm         |
|      2 | 迁移泛化：域适应、OOD泛化与鲁棒表征              |    171 |           1233 |   13.87 | training, classification, label, domain, generalization   |
|      3 | 高效大模型：推理加速、压缩与资源优化             |    137 |           1233 |   11.11 | contrastive, training, language, attention, speech        |
|      4 | 模型压缩：量化、剪枝与低秩加速                   |    131 |           1233 |   10.62 | training, deep, weights, pruning, memory                  |
|      5 | 优化理论：随机/非凸优化与收敛率                  |    121 |           1233 |    9.81 | optimization, gradient, convergence, problems, algorithms |
|      6 | 生成模型：扩散模型、采样与内容生成               |    114 |           1233 |    9.25 | generative, variational, latent, inference, distribution  |
|      7 | 图学习：图聚类、表示学习与结构匹配               |     95 |           1233 |    7.7  | graph, clustering, graphs, node, algorithms               |
|      8 | 可信安全：对抗攻击、后门、水印与隐私（安全攻防） |     77 |           1233 |    6.24 | federated, privacy, communication, private, local         |
|      9 | 可信安全：对抗攻击、后门、水印与隐私（RAG问答）  |     70 |           1233 |    5.68 | adversarial, robustness, attacks, robust, training        |
|     10 | 在线决策：Bandit、后悔界与探索                   |     68 |           1233 |    5.52 | regret, bandits, algorithm, bandit, problem               |

### ICML 2023

|   排名 | 细主题名                                         |   篇数 |   当年会议篇数 |   占比% | 关键词                                                    |
|-------:|:-------------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------------|
|      1 | 代码智能：程序分析、软件工程与代码检索           |    244 |           1828 |   13.35 | language, training, code, transformers, accuracy          |
|      2 | 鲁棒泛化：OOD检测、校准与噪声标签                |    239 |           1828 |   13.07 | label, distribution, training, problem, labels            |
|      3 | 在线决策：Bandit、后悔界与探索                   |    222 |           1828 |   12.14 | regret, algorithm, algorithms, online, problem            |
|      4 | 生成模型：文生图、扩散采样与图像编辑             |    205 |           1828 |   11.21 | diffusion, generative, image, latent, training            |
|      5 | 强化学习：离线策略、奖励建模与控制               |    196 |           1828 |   10.72 | policy, reinforcement, offline, reward, exploration       |
|      6 | 迁移泛化：域适应、OOD泛化与鲁棒表征              |    175 |           1828 |    9.57 | training, deep, generalization, gradient, features        |
|      7 | 图学习：图聚类、表示学习与结构匹配               |    159 |           1828 |    8.7  | graph, graphs, gnns, node, clustering                     |
|      8 | 优化理论：随机/非凸优化与收敛率                  |    110 |           1828 |    6.02 | optimization, gradient, problems, convergence, stochastic |
|      9 | 可信安全：对抗攻击、后门、水印与隐私（安全攻防） |     95 |           1828 |    5.2  | privacy, federated, private, clients, local               |
|     10 | 因果学习：因果发现、反事实与处理效应             |     74 |           1828 |    4.05 | causal, treatment, effects, effect, counterfactual        |

### ICML 2024

|   排名 | 细主题名                               |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:---------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 语言模型分析：认知、语言结构与可解释性 |    476 |           2610 |   18.24 | language, llms, llm, training, human                   |
|      2 | 视觉感知：目标检测、识别与视觉表征     |    362 |           2610 |   13.87 | training, deep, feature, theoretical, features         |
|      3 | 优化理论：随机/非凸优化与收敛率        |    331 |           2610 |   12.68 | optimization, algorithm, algorithms, problem, problems |
|      4 | 强化学习：离线策略、奖励建模与控制     |    306 |           2610 |   11.72 | policy, reinforcement, reward, algorithm, offline      |
|      5 | 生成模型：文生图、扩散采样与图像编辑   |    276 |           2610 |   10.57 | diffusion, training, image, sampling, generative       |
|      6 | 多模态大模型：视觉语言理解与跨模态推理 |    247 |           2610 |    9.46 | visual, image, vision, multimodal, language            |
|      7 | 可信安全：对抗攻击、后门、水印与隐私   |    199 |           2610 |    7.62 | privacy, adversarial, federated, private, training     |
|      8 | 图学习：图聚类、表示学习与结构匹配     |    178 |           2610 |    6.82 | graph, graphs, gnns, nodes, clustering                 |
|      9 | 因果学习：因果发现、反事实与处理效应   |     93 |           2610 |    3.56 | causal, variables, treatment, effect, causal discovery |
|     10 | AI4Science：蛋白结构、序列与功能建模   |     72 |           2610 |    2.76 | protein, molecular, design, molecules, drug            |

### ICML 2025

|   排名 | 细主题名                               |   篇数 |   当年会议篇数 |   占比% | 关键词                                                       |
|-------:|:---------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------------|
|      1 | 高效大模型：推理加速、压缩与资源优化   |    446 |           3330 |   13.39 | language, training, llms, attention, memory                  |
|      2 | 在线决策：Bandit、后悔界与探索         |    411 |           3330 |   12.34 | policy, reinforcement, algorithm, regret, reward             |
|      3 | 多模态大模型：视觉语言理解与跨模态推理 |    344 |           3330 |   10.33 | visual, image, multimodal, images, video                     |
|      4 | 迁移泛化：域适应、OOD泛化与鲁棒表征    |    330 |           3330 |    9.91 | training, generalization, feature, loss, theoretical         |
|      5 | 代码大模型：代码生成、程序理解与评测   |    308 |           3330 |    9.25 | reasoning, llms, language, llm, code                         |
|      6 | 可信安全：对抗攻击、后门、水印与隐私   |    267 |           3330 |    8.02 | privacy, attacks, adversarial, unlearning, federated         |
|      7 | 生成模型：文生图、扩散采样与图像编辑   |    238 |           3330 |    7.15 | diffusion, image, generation, generative, images             |
|      8 | 优化理论：随机/非凸优化与收敛率        |    220 |           3330 |    6.61 | algorithm, clustering, optimization, algorithms, convergence |
|      9 | 因果学习：因果发现、反事实与处理效应   |    181 |           3330 |    5.44 | causal, prediction, conformal, bayesian, inference           |
|     10 | 图学习：GNN、节点分类与链接预测        |    172 |           3330 |    5.17 | graph, graphs, gnns, node, theoretical                       |

### IJCAI 2020

|   排名 | 细主题名                               |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:---------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 迁移泛化：域适应、OOD泛化与鲁棒表征    |    171 |            778 |   21.98 | label, deep, training, domain, labels                  |
|      2 | 强化学习：策略优化、奖励学习与控制     |    114 |            778 |   14.65 | policy, reinforcement, agent, algorithm, search        |
|      3 | 生成模型：扩散模型、采样与内容生成     |    110 |            778 |   14.14 | knowledge, language, generation, dialogue, information |
|      4 | 高效大模型：推理加速、压缩与资源优化   |    102 |            778 |   13.11 | features, visual, video, image, attention              |
|      5 | 大模型推理：问答、常识与思维链         |    101 |            778 |   12.98 | logic, reasoning, constraint, problem, knowledge       |
|      6 | 大模型社会安全：偏见、虚假信息与检测   |     69 |            778 |    8.87 | agents, social, problem, fairness, study               |
|      7 | 图学习：GNN、节点分类与链接预测        |     57 |            778 |    7.33 | graph, embedding, prediction, information, node        |
|      8 | 多智能体强化学习：博弈、协作与策略学习 |     29 |            778 |    3.73 | financial, portfolio, risk, stock, market              |
|      9 | 推荐系统：排序、召回与点击率预测       |     25 |            778 |    3.21 | recommendation, user, news, users, prediction          |

### IJCAI 2021

|   排名 | 细主题名                                           |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:---------------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 可信安全：对抗攻击、后门、水印与隐私（安全攻防）   |    141 |            722 |   19.53 | image, deep, training, images, adversarial             |
|      2 | 优化理论：梯度方法、收敛性与训练动力学             |    141 |            722 |   19.53 | problem, algorithm, optimization, algorithms, problems |
|      3 | 视频理解：动作识别、长视频与时序建模               |    113 |            722 |   15.65 | reinforcement, planning, agent, policy, action         |
|      4 | 推荐系统：排序、召回与点击率预测                   |     92 |            722 |   12.74 | graph, information, recommendation, node, graphs       |
|      5 | 大模型社会安全：偏见、虚假信息与检测               |     79 |            722 |   10.94 | agents, fairness, games, social, study                 |
|      6 | 可信安全：对抗攻击、后门、水印与隐私（安全攻防-2） |     78 |            722 |   10.8  | label, domain, training, knowledge, privacy            |
|      7 | 开放词汇视觉：开放词汇检测、分割与CLIP语义         |     78 |            722 |   10.8  | language, text, word, knowledge, semantic              |

### IJCAI 2022

|   排名 | 细主题名                             |   篇数 |   当年会议篇数 |   占比% | 关键词                                              |
|-------:|:-------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------|
|      1 | 视觉感知：目标检测、识别与视觉表征   |    194 |            862 |   22.51 | image, images, feature, features, state-of-the-art  |
|      2 | 多智能体：机制设计、拍卖与资源分配   |    156 |            862 |   18.1  | problem, algorithm, agents, study, algorithms       |
|      3 | 图学习：GNN、节点分类与链接预测      |    135 |            862 |   15.66 | graph, graphs, information, existing, node          |
|      4 | 生成模型：扩散模型、采样与内容生成   |    121 |            862 |   14.04 | knowledge, language, information, graph, generation |
|      5 | 强化学习：策略优化、奖励学习与控制   |     99 |            862 |   11.48 | reinforcement, agents, policy, agent, planning      |
|      6 | 异常检测：图异常、欺诈检测与时序异常 |     82 |            862 |    9.51 | adversarial, deep, training, attack, attacks        |
|      7 | 大模型推理：问答、常识与思维链       |     75 |            862 |    8.7  | logic, set, reasoning, problems, answer             |

### IJCAI 2023

|   排名 | 细主题名                               |   篇数 |   当年会议篇数 |   占比% | 关键词                                               |
|-------:|:---------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------|
|      1 | 联邦学习：异构客户端、隐私与分布式优化 |    165 |            851 |   19.39 | framework, label, training, time, existing           |
|      2 | 视觉感知：目标检测、识别与视觉表征     |    156 |            851 |   18.33 | image, images, features, information, visual         |
|      3 | 大模型推理：问答、常识与思维链         |    155 |            851 |   18.21 | problem, reasoning, algorithm, algorithms, problems  |
|      4 | 强化学习：策略优化、奖励学习与控制     |    138 |            851 |   16.22 | agents, reinforcement, planning, policy, multi-agent |
|      5 | 图学习：图聚类、表示学习与结构匹配     |     84 |            851 |    9.87 | graph, graphs, nodes, information, node              |
|      6 | 多智能体：机制设计、拍卖与资源分配     |     78 |            851 |    9.17 | agents, fairness, problem, study, agent              |
|      7 | 多模态音视频生成：语音、音乐与情感生成 |     75 |            851 |    8.81 | language, text, audio, music, generation             |

### IJCAI 2024

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                               |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------|
|      1 | 视觉感知：目标检测、识别与视觉表征         |    222 |           1048 |   21.18 | image, images, features, visual, information         |
|      2 | 强化学习：策略优化、奖励学习与控制         |    126 |           1048 |   12.02 | policy, agents, reinforcement, agent, multi-agent    |
|      3 | 优化理论：梯度方法、收敛性与训练动力学     |    121 |           1048 |   11.55 | problem, algorithms, algorithm, search, optimization |
|      4 | 大模型推理：问答、常识与思维链             |    120 |           1048 |   11.45 | language, llms, knowledge, text, reasoning           |
|      5 | 图学习：图聚类、表示学习与结构匹配         |    112 |           1048 |   10.69 | graph, clustering, information, graphs, multi-view   |
|      6 | 迁移泛化：域适应、OOD泛化与鲁棒表征        |    103 |           1048 |    9.83 | label, domain, training, samples, existing           |
|      7 | 时序建模：时间序列预测、动力系统与基础模型 |     97 |           1048 |    9.26 | time, series, causal, time series, forecasting       |
|      8 | 推荐系统：排序、召回与点击率预测           |     78 |           1048 |    7.44 | agents, games, problem, fairness, study              |
|      9 | 多媒体安全：Deepfake检测、伪造识别与攻防   |     69 |           1048 |    6.58 | federated, attacks, attack, adversarial, privacy     |

### IJCAI 2025

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                               |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------|
|      1 | 多模态理解：视觉语言表征与跨模态对齐       |    277 |           1280 |   21.64 | image, visual, features, multimodal, video           |
|      2 | 强化学习：策略优化、奖励学习与控制         |    235 |           1280 |   18.36 | agents, reinforcement, planning, framework, policy   |
|      3 | 图学习：图聚类、表示学习与结构匹配         |    164 |           1280 |   12.81 | graph, clustering, multi-view, information, node     |
|      4 | 大模型推理：问答、常识与思维链             |    159 |           1280 |   12.42 | llms, knowledge, language, reasoning, generation     |
|      5 | 时序建模：时间序列预测、动力系统与基础模型 |    139 |           1280 |   10.86 | time, series, time series, forecasting, temporal     |
|      6 | 优化理论：梯度方法、收敛性与训练动力学     |    119 |           1280 |    9.3  | problem, algorithm, optimization, algorithms, agents |
|      7 | 可信安全：对抗攻击、后门、水印与隐私       |     86 |           1280 |    6.72 | federated, attacks, training, adversarial, attack    |
|      8 | 推荐系统：排序、召回与点击率预测           |     65 |           1280 |    5.08 | causal, recommendation, molecular, user, prediction  |
|      9 | 医疗视觉：医学影像分割、病理与临床影像     |     36 |           1280 |    2.81 | medical, diagnosis, clinical, images, cancer         |

### KDD 2020

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                              |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------|
|      1 | 图基础模型：LLM增强图学习与节点表示        |     64 |            217 |   29.49 | graph, node, information, different, representation |
|      2 | 可信安全：对抗攻击、后门、水印与隐私       |     52 |            217 |   23.96 | deep, attack, adversarial, algorithms, information  |
|      3 | 图学习：图聚类、表示学习与结构匹配         |     39 |            217 |   17.97 | graph, clustering, mining, graphs, algorithms       |
|      4 | 推荐系统：排序、召回与点击率预测           |     34 |            217 |   15.67 | recommendation, user, users, items, systems         |
|      5 | 时序建模：时间序列预测、动力系统与基础模型 |     28 |            217 |   12.9  | time, prediction, series, temporal, time series     |

### KDD 2021

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------|
|      1 | 推荐系统：排序、召回与点击率预测           |     86 |            239 |   35.98 | recommendation, user, adversarial, framework, problem |
|      2 | 图学习：GNN、节点分类与链接预测            |     52 |            239 |   21.76 | graph, graphs, node, nodes, information               |
|      3 | 图学习：图聚类、表示学习与结构匹配         |     38 |            239 |   15.9  | algorithm, clustering, time, algorithms, tensor       |
|      4 | 鲁棒泛化：OOD检测、校准与噪声标签          |     38 |            239 |   15.9  | labels, label, knowledge, language, framework         |
|      5 | 时序建模：时间序列预测、动力系统与基础模型 |     25 |            239 |   10.46 | temporal, forecasting, traffic, graph, spatial        |

### KDD 2022

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                           |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------|
|      1 | 图学习：GNN、节点分类与链接预测            |     62 |            253 |   24.51 | graph, node, gnns, graphs, nodes                 |
|      2 | 时序建模：时间序列预测、动力系统与基础模型 |     62 |            253 |   24.51 | time, graph, prediction, series, causal          |
|      3 | 联邦大模型微调：LoRA、客户端异构与模型合并 |     43 |            253 |   17    | label, training, labels, federated, clients      |
|      4 | 推荐系统：偏好建模、反馈学习与个性化排序   |     30 |            253 |   11.86 | recommendation, user, systems, item, recommender |
|      5 | 知识图谱：实体关系、推理与补全             |     30 |            253 |   11.86 | knowledge, graph, entity, kgs, entities          |
|      6 | 优化理论：梯度方法、收敛性与训练动力学     |     26 |            253 |   10.28 | clustering, problem, algorithm, cluster, space   |

### KDD 2023

|   排名 | 细主题名                                    |   篇数 |   当年会议篇数 |   占比% | 关键词                                         |
|-------:|:--------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------|
|      1 | 联邦大模型微调：LoRA、客户端异构与模型合并  |     81 |            313 |   25.88 | training, label, federated, framework, labels  |
|      2 | 推荐系统：排序、召回与点击率预测            |     73 |            313 |   23.32 | knowledge, recommendation, user, graph, search |
|      3 | 图学习：GNN、节点分类与链接预测（生成编辑） |     65 |            313 |   20.77 | graph, gnns, node, graphs, information         |
|      4 | 时序建模：时间序列预测、动力系统与基础模型  |     56 |            313 |   17.89 | time, series, time series, traffic, temporal   |
|      5 | 图学习：GNN、节点分类与链接预测（RAG问答）  |     38 |            313 |   12.14 | algorithm, algorithms, query, problem, nodes   |

### KDD 2024

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                          |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------------------|
|      1 | 图学习：GNN、节点分类与链接预测            |    117 |            411 |   28.47 | graph, graphs, node, nodes, gnns                                |
|      2 | 联邦大模型微调：LoRA、客户端异构与模型合并 |     90 |            411 |   21.9  | training, federated, existing, experiments, fairness            |
|      3 | 推荐系统：排序、召回与点击率预测           |     62 |            411 |   15.09 | recommendation, user, recommender, items, users                 |
|      4 | 具身智能：机器人操作、导航与视觉语言动作   |     52 |            411 |   12.65 | traffic, prediction, trajectory, online, trajectories           |
|      5 | 大模型训练：微调、数据配方与任务适配       |     48 |            411 |   11.68 | language, knowledge, graph, llms, information                   |
|      6 | 时序建模：时间序列预测、动力系统与基础模型 |     42 |            411 |   10.22 | time, series, time series, forecasting, time series forecasting |

### KDD 2025

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 图基础模型：LLM增强图学习与节点表示        |    122 |            552 |   22.1  | graph, graphs, node, nodes, gnns                       |
|      2 | 大模型推理：问答、常识与思维链             |     87 |            552 |   15.76 | knowledge, llms, language, graph, reasoning            |
|      3 | 时序建模：时间序列预测、动力系统与基础模型 |     69 |            552 |   12.5  | time, series, forecasting, time series, temporal       |
|      4 | AI4Science：分子生成、药物发现与化学建模   |     65 |            552 |   11.78 | causal, molecular, treatment, prediction, framework    |
|      5 | 推荐系统：偏好建模、反馈学习与个性化排序   |     65 |            552 |   11.78 | recommendation, user, systems, information, multimodal |
|      6 | 图学习：图异常检测、聚类与结构表示         |     49 |            552 |    8.88 | detection, anomaly, graph, anomaly detection, attack   |
|      7 | 在线决策：Bandit、后悔界与探索             |     35 |            552 |    6.34 | online, optimization, problem, algorithm, regret       |
|      8 | 具身智能：机器人操作、导航与视觉语言动作   |     33 |            552 |    5.98 | trajectory, urban, human, mobility, prediction         |
|      9 | 联邦大模型微调：LoRA、客户端异构与模型合并 |     27 |            552 |    4.89 | federated, training, client, convergence, clients      |

### NAACL 2021

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                               |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------------------------|
|      1 | 开放词汇视觉：开放词汇检测、分割与CLIP语义 |     85 |            477 |   17.82 | language, parsing, semantic, attention, natural                      |
|      2 | 大模型社会安全：偏见、虚假信息与检测       |     79 |            477 |   16.56 | language, nlp, social, detection, research                           |
|      3 | 可信安全：对抗攻击、后门、水印与隐私       |     63 |            477 |   13.21 | text, classification, text classification, training, representations |
|      4 | 多语言NLP：机器翻译、跨语言与低资源        |     59 |            477 |   12.37 | translation, machine, machine translation, languages, multilingual   |
|      5 | 知识图谱：实体关系、推理与补全             |     54 |            477 |   11.32 | knowledge, graph, entity, relation, coreference                      |
|      6 | 大模型推理：问答、常识与思维链             |     49 |            477 |   10.27 | question, answering, question answering, reasoning, questions        |
|      7 | 生成模型：扩散模型、采样与内容生成         |     46 |            477 |    9.64 | summarization, generation, text, summaries, abstractive              |
|      8 | 对话系统：响应生成、情感支持与任务型对话   |     42 |            477 |    8.81 | dialogue, responses, language, dialog, slot                          |

### NAACL 2022

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                   |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------------|
|      1 | 大模型训练：微调、数据配方与任务适配       |     85 |            442 |   19.23 | language, training, pre-trained, text, plms              |
|      2 | 大模型社会安全：偏见、虚假信息与检测       |     81 |            442 |   18.33 | language, text, bias, detection, work                    |
|      3 | 大模型推理：问答、常识与思维链             |     65 |            442 |   14.71 | summarization, question, evaluation, reasoning, answer   |
|      4 | 多语言NLP：机器翻译、跨语言与低资源        |     58 |            442 |   13.12 | translation, languages, language, machine, cross-lingual |
|      5 | 开放词汇视觉：开放词汇检测、分割与CLIP语义 |     58 |            442 |   13.12 | language, semantic, generation, knowledge, text          |
|      6 | 事件视觉：事件相机、运动估计与时序感知     |     56 |            442 |   12.67 | extraction, entity, event, information, relation         |
|      7 | 生成模型：扩散模型、采样与内容生成         |     39 |            442 |    8.82 | dialogue, generation, knowledge, responses, utterances   |

### NAACL 2024

|   排名 | 细主题名                                         |   篇数 |   当年会议篇数 |   占比% | 关键词                                                        |
|-------:|:-------------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------|
|      1 | 大模型社会安全：偏见、虚假信息与检测             |    112 |            562 |   19.93 | summarization, language, event, llms, bias                    |
|      2 | 大模型训练：微调、数据配方与任务适配（搜索排序） |    102 |            562 |   18.15 | language, llms, text, in-context, different                   |
|      3 | 检索增强大模型：RAG、知识注入与问答              |     77 |            562 |   13.7  | entity, knowledge, retrieval, language, information           |
|      4 | 大模型训练：微调、数据配方与任务适配（社会安全） |     73 |            562 |   12.99 | languages, language, translation, multilingual, cross-lingual |
|      5 | 可信安全：对抗攻击、后门、水印与隐私             |     57 |            562 |   10.14 | llms, language, adversarial, attacks, llm                     |
|      6 | 大模型推理：问答、常识与思维链                   |     55 |            562 |    9.79 | reasoning, llms, language, knowledge, prompting               |
|      7 | 大模型评测：人类偏好、任务指标与领域评估         |     51 |            562 |    9.07 | evaluation, dialogue, language, llms, generation              |
|      8 | 多模态大模型：视觉语言理解与跨模态推理           |     35 |            562 |    6.23 | multimodal, image, visual, multi-modal, vision-language       |

### NAACL 2025

|   排名 | 细主题名                               |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:---------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 大模型社会安全：偏见、虚假信息与检测   |    126 |            718 |   17.55 | language, llms, human, bias, social                    |
|      2 | 高效大模型：长上下文、注意力与推理优化 |    116 |            718 |   16.16 | language, llms, training, llm, inference               |
|      3 | 语音音频：ASR、说话人与音频理解        |     97 |            718 |   13.51 | languages, language, multilingual, translation, speech |
|      4 | 代码大模型：代码生成、程序理解与评测   |     94 |            718 |   13.09 | llms, language, evaluation, code, llm                  |
|      5 | 大模型推理：问答、常识与思维链         |     92 |            718 |   12.81 | reasoning, llms, language, agents, framework           |
|      6 | 检索增强大模型：RAG、知识注入与问答    |     83 |            718 |   11.56 | retrieval, rag, llms, question, generation             |
|      7 | 多模态大模型：视觉语言理解与跨模态推理 |     63 |            718 |    8.77 | multimodal, visual, image, images, vlms                |
|      8 | 可信安全：对抗攻击、后门、水印与隐私   |     47 |            718 |    6.55 | safety, attacks, attack, llms, llm                     |

### NeurIPS 2020

|   排名 | 细主题名                                           |   篇数 |   当年会议篇数 |   占比% | 关键词                                                     |
|-------:|:---------------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------|
|      1 | 视觉感知：目标检测、识别与视觉表征                 |    298 |           1898 |   15.7  | image, object, training, images, representations           |
|      2 | 强化学习：策略优化、奖励学习与控制                 |    277 |           1898 |   14.59 | reinforcement, policy, agents, agent, algorithm            |
|      3 | 统计学习理论：泛化界、风险分析与样本复杂度         |    229 |           1898 |   12.07 | algorithm, problem, regression, fairness, bounds           |
|      4 | 优化理论：梯度方法、收敛性与训练动力学（搜索排序） |    225 |           1898 |   11.85 | deep, training, architecture, pruning, search              |
|      5 | 高效大模型：推理加速、压缩与资源优化               |    224 |           1898 |   11.8  | training, language, meta-learning, attention, deep         |
|      6 | 优化理论：梯度方法、收敛性与训练动力学（推理效率） |    125 |           1898 |    6.59 | bayesian, variational, inference, optimization, flows      |
|      7 | 在线决策：Bandit、后悔界与探索                     |    115 |           1898 |    6.06 | regret, online, algorithm, algorithms, problem             |
|      8 | AI4Science：分子生成、药物发现与化学建模           |    115 |           1898 |    6.06 | graph, graphs, gnns, node, nodes                           |
|      9 | 优化理论：随机/非凸优化与收敛率                    |    114 |           1898 |    6.01 | gradient, optimization, algorithm, stochastic, convergence |
|     10 | 可信安全：对抗攻击、后门、水印与隐私               |    108 |           1898 |    5.69 | adversarial, robustness, attacks, training, robust         |

### NeurIPS 2021

|   排名 | 细主题名                                         |   篇数 |   当年会议篇数 |   占比% | 关键词                                                   |
|-------:|:-------------------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------------|
|      1 | 优化理论：随机/非凸优化与收敛率                  |    379 |           2334 |   16.24 | optimization, algorithm, gradient, problem, algorithms   |
|      2 | 视觉感知：目标检测、识别与视觉表征               |    373 |           2334 |   15.98 | image, transformer, object, images, training             |
|      3 | 强化学习：离线策略、奖励建模与控制               |    348 |           2334 |   14.91 | reinforcement, policy, agents, agent, offline            |
|      4 | 迁移泛化：域适应、OOD泛化与鲁棒表征（三维几何）  |    220 |           2334 |    9.43 | training, deep, gradient, pruning, generalization        |
|      5 | 迁移泛化：域适应、OOD泛化与鲁棒表征（迁移泛化）  |    188 |           2334 |    8.05 | domain, adaptation, label, target, training              |
|      6 | 多模态大模型：视觉语言理解与跨模态推理           |    177 |           2334 |    7.58 | language, speech, training, visual, brain                |
|      7 | AI4Science：分子生成、药物发现与化学建模         |    153 |           2334 |    6.56 | graph, graphs, gnns, node, gnn                           |
|      8 | 三维视觉：Gaussian Splatting、新视角合成与重建   |    139 |           2334 |    5.96 | variational, inference, bayesian, distribution, gaussian |
|      9 | 在线决策：Bandit、后悔界与探索                   |    117 |           2334 |    5.01 | bandits, regret, bandit, algorithms, problem             |
|     10 | 可信安全：对抗攻击、后门、水印与隐私（安全攻防） |     96 |           2334 |    4.11 | adversarial, robustness, attacks, training, robust       |

### NeurIPS 2022

|   排名 | 细主题名                                          |   篇数 |   当年会议篇数 |   占比% | 关键词                                                     |
|-------:|:--------------------------------------------------|-------:|---------------:|--------:|:-----------------------------------------------------------|
|      1 | 视觉感知：语义/实例/全景分割                      |    382 |           2671 |   14.3  | image, object, segmentation, video, visual                 |
|      2 | 强化学习：离线策略、奖励建模与控制                |    377 |           2671 |   14.11 | reinforcement, policy, agents, reward, agent               |
|      3 | 迁移泛化：域适应、OOD泛化与鲁棒表征（迁移泛化）   |    374 |           2671 |   14    | training, deep, loss, generalization, work                 |
|      4 | 优化理论：随机/非凸优化与收敛率                   |    314 |           2671 |   11.76 | optimization, algorithm, stochastic, convergence, gradient |
|      5 | 大模型推理：问答、常识与思维链                    |    251 |           2671 |    9.4  | language, training, knowledge, distillation, reasoning     |
|      6 | 在线决策：Bandit、后悔界与探索                    |    189 |           2671 |    7.08 | regret, fairness, algorithm, problem, online               |
|      7 | 可信安全：对抗攻击、后门、水印与隐私（Agent规划） |    155 |           2671 |    5.8  | adversarial, attacks, training, robustness, attack         |
|      8 | 图学习：GNN、节点分类与链接预测                   |    151 |           2671 |    5.65 | graph, graphs, node, gnns, nodes                           |
|      9 | 优化理论：梯度方法、收敛性与训练动力学            |    148 |           2671 |    5.54 | bayesian, optimization, posterior, uncertainty, inference  |
|     10 | 时序建模：时间序列预测、动力系统与基础模型        |    122 |           2671 |    4.57 | time, series, brain, time series, spiking                  |

### NeurIPS 2023

|   排名 | 细主题名                                         |   篇数 |   当年会议篇数 |   占比% | 关键词                                                  |
|-------:|:-------------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------|
|      1 | 优化理论：随机/非凸优化与收敛率                  |    480 |           3218 |   14.92 | optimization, algorithm, gradient, problems, algorithms |
|      2 | 视觉感知：语义/实例/全景分割                     |    425 |           3218 |   13.21 | segmentation, image, training, visual, features         |
|      3 | 强化学习：离线策略、奖励建模与控制               |    365 |           3218 |   11.34 | policy, reinforcement, offline, reward, policies        |
|      4 | 大模型推理：问答、常识与思维链                   |    291 |           3218 |    9.04 | language, llms, reasoning, training, transformers       |
|      5 | 生成模型：文生图、扩散采样与图像编辑             |    274 |           3218 |    8.51 | diffusion, image, generation, generative, images        |
|      6 | 迁移泛化：域适应、OOD泛化与鲁棒表征              |    256 |           3218 |    7.96 | training, deep, gradient, generalization, functions     |
|      7 | 可信安全：对抗攻击、后门、水印与隐私（安全攻防） |    171 |           3218 |    5.31 | adversarial, robustness, attacks, robust, training      |
|      8 | 在线决策：Bandit、后悔界与探索                   |    169 |           3218 |    5.25 | regret, games, algorithm, bandits, bandit               |
|      9 | 时序建模：时间序列预测、动力系统与基础模型       |    160 |           3218 |    4.97 | brain, time, series, time series, forecasting           |
|     10 | 图学习：GNN、节点分类与链接预测                  |    157 |           3218 |    4.88 | graph, graphs, gnns, node, nodes                        |

### NeurIPS 2024

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                 |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------|
|      1 | 优化理论：梯度方法、收敛性与训练动力学     |    662 |           4034 |   16.41 | optimization, algorithms, algorithm, problem, training |
|      2 | 多模态大模型：视觉语言理解与跨模态推理     |    469 |           4034 |   11.63 | visual, vision, image, training, multimodal            |
|      3 | 生成模型：视频扩散生成与编辑               |    464 |           4034 |   11.5  | diffusion, image, generation, images, video            |
|      4 | 高效大模型：推理加速、压缩与资源优化       |    393 |           4034 |    9.74 | language, llms, training, transformers, attention      |
|      5 | 强化学习：离线策略、奖励建模与控制         |    349 |           4034 |    8.65 | policy, reinforcement, offline, policies, agents       |
|      6 | 三维视觉：点云、深度估计与相机姿态         |    287 |           4034 |    7.11 | gaussian, scene, rendering, point, object              |
|      7 | 大模型推理：RL驱动推理与奖励学习           |    284 |           4034 |    7.04 | llms, language, reasoning, human, preference           |
|      8 | 时序建模：时间序列预测、动力系统与基础模型 |    235 |           4034 |    5.83 | time, series, time series, dynamics, forecasting       |
|      9 | 图学习：图异常检测、聚类与结构表示         |    216 |           4034 |    5.35 | graph, graphs, ood, node, gnns                         |
|     10 | 可信安全：对抗攻击、后门、水印与隐私       |    207 |           4034 |    5.13 | adversarial, attacks, attack, privacy, private         |

### NeurIPS 2025

|   排名 | 细主题名                               |   篇数 |   当年会议篇数 |   占比% | 关键词                                                   |
|-------:|:---------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------------|
|      1 | 大模型推理：RL驱动推理与奖励学习       |    781 |           5286 |   14.77 | reasoning, llms, language, llm, training                 |
|      2 | 生成模型：视频扩散生成与编辑           |    583 |           5286 |   11.03 | video, motion, generation, scene, reconstruction         |
|      3 | 多模态大模型：视觉语言理解与跨模态推理 |    575 |           5286 |   10.88 | visual, reasoning, multimodal, video, understanding      |
|      4 | 在线决策：Bandit、后悔界与探索         |    570 |           5286 |   10.78 | policy, reinforcement, regret, algorithm, optimization   |
|      5 | 优化理论：随机/非凸优化与收敛率        |    548 |           5286 |   10.37 | algorithm, gradient, algorithms, optimization, training  |
|      6 | 生成模型：文生图、扩散采样与图像编辑   |    538 |           5286 |   10.18 | diffusion, image, generation, sampling, generative       |
|      7 | 高效大模型：推理加速、压缩与资源优化   |    414 |           5286 |    7.83 | attention, training, language, transformers, transformer |
|      8 | 图基础模型：LLM增强图学习与节点表示    |    235 |           5286 |    4.45 | graph, graphs, node, gnns, framework                     |
|      9 | 可信安全：对抗攻击、后门、水印与隐私   |    226 |           5286 |    4.28 | attacks, adversarial, attack, unlearning, safety         |
|     10 | 神经科学AI：脑活动建模、EEG与脉冲网络  |    193 |           5286 |    3.65 | brain, dynamics, activity, spiking, neurons              |

### SIGIR 2020

|   排名 | 细主题名                                     |   篇数 |   当年会议篇数 |   占比% | 关键词                                           |
|-------:|:---------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------|
|      1 | 推荐系统：排序、召回与点击率预测（推荐排序） |     70 |            147 |   47.62 | recommendation, user, graph, users, items        |
|      2 | 信息检索：搜索排序、文档检索与重排           |     46 |            147 |   31.29 | retrieval, information, ranking, query, training |
|      3 | 推荐系统：排序、召回与点击率预测（搜索排序） |     31 |            147 |   21.09 | search, user, evaluation, metrics, relevance     |

### SIGIR 2021

|   排名 | 细主题名                                     |   篇数 |   当年会议篇数 |   占比% | 关键词                                              |
|-------:|:---------------------------------------------|-------:|---------------:|--------:|:----------------------------------------------------|
|      1 | 多媒体检索：跨模态检索、语义匹配与内容理解   |     41 |            151 |   27.15 | retrieval, image, query, training, video            |
|      2 | 推荐系统：偏好建模、反馈学习与个性化排序     |     40 |            151 |   26.49 | recommendation, users, user, recommender, systems   |
|      3 | 推荐系统：排序、召回与点击率预测（推荐排序） |     30 |            151 |   19.87 | graph, user, recommendation, information, users     |
|      4 | 知识图谱：实体关系、推理与补全               |     24 |            151 |   15.89 | news, knowledge, graph, relation, information       |
|      5 | 推荐系统：排序、召回与点击率预测（搜索排序） |     16 |            151 |   10.6  | knowledge, user, conversational, dialogue, language |

### SIGIR 2022

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                            |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------|
|      1 | 多媒体检索：跨模态检索、语义匹配与内容理解 |     68 |            161 |   42.24 | retrieval, knowledge, text, information, query    |
|      2 | 推荐系统：偏好建模、反馈学习与个性化排序   |     50 |            161 |   31.06 | user, recommendation, users, systems, recommender |
|      3 | 推荐系统：排序、召回与点击率预测           |     43 |            161 |   26.71 | recommendation, graph, user, information, users   |

### SIGIR 2023

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                                |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------|
|      1 | 推荐系统：排序、召回与点击率预测（推荐排序）   |     59 |            165 |   35.76 | recommendation, user, users, items, recommender       |
|      2 | 信息检索：搜索排序、文档检索与重排             |     47 |            165 |   28.48 | retrieval, information, legal, language, framework    |
|      3 | 推荐系统：排序、召回与点击率预测（推荐排序-2） |     31 |            165 |   18.79 | graph, recommendation, contrastive, user, information |
|      4 | 视觉感知：语义/实例/全景分割                   |     28 |            165 |   16.97 | graph, knowledge, entities, information, semantic     |

### SIGIR 2024

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                                       |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------------------|
|      1 | 多媒体检索：跨模态检索、语义匹配与内容理解     |     93 |            214 |   43.46 | retrieval, information, search, relevance, language          |
|      2 | 推荐系统：排序、召回与点击率预测（推荐排序）   |     59 |            214 |   27.57 | recommendation, user, users, recommendations, fairness       |
|      3 | 推荐系统：排序、召回与点击率预测（推荐排序-2） |     37 |            214 |   17.29 | graph, knowledge, collaborative, information, recommendation |
|      4 | 推荐系统：排序、召回与点击率预测（推荐排序-3） |     25 |            214 |   11.68 | recommendation, llms, language, sequential, recommender      |

### SIGIR 2025

|   排名 | 细主题名                                   |   篇数 |   当年会议篇数 |   占比% | 关键词                                                |
|-------:|:-------------------------------------------|-------:|---------------:|--------:|:------------------------------------------------------|
|      1 | 推荐系统：偏好建模、反馈学习与个性化排序   |     85 |            239 |   35.56 | recommendation, user, item, users, preferences        |
|      2 | 多媒体检索：跨模态检索、语义匹配与内容理解 |     77 |            239 |   32.22 | retrieval, search, document, ranking, text            |
|      3 | 检索增强大模型：RAG、知识注入与问答        |     51 |            239 |   21.34 | rag, knowledge, llms, search, information             |
|      4 | 推荐系统：排序、召回与点击率预测           |     26 |            239 |   10.88 | graph, knowledge, graphs, recommendation, information |

### WWW 2020

|   排名 | 细主题名                                       |   篇数 |   当年会议篇数 |   占比% | 关键词                                             |
|-------:|:-----------------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------|
|      1 | 大模型推理：问答、常识与思维链                 |     93 |            317 |   29.34 | knowledge, information, search, question, entities |
|      2 | 图学习：图聚类、表示学习与结构匹配             |     63 |            317 |   19.87 | graph, node, graphs, embedding, information        |
|      3 | 推荐系统：排序、召回与点击率预测（推荐排序）   |     62 |            317 |   19.56 | social, online, users, content, user               |
|      4 | 推荐系统：排序、召回与点击率预测（搜索排序）   |     54 |            317 |   17.03 | recommendation, user, users, item, items           |
|      5 | 推荐系统：排序、召回与点击率预测（推荐排序-2） |     45 |            317 |   14.2  | web, users, applications, privacy, security        |

### WWW 2021

|   排名 | 细主题名                                 |   篇数 |   当年会议篇数 |   占比% | 关键词                                          |
|-------:|:-----------------------------------------|-------:|---------------:|--------:|:------------------------------------------------|
|      1 | 知识图谱：实体关系、推理与补全           |     90 |            355 |   25.35 | knowledge, graph, information, entity, entities |
|      2 | 推荐系统：偏好建模、反馈学习与个性化排序 |     90 |            355 |   25.35 | user, recommendation, users, items, search      |
|      3 | 大模型社会安全：偏见、虚假信息与检测     |     83 |            355 |   23.38 | graph, node, graphs, information, social        |
|      4 | 推荐系统：排序、召回与点击率预测         |     66 |            355 |   18.59 | privacy, user, web, mobile, apps                |
|      5 | 语音音频：ASR、说话人与音频理解          |     26 |            355 |    7.32 | online, emotion, conversations, news, social    |

### WWW 2022

|   排名 | 细主题名                                     |   篇数 |   当年会议篇数 |   占比% | 关键词                                             |
|-------:|:---------------------------------------------|-------:|---------------:|--------:|:---------------------------------------------------|
|      1 | 图学习：GNN、节点分类与链接预测              |     99 |            364 |   27.2  | graph, graphs, node, gnns, temporal                |
|      2 | 推荐系统：排序、召回与点击率预测（搜索排序） |     96 |            364 |   26.37 | recommendation, user, users, items, framework      |
|      3 | 推荐系统：排序、召回与点击率预测（推荐排序） |     83 |            364 |   22.8  | social, news, users, media, web                    |
|      4 | 知识抽取：实体识别、关系抽取与事件理解       |     57 |            364 |   15.66 | knowledge, information, language, entity, document |
|      5 | 多智能体：机制设计、拍卖与资源分配           |     29 |            364 |    7.97 | auction, auctions, optimal, online, objectives     |

### WWW 2023

|   排名 | 细主题名                             |   篇数 |   当年会议篇数 |   占比% | 关键词                                                        |
|-------:|:-------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------------------|
|      1 | 生成模型：扩散模型、采样与内容生成   |     96 |            371 |   25.88 | information, text, web, search, generation                    |
|      2 | 推荐系统：排序、召回与点击率预测     |     74 |            371 |   19.95 | recommendation, item, user, users, items                      |
|      3 | 可信安全：对抗攻击、后门、水印与隐私 |     70 |            371 |   18.87 | web, attacks, training, privacy, edge                         |
|      4 | 图学习：GNN、节点分类与链接预测      |     66 |            371 |   17.79 | graph, node, nodes, graphs, gnns                              |
|      5 | 多智能体：机制设计、拍卖与资源分配   |     37 |            371 |    9.97 | fairness, problem, online, auction, algorithm                 |
|      6 | 知识图谱：实体关系、推理与补全       |     28 |            371 |    7.55 | knowledge, knowledge graph, graph, entities, knowledge graphs |

### WWW 2024

|   排名 | 细主题名                                     |   篇数 |   当年会议篇数 |   占比% | 关键词                                            |
|-------:|:---------------------------------------------|-------:|---------------:|--------:|:--------------------------------------------------|
|      1 | 图学习：GNN、节点分类与链接预测              |     84 |            404 |   20.79 | graph, node, graphs, nodes, gnns                  |
|      2 | 视觉感知：目标检测、识别与视觉表征           |     78 |            404 |   19.31 | web, applications, security, detection, websites  |
|      3 | 推荐系统：排序、召回与点击率预测（搜索排序） |     69 |            404 |   17.08 | language, llms, user, search, recommendation      |
|      4 | 推荐系统：排序、召回与点击率预测（推荐排序） |     63 |            404 |   15.59 | recommendation, user, users, federated, systems   |
|      5 | 知识图谱：实体关系、推理与补全               |     43 |            404 |   10.64 | knowledge, graph, knowledge graph, embedding, kgs |
|      6 | 大模型社会安全：偏见、虚假信息与检测         |     35 |            404 |    8.66 | news, social, media, content, social media        |
|      7 | 大模型应用：训练、评测与任务适配             |     32 |            404 |    7.92 | auction, auctions, mechanisms, welfare, optimal   |

### WWW 2025

|   排名 | 细主题名                             |   篇数 |   当年会议篇数 |   占比% | 关键词                                           |
|-------:|:-------------------------------------|-------:|---------------:|--------:|:-------------------------------------------------|
|      1 | 推荐系统：检索增强推荐、排序与个性化 |     40 |            154 |   25.97 | recommendation, user, social, systems, retrieval |
|      2 | 图学习：图异常检测、聚类与结构表示   |     36 |            154 |   23.38 | graph, graphs, nodes, node, knowledge            |
|      3 | 视觉感知：目标检测、识别与视觉表征   |     32 |            154 |   20.78 | websites, web, phishing, security, detection     |
|      4 | 大模型推理：问答、常识与思维链       |     29 |            154 |   18.83 | llms, language, web, reasoning, knowledge        |
|      5 | 可信安全：对抗攻击、后门、水印与隐私 |     17 |            154 |   11.04 | privacy, federated, global, clients, local       |
