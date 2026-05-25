#!/usr/bin/env python3
"""Build a static topic atlas from fine-grained topic results."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pandas as pd


DEFAULT_TOPIC_ROOT = Path("results/fine_grained_venue_year_topics_2020_2026_mcs_fine")
DEFAULT_OUTPUT_ROOT = Path("docs/topic-atlas")

ROBOTICS_VENUES = {"ICRA", "IROS", "RSS"}


CN_TOPIC_RULES: list[tuple[tuple[str, ...], str]] = [
    (("rlvr", "verifiable"), "可验证奖励驱动的大模型推理"),
    (("diffusion language",), "扩散语言模型与并行解码"),
    (("llm-as-a-judge",), "LLM-as-Judge 与自动评测"),
    (("vision-language", "robotic"), "视觉语言驱动的机器人操作"),
    (("vision-language", "manipulation"), "视觉语言驱动的机器人操作"),
    (("vision-and-language", "robot"), "视觉语言导航与具身指令跟随"),
    (("vision-and-language", "navigation"), "视觉语言导航与具身指令跟随"),
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
    (("language", "planning", "robot"), "语言模型驱动的机器人任务规划"),
    (("llm", "planning", "robot"), "语言模型驱动的机器人任务规划"),
    (("large language", "planning", "robot"), "语言模型驱动的机器人任务规划"),
    (("behavior tree", "planning"), "机器人任务规划与行为树生成"),
    (("temporal logic", "planning"), "时序逻辑约束下的机器人任务规划"),
    (("logic", "temporal", "planning"), "时序逻辑约束下的机器人任务规划"),
    (("multi-robot", "path"), "多机器人路径规划与协同导航"),
    (("multi-agent", "multi-robot"), "多机器人路径规划与协同导航"),
    (("vision-and-language navigation",), "视觉语言导航与具身指令跟随"),
    (("language", "robot", "navigation"), "语言/视觉语言引导的机器人导航"),
    (("language", "robot", "grounding"), "语言/视觉语言引导的机器人语义接地"),
    (("human-robot", "collaboration"), "人机交互与协作机器人"),
    (("human-robot", "trust"), "人机交互与协作机器人"),
    (("human-robot", "social"), "社交导航与人机交互"),
    (("agricultural", "robotic"), "农业机器人与自主采摘"),
    (("harvesting", "robotic"), "农业机器人与自主采摘"),
    (("continuum", "manipulator"), "连续体机械臂与柔顺控制"),
    (("cable-driven", "control"), "绳驱连续体机器人控制"),
    (("magnetic", "microrobots"), "磁控微纳机器人"),
    (("micro", "nano", "robots"), "微纳机器人与微尺度操控"),
    (("slam", "mapping"), "机器人 SLAM 与定位建图"),
    (("parking", "mapping"), "停车场与室内场景建图"),
    (("tactile", "sensor"), "机器人触觉传感与力感知"),
    (("inverse kinematics", "manipulators"), "机器人逆运动学与机械臂控制"),
    (("planetary", "tracking"), "空间机器人与在轨装配"),
    (("pose estimation", "object pose"), "物体姿态估计与机器人感知"),
    (("predictive", "vehicles", "autonomous"), "自动驾驶预测控制"),
    (("gait", "exoskeleton"), "外骨骼步态与假肢控制"),
    (("path integral", "mpc"), "MPC 与安全路径规划"),
    (("underwater", "communication"), "水下机器人通信与识别"),
    (("underwater", "control", "vessels"), "水下/海洋机器人控制"),
    (("open-vocabulary", "graphs"), "开放词汇场景图与语义建图"),
    (("exoskeleton", "actuators"), "外骨骼执行器与力反馈"),
    (("visual servoing",), "视觉伺服与物理仿真"),
    (("optical flow", "event-based"), "事件相机光流与运动估计"),
    (("behaviors", "robots"), "机器人行为识别与人机交互"),
    (("underwater", "target", "localization"), "水下目标定位与 AUV 协同"),
    (("tracking", "object pose"), "高速目标跟踪与姿态估计"),
    (("detection", "domain", "object"), "机器人视觉目标检测与域适应"),
    (("navigation", "zero-shot", "visual"), "视觉语义导航与开放词汇定位"),
    (("navigation", "visual"), "视觉导航与语义地图"),
    (("tactile", "manipulation"), "机器人触觉感知与操作"),
    (("tactile", "robotic"), "机器人触觉感知与操作"),
    (("robotic", "manipulation"), "机器人操作与抓取"),
    (("robot", "manipulation"), "机器人操作与抓取"),
    (("grasping", "robotic"), "机器人操作与抓取"),
    (("imitation", "demonstrations"), "模仿学习与机器人示教"),
    (("dexterous", "gripper"), "灵巧操作与夹爪控制"),
    (("dexterous", "manipulation"), "灵巧机器人操作"),
    (("sim-to-real", "manipulation"), "机器人操作的 Sim-to-Real 迁移"),
    (("sim2real", "manipulation"), "机器人操作的 Sim-to-Real 迁移"),
    (("quadruped",), "足式机器人与运动控制"),
    (("legged",), "足式机器人与运动控制"),
    (("humanoid",), "人形机器人与全身控制"),
    (("locomotion", "robot"), "机器人运动控制与移动能力"),
    (("soft", "robot"), "软体机器人与柔性执行器"),
    (("soft", "robotic"), "软体机器人与柔性执行器"),
    (("underwater", "robot"), "水下机器人与海洋自主系统"),
    (("aerial", "robot"), "无人机/空中机器人规划与控制"),
    (("quadrotor",), "无人机/四旋翼规划与控制"),
    (("uav",), "无人机/四旋翼规划与控制"),
    (("drone",), "无人机/四旋翼规划与控制"),
    (("teleoperation",), "机器人遥操作与触觉交互"),
    (("haptic",), "机器人遥操作与触觉交互"),
    (("robotic", "surgery"), "手术机器人与医学机器人"),
    (("surgical", "robotic"), "手术机器人与医学机器人"),
    (("endoscopic", "robotic"), "手术机器人与医学机器人"),
    (("agricultural", "robotic"), "农业机器人与自主采摘"),
    (("harvesting", "robotic"), "农业机器人与自主采摘"),
    (("slam", "robot"), "机器人 SLAM 与定位建图"),
    (("slam", "lidar"), "机器人 SLAM 与定位建图"),
    (("lora",), "LoRA 与参数高效微调"),
    (("peft",), "参数高效微调与模型适配"),
    (("long-context",), "长上下文建模与压缩"),
    (("cuda", "llms"), "LLM 推理系统与 GPU Kernel 优化"),
    (("gpu", "llms"), "LLM 推理系统与 GPU Kernel 优化"),
    (("kernels", "gpu"), "GPU Kernel 与高性能深度学习系统"),
    (("code generation",), "代码生成与程序理解"),
    (("dialogue", "conversation"), "对话系统与会话建模"),
    (("open-domain", "responses"), "开放域对话与响应生成"),
    (("entity", "relation", "ner"), "信息抽取、实体识别与关系抽取"),
    (("entities", "relation"), "信息抽取、实体识别与关系抽取"),
    (("combinatorial", "optimization"), "组合优化与神经求解"),
    (("optimization problems", "mip"), "组合优化与神经求解"),
    (("neural architecture",), "神经架构搜索与模型设计"),
    (("nas",), "神经架构搜索与模型设计"),
    (("hate speech",), "仇恨言论与有害内容检测"),
    (("offensive", "social"), "仇恨言论与有害内容检测"),
    (("stance", "offensive"), "立场、仇恨言论与有害内容检测"),
    (("medical", "segmentation"), "医学影像分割"),
    (("mri", "segmentation"), "医学影像分割"),
    (("cartilage", "segmentation"), "医学影像分割"),
    (("prostate", "segmentation"), "医学影像分割"),
    (("brain", "segmentation"), "医学影像分割"),
    (("lung", "segmentation"), "医学影像分割"),
    (("chest", "x-ray"), "医学影像分析"),
    (("pathological", "segmentation"), "病理图像与医学影像分割"),
    (("meta-learning", "maml"), "元学习、MAML 与少样本适应"),
    (("maml",), "元学习、MAML 与少样本适应"),
    (("domain", "adaptation"), "领域自适应与迁移学习"),
    (("neural machine",), "机器翻译与跨语言对齐"),
    (("machine translation",), "机器翻译与跨语言对齐"),
    (("nmt",), "机器翻译与跨语言对齐"),
    (("wmt",), "机器翻译与跨语言对齐"),
    (("translation", "cross-lingual"), "机器翻译与跨语言对齐"),
    (("translation", "multilingual"), "机器翻译与跨语言对齐"),
    (("translation", "languages"), "机器翻译与跨语言对齐"),
    (("speech", "translation"), "语音翻译与跨语言语音处理"),
    (("sign", "language", "translation"), "手语识别与视觉语言翻译"),
    (("captioning", "translation"), "图像/视频描述与视觉语言生成"),
    (("captioning", "multilingual"), "多语言图像/视频描述与视觉语言生成"),
    (("image-text", "multilingual"), "多语言视觉语言预训练与跨模态检索"),
    (("image-to-image",), "图像到图像转换与风格迁移"),
    (("i2i",), "图像到图像转换与风格迁移"),
    (("unpaired", "translation"), "图像到图像转换与风格迁移"),
    (("domain translation",), "跨域生成与风格迁移"),
    (("camera", "pose", "rotation"), "相机姿态估计与几何重建"),
    (("pose", "rotation", "translation"), "相机姿态估计与几何重建"),
    (("multimodal", "fusion"), "多模态融合与跨模态表示"),
    (("query", "program", "predicates"), "逻辑查询、约束求解与程序推理"),
    (("multilingual",), "多语言建模与跨语言迁移"),
    (("translation",), "跨域转换与序列转换"),
    (("document", "summarization"), "文档摘要与信息压缩"),
    (("text", "summarization"), "文档摘要与信息压缩"),
    (("video", "summarization"), "视频摘要与精彩片段检测"),
    (("visual", "summarization"), "视频/视觉摘要与信息压缩"),
    (("clustering", "summarization"), "数据摘要、聚类与原型选择"),
    (("summarization",), "摘要生成与信息压缩"),
    (("answer", "question"), "问答生成、阅读理解与答案选择"),
    (("semantic parsing",), "语义解析与结构化语言理解"),
    (("dependency", "parsing"), "句法解析与语言结构建模"),
    (("syntactic", "parsing"), "句法解析与语言结构建模"),
    (("scene graph", "parsing"), "场景图、关系推理与视觉理解"),
    (("scene graph",), "场景图、关系推理与视觉理解"),
    (("human parsing",), "人体解析与细粒度分割"),
    (("face parsing",), "人脸解析与细粒度分割"),
    (("object", "segmentation", "instance"), "实例分割与开放词汇分割"),
    (("parsing",), "结构化解析与表示学习"),
    (("syntactic",), "句法知识与语言学分析"),
    (("machine learning", "explanations"), "可解释机器学习与数据科学工具"),
    (("autoregressive", "transformer"), "自回归 Transformer 与语言建模"),
    (("few-shot", "fine-grained"), "少样本细粒度视觉识别"),
    (("few-shot", "classification"), "少样本分类与开放集识别"),
    (("zero-shot", "classification"), "零样本分类与跨模态识别"),
    (("fine-grained", "classification"), "细粒度视觉分类与识别"),
    (("fine-grained", "recognition"), "细粒度视觉分类与识别"),
    (("long-tailed", "classification"), "长尾视觉识别与分类"),
    (("medical image", "segmentation"), "医学影像分割"),
    (("medical image",), "医学影像分析"),
    (("lesion",), "医学影像病灶分析"),
    (("pathology",), "病理图像与临床 AI"),
    (("clinical",), "医疗健康与临床 AI"),
    (("healthcare",), "医疗健康与临床 AI"),
    (("diffusion", "denoising"), "扩散生成模型"),
    (("diffusion", "generative"), "扩散生成模型"),
    (("score-based",), "扩散生成模型"),
    (("video", "retrieval"), "视频检索、时刻定位与事件理解"),
    (("moment", "localization"), "视频时刻定位与文本检索"),
    (("video", "action"), "视频动作识别与时序定位"),
    (("temporal", "action"), "视频动作识别与时序定位"),
    (("action", "recognition"), "视频动作识别与时序定位"),
    (("action", "temporal"), "视频动作识别与时序定位"),
    (("event", "camera"), "事件相机与高动态范围视觉"),
    (("dynamic range", "camera"), "事件相机与高动态范围视觉"),
    (("hand", "human-object"), "手部姿态与人-物交互"),
    (("human-object", "contact"), "手部姿态与人-物交互"),
    (("pose", "human", "estimation"), "人体姿态估计与运动理解"),
    (("human", "pose"), "人体姿态估计与运动理解"),
    (("person", "re-identification"), "行人重识别与人群计数"),
    (("crowd", "counting"), "行人重识别与人群计数"),
    (("albedo", "light"), "材质、光照与反射率估计"),
    (("spatially-varying", "surface"), "材质、光照与反射率估计"),
    (("implicit", "reconstruction", "surface"), "3D 形状重建与隐式表示"),
    (("shape", "implicit"), "3D 形状重建与隐式表示"),
    (("chinese", "subword"), "中文分词、字符与子词建模"),
    (("segmentation", "chinese"), "中文分词、字符与子词建模"),
    (("heads", "pruning"), "Transformer 剪枝与注意力头分析"),
    (("batch", "normalization"), "归一化、网络层与训练稳定性"),
    (("convolutional", "cnns"), "卷积网络结构与训练"),
    (("facial", "face"), "人脸分析、表情与属性建模"),
    (("face", "facial"), "人脸分析、表情与属性建模"),
    (("talking", "head"), "说话人视频生成与人脸动画"),
    (("noisy", "labels"), "噪声标签学习与半监督鲁棒训练"),
    (("semi-supervised", "labels"), "半监督学习与噪声标签建模"),
    (("zero-shot", "classes"), "零样本分类与开放集识别"),
    (("emotion", "recognition"), "情感识别与多模态情绪理解"),
    (("emotion", "multimodal"), "多模态情感理解"),
    (("entry", "keyboard"), "文本输入与移动交互"),
    (("mobile", "keyboard"), "文本输入与移动交互"),
    (("deepfakes",), "Deepfake 检测、多媒体取证与内容安全"),
    (("deepfake",), "Deepfake 检测、多媒体取证与内容安全"),
    (("gans", "generative"), "GAN 与图像生成"),
    (("gan", "generator"), "GAN 与图像生成"),
    (("generative", "adversarial networks"), "GAN 与图像生成"),
    (("generative adversarial",), "GAN 与图像生成"),
    (("adversarial network", "generation"), "GAN 与图像生成"),
    (("adversarial networks", "generation"), "GAN 与图像生成"),
    (("adversarial", "examples", "speech"), "语音与视觉对抗攻击"),
    (("adversarial", "examples", "audio"), "语音与音频对抗攻击"),
    (("attack", "speech"), "语音与音频对抗攻击"),
    (("attacks", "audio"), "语音与音频对抗攻击"),
    (("audio-visual",), "音视频多模态理解"),
    (("audio visual",), "音视频多模态理解"),
    (("tts",), "语音合成与音频生成"),
    (("asr",), "语音识别与语音理解"),
    (("speech", "recognition"), "语音识别与语音理解"),
    (("speech", "synthesis"), "语音合成与音频生成"),
    (("speaker",), "说话人建模与语音表征"),
    (("music",), "音乐与音频生成/理解"),
    (("sound", "event"), "声音事件检测与声学场景理解"),
    (("acoustic",), "语音、音频与声学建模"),
    (("voice", "assistant"), "语音交互与语音助手"),
    (("voice",), "语音交互与语音表征"),
    (("speech",), "语音理解、识别与交互"),
    (("audio",), "音频理解与生成"),
    (("emotion", "multimodal"), "多模态情感理解"),
    (("emotion", "recognition"), "情感识别与多模态情绪理解"),
    (("eeg",), "脑电信号表征与解码"),
    (("graph", "gnn"), "图神经网络与图表示学习"),
    (("graph", "gnns"), "图神经网络与图表示学习"),
    (("graph", "node"), "图神经网络与节点表示学习"),
    (("knowledge graph",), "知识图谱推理与表示学习"),
    (("multi-view", "clustering"), "多视图聚类与图学习"),
    (("recommendation",), "推荐系统与用户建模"),
    (("recommender",), "推荐系统与用户建模"),
    (("ranking",), "搜索排序与相关性建模"),
    (("query", "retrieval"), "查询理解与检索优化"),
    (("query", "ranking"), "查询理解与检索优化"),
    (("generative retrieval",), "生成式检索"),
    (("vision-language-action",), "视觉语言动作模型与具身操作"),
    (("vla",), "视觉语言动作模型与具身操作"),
    (("text-to-image",), "文生图生成与个性化编辑"),
    (("t2i",), "文生图生成与个性化编辑"),
    (("video", "diffusion"), "视频扩散生成与运动控制"),
    (("video", "retrieval"), "视频检索、时刻定位与事件理解"),
    (("moment", "localization"), "视频时刻定位与文本检索"),
    (("temporal", "action"), "视频动作识别与时序定位"),
    (("action", "recognition"), "视频动作识别与时序定位"),
    (("action", "temporal"), "视频动作识别与时序定位"),
    (("motion", "video"), "视频动作生成与运动控制"),
    (("diffusion",), "扩散生成模型"),
    (("multimodal",), "多模态学习与跨模态理解"),
    (("gaussian", "splatting"), "3D Gaussian Splatting 与场景重建"),
    (("nerf",), "NeRF 与神经渲染"),
    (("radiance",), "NeRF 与神经渲染"),
    (("avatar",), "3D Avatar 与人脸头部建模"),
    (("talking", "head"), "说话人视频生成与人脸动画"),
    (("facial", "face"), "人脸分析、表情与属性建模"),
    (("face", "facial"), "人脸分析、表情与属性建模"),
    (("depth",), "深度估计与立体匹配"),
    (("stereo",), "深度估计与立体匹配"),
    (("lidar",), "LiDAR 点云与 3D 感知"),
    (("point", "cloud"), "点云表示与 3D 感知"),
    (("autonomous driving",), "自动驾驶感知与世界模型"),
    (("driving",), "自动驾驶感知与世界模型"),
    (("robot",), "机器人操作与具身智能"),
    (("embodied",), "具身智能与物理交互"),
    (("time series",), "时间序列建模与预测"),
    (("time-series",), "时间序列建模与预测"),
    (("forecasting",), "时间序列预测"),
    (("medical", "segmentation"), "医学影像分割"),
    (("instance", "segmentation"), "实例分割与开放词汇分割"),
    (("semantic", "segmentation"), "语义分割与场景解析"),
    (("segmentation",), "图像分割、语义分割与场景解析"),
    (("object detection",), "目标检测与开放世界检测"),
    (("restoration",), "图像复原与超分辨率"),
    (("super-resolution",), "图像复原与超分辨率"),
    (("noisy", "labels"), "噪声标签学习与半监督鲁棒训练"),
    (("semi-supervised", "labels"), "半监督学习与噪声标签建模"),
    (("zero-shot", "classes"), "零样本分类与开放集识别"),
    (("batch", "normalization"), "归一化、网络层与训练稳定性"),
    (("convolutional", "cnns"), "卷积网络结构与训练"),
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
    (("llm", "agent"), "LLM Agent 与工具使用"),
    (("llm", "agents"), "LLM Agent 与工具使用"),
    (("llms", "agent"), "LLM Agent 与工具使用"),
    (("llms", "agents"), "LLM Agent 与工具使用"),
    (("large language", "agent"), "LLM Agent 与工具使用"),
    (("tool", "agent"), "LLM Agent 与工具使用"),
    (("tool learning",), "LLM Agent 与工具使用"),
    (("ai agents",), "AI Agent、人机协作与交互评估"),
    (("conversational", "agents"), "对话系统与会话智能体"),
    (("dialogue", "agents"), "对话系统与会话智能体"),
    (("allocations", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("allocation", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("valuations", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("auction", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("auctions", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("mechanisms", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("games", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("equilibrium", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("equilibria", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("matchings", "agents"), "多智能体博弈、机制设计与社会选择"),
    (("fair division",), "公平分配、机制设计与社会选择"),
    (("voting", "ranking"), "投票规则、排序聚合与社会选择"),
    (("argumentation", "agents"), "多智能体逻辑、论辩与社会选择"),
    (("multi-agent",), "多智能体协作与规划"),
    (("agents",), "智能体决策与多智能体系统"),
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
    "few-shot": "少样本",
    "zero-shot": "零样本",
    "fine-grained": "细粒度识别",
    "classification": "分类",
    "medical": "医学AI",
    "clinical": "临床AI",
    "health": "健康AI",
    "healthcare": "医疗健康",
    "open-set": "开放集",
    "long-tailed": "长尾",
    "privacy": "隐私",
    "fairness": "公平性",
    "robust": "鲁棒性",
    "attack": "攻击防御",
    "optimization": "优化",
    "forecasting": "预测",
}

QUALIFIER_RULES: list[tuple[tuple[str, ...], str]] = [
    (("locomotion", "bipedal"), "双足步态控制"),
    (("locomotion", "quadruped"), "四足步态控制"),
    (("locomotion", "humanoid"), "人形越野运动"),
    (("terrain", "learning"), "复杂地形学习"),
    (("continuum", "manipulator"), "连续体机械臂"),
    (("cable-driven",), "绳驱机构"),
    (("magnetic", "microrobots"), "磁控微纳机器人"),
    (("cell", "microrobots"), "细胞/微尺度机器人"),
    (("surgical", "endoscopic"), "内窥镜手术器械"),
    (("steerable", "instrument"), "可转向手术器械"),
    (("soft", "actuators"), "软体执行器"),
    (("soft", "grasping"), "软体夹爪抓取"),
    (("hand", "dexterous"), "灵巧手操作"),
    (("bimanual",), "双手协作"),
    (("grasping", "gripper"), "夹爪与抓取"),
    (("manipulation", "language"), "语言引导操作"),
    (("manipulation", "diffusion"), "扩散策略操作"),
    (("imitation", "visuomotor"), "视觉运动模仿"),
    (("navigation", "reinforcement"), "强化学习导航"),
    (("navigation", "language"), "语言引导导航"),
    (("grounding", "object"), "语义接地与物体感知"),
    (("slam", "mapping"), "SLAM 与建图"),
    (("lidar", "segmentation"), "LiDAR 语义分割"),
    (("lidar", "detection"), "LiDAR 目标检测"),
    (("trajectory", "quadrotor"), "四旋翼轨迹规划"),
    (("uav", "detection"), "无人机检测感知"),
    (("multi-agent", "path"), "多智能体路径规划"),
    (("multi-robot", "exploration"), "多机器人协同探索"),
    (("grasping", "cluttered"), "杂乱场景抓取"),
    (("localization", "camera"), "相机定位"),
    (("odometry", "lidar"), "LiDAR 里程计"),
    (("calibration", "extrinsic"), "外参标定"),
    (("decentralized", "multi-robot"), "分布式多机器人编队"),
    (("multi-robot", "allocation"), "多机器人任务分配"),
    (("trajectory prediction", "forecasting"), "轨迹预测"),
    (("underwater", "swimming"), "水下游动机器人"),
    (("driving", "traffic"), "交通交互与自动驾驶"),
    (("diffusion", "video"), "视频扩散"),
    (("diffusion", "denoising"), "去噪扩散"),
    (("diffusion", "editing"), "扩散编辑"),
    (("diffusion", "policy"), "扩散策略"),
    (("diffusion", "text-to-image"), "文生图"),
    (("diffusion", "audio"), "音频扩散"),
    (("diffusion", "molecular"), "分子扩散"),
    (("depth", "monocular"), "单目深度"),
    (("depth", "stereo"), "双目/立体深度"),
    (("depth", "completion"), "深度补全"),
    (("recommendation", "sequential"), "序列推荐"),
    (("recommendation", "cold-start"), "冷启动推荐"),
    (("recommendation", "graph"), "图推荐"),
    (("recommendation", "fairness"), "公平推荐"),
    (("retrieval", "cross-modal"), "跨模态检索"),
    (("retrieval", "video"), "视频检索"),
    (("ranking", "query"), "查询排序"),
    (("vision-language", "reasoning"), "视觉语言推理"),
    (("vision-language", "segmentation"), "视觉语言分割"),
    (("vision-language", "retrieval"), "视觉语言检索"),
    (("captioning",), "图像/视频描述"),
    (("vqa",), "视觉问答"),
    (("open-vocabulary",), "开放词汇"),
    (("prompt", "tuning"), "提示调优"),
    (("chain-of-thought",), "思维链推理"),
    (("agent", "tool"), "工具使用"),
    (("agent", "web"), "网页任务"),
    (("agent", "gui"), "GUI 操作"),
    (("multi-agent", "game"), "多智能体博弈"),
    (("bandit",), "Bandit"),
    (("offline", "reinforcement"), "离线强化学习"),
    (("reward", "preference"), "奖励/偏好建模"),
    (("privacy",), "隐私保护"),
    (("fairness",), "公平性"),
    (("adversarial", "attack"), "对抗攻击"),
    (("backdoor",), "后门安全"),
    (("time series", "forecasting"), "时间序列预测"),
    (("protein",), "蛋白质建模"),
    (("molecule",), "分子建模"),
    (("medical", "segmentation"), "医学分割"),
    (("clinical", "prediction"), "临床预测"),
]

QUALIFIER_TERM_CN = TERM_CN | {
    "humanoid": "人形",
    "quadruped": "四足",
    "bipedal": "双足",
    "locomotion": "运动控制",
    "terrain": "地形",
    "continuum": "连续体",
    "manipulator": "机械臂",
    "cable-driven": "绳驱",
    "magnetic": "磁控",
    "microrobots": "微纳机器人",
    "micro": "微尺度",
    "nano": "纳米",
    "surgical": "手术",
    "endoscopic": "内窥镜",
    "steerable": "可转向",
    "soft": "软体",
    "actuators": "执行器",
    "dexterous": "灵巧",
    "bimanual": "双手",
    "grasping": "抓取",
    "gripper": "夹爪",
    "navigation": "导航",
    "grounding": "语义接地",
    "semantic": "语义",
    "slam": "SLAM",
    "mapping": "建图",
    "lidar": "LiDAR",
    "quadrotor": "四旋翼",
    "uav": "无人机",
    "drone": "无人机",
    "underwater": "水下",
    "swimming": "游动",
    "diffusion": "扩散",
    "denoising": "去噪",
    "editing": "编辑",
    "text-to-image": "文生图",
    "monocular": "单目",
    "stereo": "立体",
    "completion": "补全",
    "sequential": "序列",
    "cold-start": "冷启动",
    "cross-modal": "跨模态",
    "open-vocabulary": "开放词汇",
    "captioning": "图像/视频描述",
    "vqa": "视觉问答",
    "prompt": "提示",
    "tuning": "调优",
    "tool": "工具",
    "gui": "GUI",
    "web": "网页",
    "offline": "离线",
    "bandit": "Bandit",
    "backdoor": "后门",
    "clinical": "临床",
}

GENERIC_QUALIFIER_TERMS = {
    "learning",
    "model",
    "models",
    "method",
    "methods",
    "data",
    "task",
    "tasks",
    "deep",
    "neural",
    "network",
    "networks",
    "using",
    "based",
    "control",
    "robot",
    "robots",
    "robotic",
    "system",
    "systems",
    "efficient",
    "robust",
    "improving",
    "new",
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


def topic_text(row: pd.Series, include_representatives: bool = False) -> str:
    parts = [
        str(row.get("topic_label", "")),
        str(row.get("keywords", "")),
    ]
    if include_representatives:
        parts.append(str(row.get("representative_titles", "")))
    return " ".join(parts).lower()


def display_macro_topic(row: pd.Series) -> str:
    text = topic_text(row)
    venue = str(row.get("venue", "")).upper()
    robotics_context = (
        venue in ROBOTICS_VENUES
        or any(
            contains_term(text, term)
            for term in (
                "robot",
                "robotic",
                "robotics",
                "multi-robot",
                "human-robot",
                "embodied",
                "manipulation",
                "grasping",
                "locomotion",
                "slam",
                "quadrotor",
                "uav",
                "drone",
                "teleoperation",
                "haptic",
            )
        )
    )
    if robotics_context:
        return "3D/具身/机器人"
    if contains_term(text, "point") and contains_term(text, "cloud"):
        return "3D/具身/机器人"
    if any(
        contains_term(text, term)
        for term in (
            "medical",
            "clinical",
            "healthcare",
            "patient",
            "disease",
            "lesion",
            "pathology",
            "mri",
            "ct",
            "x-ray",
            "surgical",
            "surgery",
            "brain",
            "lung",
            "chest",
            "tumor",
        )
    ):
        return "AI4Science/医疗"
    if any(
        contains_term(text, term)
        for term in (
            "vision-language",
            "vision language",
            "image-text",
            "captioning",
            "vqa",
            "vlm",
            "vlms",
            "mllm",
            "mllms",
            "multimodal",
            "cross-modal",
            "audio-visual",
            "sign language",
        )
    ):
        return "多模态/VLM"
    if any(contains_term(text, term) for term in ("gan", "gans", "generator", "generative adversarial")):
        return "生成模型"
    if any(
        contains_term(text, term)
        for term in ("adversarial", "attack", "privacy", "fairness", "bias", "safety", "deepfake", "deepfakes")
    ):
        return "可信/安全/公平"
    if any(
        contains_term(text, term)
        for term in (
            "image-to-image",
            "i2i",
            "object detection",
            "segmentation",
            "semantic segmentation",
            "instance segmentation",
            "face",
            "facial",
            "video",
            "action recognition",
            "moment localization",
            "scene graph",
            "human parsing",
            "camera",
            "event camera",
            "dynamic range",
            "pose",
            "tracking",
            "restoration",
            "super-resolution",
        )
    ):
        return "计算机视觉"
    if any(
        contains_term(text, term)
        for term in ("speech", "audio", "music", "sound", "acoustic", "speaker", "voice", "tts", "asr")
    ):
        return "语音/音频/音乐"
    if any(
        contains_term(text, term)
        for term in (
            "recommendation",
            "recommender",
            "retrieval",
            "ranking",
            "click",
        )
    ):
        return "推荐/检索/排序"
    if any(
        contains_term(text, term)
        for term in ("cuda", "gpu", "kernel", "kernels", "quantization", "pruning", "compression", "serving")
    ):
        return "系统/效率/压缩"
    if any(
        contains_term(text, term)
        for term in (
            "nmt",
            "machine translation",
            "cross-lingual",
            "summarization",
            "dialogue",
            "parsing",
            "ner",
            "entity",
            "entities",
            "relation",
        )
    ):
        return "NLP任务"
    if any(
        contains_term(text, term)
        for term in ("neural architecture", "nas", "combinatorial", "optimization problems", "mip")
    ):
        return "理论/优化"
    if any(contains_term(text, term) for term in ("agent", "agents", "game", "games", "policy", "reward", "planning")):
        return "强化学习/决策"
    return str(row.get("macro_topic", ""))


def topic_name_cn(row: pd.Series) -> str:
    text = topic_text(row)
    for required_terms, name in CN_TOPIC_RULES:
        if all(contains_term(text, term) for term in required_terms):
            return name

    terms = [term.strip().lower() for term in str(row.get("topic_label", "")).split("/") if term.strip()]
    translated = []
    for term in terms:
        translated.append(TERM_CN.get(term, term))
    if translated:
        return " / ".join(translated[:4])
    return f"{display_macro_topic(row) or '主题'}细分方向"


def topic_qualifier_cn(row: pd.Series) -> str:
    text = topic_text(row)
    label_text = str(row.get("topic_label", "")).lower()
    for required_terms, qualifier in QUALIFIER_RULES:
        if all(contains_term(label_text, term) for term in required_terms):
            return qualifier
    for required_terms, qualifier in QUALIFIER_RULES:
        if all(contains_term(text, term) for term in required_terms):
            return qualifier

    candidates: list[str] = []
    for raw_part in str(row.get("topic_label", "")).split("/"):
        candidates.append(raw_part.strip().lower())
    for raw_part in str(row.get("keywords", "")).split(";"):
        candidates.append(raw_part.strip().lower())

    translated: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        if not term or term in seen or term in GENERIC_QUALIFIER_TERMS:
            continue
        seen.add(term)
        cn = QUALIFIER_TERM_CN.get(term, term)
        if len(cn) > 28:
            continue
        translated.append(cn)
        if len(translated) >= 2:
            break

    if translated:
        return " / ".join(translated)
    return f"Topic {int(row.get('topic_id', 0)):03d}"


def unique_topic_names(topics: pd.DataFrame) -> dict[int, str]:
    base_names = {idx: topic_name_cn(topic) for idx, topic in topics.iterrows()}
    duplicate_counts = Counter(base_names.values())
    used: set[str] = set()
    names: dict[int, str] = {}
    for idx, topic in topics.iterrows():
        topic_id = int(topic["topic_id"])
        base_name = base_names[idx]
        name = base_name
        if duplicate_counts[base_name] > 1:
            name = f"{base_name}：{topic_qualifier_cn(topic)}"
        if name in used:
            name = f"{name}（Topic {topic_id:03d}）"
        used.add(name)
        names[topic_id] = name
    return names


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
    cn_name: str | None = None,
) -> None:
    topic_id = int(topic_row["topic_id"])
    topic_records = [record for record in records if int(record.get("fine_topic", -999999)) == topic_id]
    cn_name = cn_name or topic_name_cn(topic_row)
    lines = [
        f"# {venue} {year}: {cn_name}",
        "",
        f"- Topic ID: `{topic_id}`",
        f"- Papers: **{int(topic_row['paper_count'])}** ({float(topic_row['paper_share']) * 100:.2f}%)",
        f"- Macro topic: {md_escape(display_macro_topic(topic_row))}",
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
    topic_names: dict[int, str],
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
        cn_name = topic_names[int(topic["topic_id"])]
        representative = str(topic.get("representative_titles", "")).split(" || ")[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    topic_link,
                    table_escape(cn_name),
                    str(int(topic["paper_count"])),
                    f"{float(topic['paper_share']) * 100:.2f}%",
                    table_escape(display_macro_topic(topic)),
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
    venue_groups = [
        ("ML / learning theory", ["ICLR", "ICML", "NeurIPS"]),
        ("CV top conferences", ["CVPR", "ICCV", "ECCV"]),
        ("NLP / language", ["ACL", "EMNLP", "NAACL", "COLM"]),
        ("General AI", ["AAAI", "IJCAI"]),
        ("Embodied AI / robotics", ["ICRA", "IROS", "RSS"]),
        ("Multimedia / graphics / HCI", ["ACMMM", "SIGGRAPH", "SIGGRAPH-Asia", "CHI"]),
        ("Data mining / IR / Web / DB", ["KDD", "SIGIR", "WWW", "ICDE", "SIGMOD"]),
        ("Medical AI", ["MICCAI"]),
        ("Journals: ML / general AI", ["AIJ", "JMLR", "TNNLS"]),
        ("Journals: vision / image", ["TPAMI", "IJCV", "TIP", "PR"]),
        ("Journals: multimedia / data", ["TMM", "TKDE"]),
    ]
    planned_additions = [
        ("ML / AI", "AISTATS, UAI, COLT, JAIR, Machine Learning"),
        ("CV / graphics", "WACV, BMVC, ACCV, 3DV, TVCG"),
        ("NLP / speech", "EACL, TACL, Computational Linguistics, Interspeech"),
        ("Robotics / embodied AI", "CoRL, RA-L, T-RO, IJRR, Autonomous Robots"),
        ("Data / IR / Web", "CIKM, WSDM, RecSys, ICDM, SDM, VLDB, EDBT, PODS"),
        ("HCI / systems", "UIST, CSCW, IMWUT, UbiComp"),
        ("Medical AI", "TMI, Medical Image Analysis, ISBI"),
    ]
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
        "Continuously updated fine-grained topic index generated from AI conference and journal papers.",
        "",
        "Navigation pattern: **year -> venue -> topic -> paper**.",
        "",
        "The numbers below describe the current checked-in index. They are expected to grow as new "
        "venues, years, and proceedings are added.",
        "",
        f"- Indexed venue-year groups: **{len(summary)}**",
        f"- Indexed papers: **{int(summary['papers'].sum()):,}**",
        f"- Fine-grained topic pages: **{int(summary['final_topics'].sum()):,}**",
        f"- Unassigned papers after reassignment: **{int(summary['final_outliers'].sum())}**",
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
            "## Venue Groups",
            "",
            "| Group | Indexed venues | Venue-years | Papers | Fine topics |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for group_name, venues in venue_groups:
        group_rows = summary[summary["venue"].isin(venues)]
        indexed_venues = [venue for venue in venues if venue in set(group_rows["venue"])]
        lines.append(
            f"| {group_name} | {', '.join(indexed_venues)} | {len(group_rows)} "
            f"| {int(group_rows['papers'].sum()):,} | {int(group_rows['final_topics'].sum()):,} |"
        )

    lines.extend(
        [
            "",
            "## Planned Additions",
            "",
            "Candidate sources are tracked for future expansion. Inclusion depends on public metadata "
            "availability and source quality.",
            "",
            "| Area | Candidate venues and journals |",
            "|---|---|",
        ]
    )
    for area, candidates in planned_additions:
        lines.append(f"| {area} | {candidates} |")

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
            "Chinese display names are disambiguated within each venue-year when multiple fine topics share "
            "the same base label. Use representative paper titles for audit.",
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
        topic_names = unique_topic_names(topics)
        records = read_jsonl(papers_path)
        venue_output_dir = output_root / str(year) / venue
        venue_page = venue_output_dir / "README.md"

        for _, topic in topics.iterrows():
            topic_file = venue_output_dir / f"topic-{int(topic['topic_id']):03d}.md"
            topic_id = int(topic["topic_id"])
            build_topic_page(output_root, year, venue, topic, records, topic_file, topic_names[topic_id])
            topic_record = topic.to_dict()
            topic_record["macro_topic"] = display_macro_topic(topic)
            topic_record["topic_name_cn"] = topic_names[topic_id]
            topic_record["topic_page"] = str(topic_file.relative_to(output_root)).replace("\\", "/")
            topic_rows.append(topic_record)

        build_venue_page(output_root, year, venue, group, topics, venue_page, topic_names)

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
