"""Post-process BERTopic outputs with more specific topic names.

The clustering step intentionally stays model-only and deterministic. This
script adds an interpretation layer using:
- c-TF-IDF topic keywords;
- representative paper titles nearest to each topic centroid;
- venue/year counts already produced by the yearly run.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = "results/yearly_main_accepted_topics_bge_m30"
DEFAULT_MODEL_SAFE = "models__bge-base-en-v1.5"

GENERIC_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "by",
    "data",
    "dataset",
    "datasets",
    "deep",
    "efficient",
    "enhanced",
    "for",
    "from",
    "in",
    "is",
    "large",
    "learning",
    "model",
    "models",
    "network",
    "networks",
    "neural",
    "new",
    "of",
    "on",
    "paper",
    "papers",
    "robust",
    "the",
    "to",
    "towards",
    "training",
    "using",
    "via",
    "with",
}

MODEL_EFFICIENCY_TERMS = (
    "decoding",
    "tokens",
    "token",
    "longcontext",
    "long context",
    "kv cache",
    "cache",
    "quantization",
    "quantized",
    "pruning",
    "speculative",
)

MULTILINGUAL_TERMS = (
    "translation",
    "machine translation",
    "multilingual",
    "crosslingual",
    "cross lingual",
    "lowresource",
    "low resource",
    "languages",
    "nmt",
)

RL_TERMS = (
    "reinforcement",
    "reinforcement learning",
    "policy",
    "policies",
    "reward",
    "offline rl",
    "mdp",
    "q learning",
    "qlearning",
    "actor critic",
)

BANDIT_TERMS = (
    "bandit",
    "bandits",
    "regret",
    "arm",
    "arms",
)

OPTIMIZATION_TERMS = (
    "optimization",
    "gradient",
    "gradients",
    "sgd",
    "adam",
    "convex",
    "nonconvex",
    "stochastic",
    "convergence",
    "rates",
    "minimax",
)

BAYESIAN_TERMS = (
    "bayesian",
    "variational",
    "posterior",
    "probabilistic",
    "gaussian process",
    "mcmc",
    "monte carlo",
)


def split_keywords(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value or "").split(",") if item.strip()]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("-", " ")).lower()


def has_any(text: str, needles: tuple[str, ...]) -> bool:
    for needle in needles:
        normalized = normalize_text(needle)
        if not normalized:
            continue
        if " " in normalized:
            pattern = r"\b" + r"\s+".join(re.escape(part) for part in normalized.split()) + r"\b"
        else:
            pattern = r"\b" + re.escape(normalized) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def title_terms(titles: list[str], top_n: int = 8) -> str:
    tokens: list[str] = []
    for title in titles:
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", title.lower()):
            if token not in GENERIC_WORDS:
                tokens.append(token)
    return " / ".join(term for term, _ in Counter(tokens).most_common(top_n))


def pick_phrase(text: str, keywords: list[str]) -> str:
    for keyword in keywords:
        if keyword and keyword != "outlier":
            return keyword
    terms = [term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", text.lower()) if term not in GENERIC_WORDS]
    return " / ".join(terms[:3]) if terms else "misc"


def specific_label_cn(topic_keywords: str, representative_titles: list[str]) -> tuple[str, str, str]:
    keywords = split_keywords(topic_keywords)
    kw_text = normalize_text(topic_keywords)
    top_kw_text = normalize_text(", ".join(keywords[:5]))
    rep_terms = title_terms(representative_titles, top_n=12)
    full_title_text = normalize_text(" ".join(str(title) for title in representative_titles))
    text = normalize_text(topic_keywords + " " + rep_terms + " " + full_title_text)

    # Strong top-keyword overrides. Representative titles are useful, but a
    # single phrase in one title should not outweigh the c-TF-IDF topic signal.
    if has_any(top_kw_text, ("retrieval", "rag", "retrieval augmented", "retrieval-augmented", "retriever")):
        if has_any(kw_text, ("recommendation", "recommender", "user", "item")):
            return "推荐系统：检索增强推荐、排序与个性化", "Recommendation", "retrieval-enhanced recommendation/ranking"
        if has_any(top_kw_text, ("rag", "retrieval augmented", "retrieval-augmented")) or has_any(
            kw_text,
            ("retrieval augmented generation", "retrieval-augmented generation"),
        ):
            return "检索增强大模型：RAG、知识注入与问答", "LLM / IR", "RAG/knowledge-grounded QA"
        if has_any(kw_text, ("llm", "llms", "language model", "generation")) and has_any(kw_text, ("knowledge", "answering")):
            return "检索增强大模型：RAG、知识注入与问答", "LLM / IR", "RAG/knowledge-grounded QA"
        if has_any(kw_text, ("multimodal", "cross-modal", "cross modal", "image", "video", "semantic")):
            return "多媒体检索：跨模态检索、语义匹配与内容理解", "Multimedia Retrieval", "cross-modal/semantic multimedia retrieval"
        if has_any(kw_text, ("ranking", "reranking", "queries", "query", "document", "search")):
            return "信息检索：搜索排序、文档检索与重排", "Information Retrieval", "search/ranking/document retrieval"
        return "信息检索：密集检索、向量召回与表示学习", "Information Retrieval", "dense/vector retrieval"

    if has_any(top_kw_text, ("code", "program", "programming", "software")):
        if has_any(kw_text, ("generation", "llm", "llms", "instruction", "benchmark")):
            return "代码大模型：代码生成、程序理解与评测", "Code Intelligence", "code LLMs/generation/evaluation"
        return "代码智能：程序分析、软件工程与代码检索", "Code Intelligence", "program analysis/software engineering"

    if has_any(top_kw_text, ("medical", "clinical", "patient", "healthcare", "pathology", "biomedical")):
        if has_any(kw_text, ("segmentation", "image", "images", "imaging", "tumor")):
            return "医疗视觉：医学影像分割、病理与临床影像", "AI for Health", "medical image segmentation/pathology"
        return "医疗大模型：临床推理、医学影像与健康问答", "Medical LLM", "clinical/medical large models"

    if has_any(top_kw_text, ("graph", "graphs", "clustering", "node")) and has_any(kw_text, ("anomaly", "detection", "fraud")):
        return "图学习：图异常检测、聚类与结构表示", "Graph Learning", "graph anomaly detection/clustering"

    if has_any(top_kw_text, ("recommendation", "recommender", "user", "item", "users", "items")):
        if has_any(kw_text, ("preference", "preferences", "alignment", "feedback")):
            return "推荐系统：偏好建模、反馈学习与个性化排序", "Recommendation", "preference feedback/recommendation ranking"
        return "推荐系统：排序、召回与点击率预测", "Recommendation", "ranking/retrieval/CTR recommendation"

    if has_any(top_kw_text, ("adversarial", "attack", "attacks", "anomaly", "deepfake", "privacy", "backdoor", "watermark")):
        if has_any(kw_text, ("deepfake", "forgery", "face", "audio", "visual")):
            return "多媒体安全：Deepfake检测、伪造识别与攻防", "Security / Vision", "deepfake/forgery/multimedia security"
        if has_any(kw_text, ("anomaly", "fraud", "outlier")):
            return "异常检测：图异常、欺诈检测与时序异常", "Anomaly Detection", "graph/fraud/time-series anomaly detection"
        return "可信安全：对抗攻击、后门、水印与隐私", "Security / Privacy", "adversarial/backdoor/watermark/privacy"

    if has_any(top_kw_text, ("domain", "generalization", "domains", "target", "classes")):
        if has_any(kw_text, ("few-shot", "few shot", "incremental", "class incremental")):
            return "迁移泛化：小样本、类增量与域泛化", "Robust / Transfer Learning", "few-shot/incremental/domain generalization"
        return "迁移泛化：域适应、OOD泛化与鲁棒表征", "Robust / Transfer Learning", "domain adaptation/OOD generalization"

    if has_any(top_kw_text, ("restoration", "superresolution", "super resolution", "denoising", "deblurring", "enhancement", "quality")):
        return "底层视觉：超分、去噪、增强与图像复原", "Computer Vision", "image restoration/super-resolution/enhancement"

    if has_any(top_kw_text, ("point", "depth", "pose", "camera", "lidar", "geometric", "estimation")):
        if has_any(kw_text, ("driving", "autonomous", "lidar", "trajectory")):
            return "自动驾驶感知：LiDAR、轨迹与三维检测", "Autonomous Driving", "LiDAR/trajectory/3D detection"
        return "三维视觉：点云、深度估计与相机姿态", "3D Vision", "point cloud/depth/pose"

    if has_any(top_kw_text, ("segmentation", "semantic", "panoptic", "mask")):
        if has_any(kw_text, ("medical", "clinical", "pathology")):
            return "医疗视觉：医学影像分割、病理与临床影像", "AI for Health", "medical image segmentation/pathology"
        if has_any(kw_text, ("open vocabulary", "openvocabulary", "clip", "language", "text")):
            return "开放词汇视觉：开放词汇检测、分割与CLIP语义", "Computer Vision", "open-vocabulary detection/segmentation"
        return "视觉感知：语义/实例/全景分割", "Computer Vision", "segmentation"

    if has_any(top_kw_text, ("time series", "forecasting")) or (
        has_any(top_kw_text, ("series",))
        and has_any(kw_text, ("time", "forecasting", "temporal", "dynamics"))
    ):
        return "时序建模：时间序列预测、动力系统与基础模型", "Time Series", "time-series forecasting/dynamical systems"

    if (
        has_any(top_kw_text, ("pruning", "compression", "token", "tokens", "attention", "quantization"))
        or has_any(kw_text, ("pruning", "compression", "quantization"))
    ) and has_any(
        kw_text,
        ("language", "llm", "llms", "visual", "vision"),
    ):
        return "高效大模型：推理加速、压缩与资源优化", "Efficient LLMs", "inference acceleration/compression"

    if has_any(top_kw_text, ("visual", "image", "multimodal", "vision", "video")) and has_any(
        kw_text,
        ("multimodal", "vision-language", "mllm", "mllms", "vlm", "vlms", "language", "reasoning", "understanding"),
    ) and not has_any(top_kw_text, ("generation", "diffusion", "generative")):
        if has_any(kw_text, ("reasoning", "llm", "llms", "language")):
            return "多模态大模型：视觉语言理解与跨模态推理", "Multimodal LLM", "multimodal/VLM understanding"
        return "多模态理解：视觉语言表征与跨模态对齐", "Multimodal AI", "vision-language representation/alignment"

    if has_any(top_kw_text, ("video", "videos", "action")) and not has_any(top_kw_text, ("generation", "diffusion", "generative")):
        if has_any(kw_text, ("retrieval", "moment", "language")):
            return "视频语言理解：视频检索、时刻定位与时序推理", "Video Understanding", "video-language retrieval/moment grounding"
        return "视频理解：动作识别、长视频与时序建模", "Video Understanding", "action/long-video/temporal modeling"

    if has_any(top_kw_text, ("audio", "speech", "music", "emotion", "emotional")):
        if has_any(kw_text, ("generation", "multimodal", "video")):
            return "多模态音视频生成：语音、音乐与情感生成", "Multimodal Generation", "audio/music/emotion generation"
        return "语音音频：ASR、说话人与音频理解", "Speech / Audio", "ASR/speaker/audio understanding"

    if has_any(top_kw_text, ("facial", "face", "avatar", "head", "pose")) and has_any(
        kw_text,
        ("generation", "avatar", "facial", "head", "human", "pose"),
    ):
        return "人体与头像生成：姿态、表情与数字人合成", "Human-Centric Generation", "human/avatar/facial generation"

    if has_any(top_kw_text, ("reasoning", "cot", "chain-of-thought", "question", "answering")) and has_any(
        kw_text,
        ("llm", "llms", "language", "knowledge", "reasoning"),
    ):
        if has_any(kw_text, ("reinforcement", "reward", "policy", "rl")):
            return "大模型推理：RL驱动推理与奖励学习", "LLM Reasoning", "RL for LLM reasoning"
        return "大模型推理：问答、常识与思维链", "LLM Reasoning", "QA/commonsense/chain-of-thought reasoning"

    if has_any(top_kw_text, ("bias", "social", "cultural", "gender", "misinformation", "fake news")):
        return "大模型社会安全：偏见、虚假信息与检测", "Trustworthy LLM", "bias/misinformation/detection"

    if has_any(top_kw_text, ("evaluation", "metrics", "benchmark", "legal", "clinical")) and has_any(
        kw_text,
        ("llm", "llms", "language", "human", "evaluation", "metrics"),
    ):
        return "大模型评测：人类偏好、任务指标与领域评估", "LLM Evaluation", "human preference/task/domain evaluation"

    if has_any(top_kw_text, ("lora", "adapter", "adapters", "low-rank", "lowrank", "federated")):
        if has_any(kw_text, ("federated", "clients", "client")):
            return "联邦大模型微调：LoRA、客户端异构与模型合并", "Efficient LLMs", "federated LoRA/fine-tuning"
        return "高效大模型：参数高效微调与LoRA适配", "Efficient LLMs", "PEFT/LoRA/adaptation"

    if has_any(top_kw_text, ("policy", "reinforcement", "reward", "regret", "bandit", "exploration")):
        if has_any(kw_text, ("bandit", "regret")):
            return "在线决策：Bandit、后悔界与探索", "Reinforcement Learning", "bandits/regret/exploration"
        if has_any(kw_text, ("offline", "dataset", "batch")):
            return "强化学习：离线策略、奖励建模与控制", "Reinforcement Learning", "offline RL/reward/control"
        return "强化学习：策略优化、奖励学习与控制", "Reinforcement Learning", "policy optimization/reward/control"

    if has_any(top_kw_text, ("language", "llms", "llm", "fine-tuning", "finetuning", "inference", "tokens")) and has_any(
        kw_text,
        ("llm", "llms", "language llms", "fine-tuning", "finetuning", "instruction"),
    ):
        if has_any(kw_text, ("human", "linguistic", "reading", "cognitive", "semantic")):
            return "语言模型分析：认知、语言结构与可解释性", "LLM Analysis", "cognitive/linguistic interpretability"
        if has_any(kw_text, ("tokens", "attention", "inference", "long", "cache")):
            return "高效大模型：长上下文、注意力与推理优化", "Efficient LLMs", "long-context/attention/inference optimization"
        return "大模型训练：微调、数据配方与任务适配", "LLM", "fine-tuning/data recipes/adaptation"

    if has_any(top_kw_text, ("diffusion", "generation", "generative", "score", "denoising")):
        if has_any(kw_text, ("video", "text to video", "texttovideo")):
            return "生成模型：视频扩散生成与编辑", "Generative AI", "video diffusion/generation"
        if has_any(kw_text, ("3d", "nerf", "gaussian", "splatting", "multiview", "multi view")):
            return "生成模型：三维生成、新视角与Gaussian Splatting", "3D Generation", "3D generation/view synthesis"
        if has_any(kw_text, ("image", "text to image", "texttoimage")):
            return "生成模型：文生图、扩散采样与图像编辑", "Generative AI", "text-to-image diffusion/editing"
        return "生成模型：扩散模型、采样与内容生成", "Generative AI", "diffusion/sampling/content generation"

    if has_any(top_kw_text, ("preference", "preferences", "alignment", "dpo", "rlhf", "human feedback")):
        return "大模型对齐：偏好优化、RLHF与奖励建模", "Alignment", "preference alignment/RLHF"

    if has_any(top_kw_text, ("event", "event-based", "event cameras", "cameras")):
        return "事件视觉：事件相机、运动估计与时序感知", "Event Vision", "event cameras/motion estimation"

    if has_any(top_kw_text, ("robot", "robotic", "manipulation", "navigation", "trajectory", "motion")) and not has_any(
        kw_text,
        ("event", "event-based", "event cameras", "camera"),
    ) and has_any(
        kw_text,
        ("robot", "robotic", "manipulation", "navigation", "trajectory", "driving", "pose", "motion"),
    ):
        return "具身智能：机器人操作、导航与视觉语言动作", "Embodied AI", "robot manipulation/navigation/vision-language-action"

    if has_any(top_kw_text, ("gaussian", "splatting", "radiance", "rendering", "reconstruction", "novel view")) and has_any(
        kw_text,
        ("image", "visual", "3d", "scene", "object", "view", "surface", "reconstruction", "gaussian"),
    ):
        return "三维视觉：Gaussian Splatting、新视角合成与重建", "3D Vision", "Gaussian Splatting/view synthesis/reconstruction"

    if has_any(top_kw_text, ("image", "images", "visual", "object", "detection", "features", "feature")) and has_any(
        kw_text,
        ("detection", "segmentation", "localization", "recognition", "object", "features", "reconstruction"),
    ):
        return "视觉感知：目标检测、识别与视觉表征", "Computer Vision", "detection/recognition/visual representation"

    if has_any(top_kw_text, OPTIMIZATION_TERMS) and not has_any(top_kw_text, ("brain", "neuroscience", "eeg", "spiking", "synaptic")):
        if has_any(kw_text, ("convex", "nonconvex", "stochastic", "convergence", "rates")):
            return "优化理论：随机/非凸优化与收敛率", "Optimization", "stochastic/non-convex optimization"
        return "优化理论：梯度方法、收敛性与训练动力学", "Optimization", "gradient optimization/training dynamics"

    if has_any(top_kw_text, ("brain", "neuroscience", "neural activity", "eeg", "spiking", "synaptic")):
        return "神经科学AI：脑活动建模、EEG与脉冲网络", "AI for Neuroscience", "brain activity/EEG/spiking networks"

    if has_any(kw_text, ("molecular", "molecule", "protein", "drug", "chemical", "biology", "gene")):
        if has_any(kw_text, ("protein", "folding")):
            return "AI4Science：蛋白结构、序列与功能建模", "AI for Science", "protein modeling"
        return "AI4Science：分子生成、药物发现与化学建模", "AI for Science", "molecular/drug discovery"

    if has_any(kw_text, ("radiance", "rendering", "surface", "view synthesis", "novel view")) or (
        has_any(kw_text, ("reconstruction", "shape", "scene", "pose", "cloud"))
        and has_any(kw_text, ("3d", "rendering", "radiance", "surface", "cloud", "depth", "lidar", "gaussian", "splatting"))
    ):
        return "三维视觉：Gaussian Splatting、新视角合成与重建", "3D Vision", "Gaussian Splatting/view synthesis/reconstruction"

    if has_any(kw_text, ("diffusion", "score", "denoising diffusion", "text to image", "texttoimage", "image generation")):
        if has_any(kw_text, ("video", "text to video", "texttovideo")):
            return "生成模型：视频扩散生成与编辑", "Generative AI", "video diffusion/generation"
        if has_any(kw_text, ("3d", "nerf", "gaussian", "splatting", "multiview", "multi view")):
            return "生成模型：三维生成、新视角与Gaussian Splatting", "3D Generation", "3D generation/view synthesis"
        return "生成模型：文生图、扩散采样与图像编辑", "Generative AI", "text-to-image diffusion/editing"

    if has_any(kw_text, ("molecular", "molecule", "protein", "drug", "chemical", "biology")):
        if has_any(kw_text, ("protein", "folding")):
            return "AI4Science：蛋白结构、序列与功能建模", "AI for Science", "protein modeling"
        return "AI4Science：分子生成、药物发现与化学建模", "AI for Science", "molecular/drug discovery"

    if has_any(kw_text, ("causal", "treatment", "counterfactual", "causal discovery", "confounding", "interventions")):
        return "因果学习：因果发现、反事实与处理效应", "Causal ML", "causal discovery/counterfactuals"

    llm_signal_terms = (
        "llm",
        "llms",
        "mllm",
        "mllms",
        "vlm",
        "vlms",
        "large language model",
        "large language models",
        "large multimodal model",
        "large multimodal models",
        "vision language model",
        "vision language models",
        "vision-language model",
        "vision-language models",
        "vision language modeling",
        "vision-language modeling",
        "chatgpt",
        "gpt",
        "foundation model",
        "foundation models",
        "slm",
        "slms",
        "small language model",
        "small language models",
    )
    has_llm_signal = has_any(text, llm_signal_terms) or has_any(
        kw_text,
        (
            "visionlanguage",
            "multimodal language",
        ),
    ) or (
        has_any(kw_text, ("language model", "language models"))
        and has_any(text, ("reasoning", "instruction", "alignment", "agent", "prompt", "cot", "chain of thought", "rlhf", "dpo", "tool"))
    )
    if has_llm_signal:
        if has_any(kw_text, ("cache", "quantization", "compression", "memory", "speculative", "latency", "decoding", "longcontext", "long context")):
            if has_any(kw_text, ("cache", "kv cache", "decoding", "speculative", "latency")):
                return "高效大模型：解码加速、KV缓存与推理优化", "Efficient LLMs", "decoding/KV-cache/inference optimization"
            return "高效大模型：推理加速、压缩与资源优化", "Efficient LLMs", "inference acceleration/compression"
        if has_any(kw_text, ("lora", "adapter", "adapters", "parameter efficient", "peft", "lowrank", "low rank")):
            return "高效大模型：参数高效微调与LoRA适配", "Efficient LLMs", "PEFT/LoRA/adaptation"
        if has_any(kw_text, ("recommendation", "recommender", "user", "users", "item", "items", "personalization")):
            return "大模型推荐：LLM增强推荐、用户建模与个性化", "LLM / Recommendation", "LLM-enhanced recommendation/personalization"
        if has_any(kw_text, ("speech", "audio", "asr", "tts", "spoken", "voice", "speaker")):
            return "语音大模型：音频理解、ASR与语音生成", "Speech LLM", "audio/speech language models"
        if has_any(kw_text, ("medical", "clinical", "patient", "healthcare", "diagnosis", "radiology", "pathology")):
            return "医疗大模型：临床推理、医学影像与健康问答", "Medical LLM", "clinical/medical large models"
        if has_any(text, ("jailbreak", "harmful", "unsafe", "red teaming", "prompt injection", "attack", "attacks", "adversarial", "backdoor", "privacy", "unlearning", "watermark", "security", "safety", "defense")):
            return "大模型安全：越狱、攻击防护与隐私", "Trustworthy AI", "LLM safety/adversarial/privacy"
        if has_any(text, ("rag", "retrieval augmented", "retrievalaugmented", "retriever", "retrievers")):
            return "检索增强大模型：RAG、知识注入与问答", "LLM / IR", "RAG/knowledge-grounded QA"
        if has_any(kw_text, ("reinforcement", "policy", "policies", "reward", "offline", "exploration")):
            return "大模型强化学习：策略迭代、奖励学习与控制", "LLM / RL", "LLM-based reinforcement learning"
        if has_any(kw_text, ("agent", "agents", "multiagent", "tool", "workflow", "planning", "gui", "web")):
            return "LLM智能体：工具调用、规划与工作流", "LLM Agents", "tool/planning agents"
        if has_any(kw_text, ("fake news", "misinformation", "deepfake", "forgery", "bias", "fairness", "hate", "toxic", "toxicity", "stance", "social bias")):
            return "大模型社会安全：偏见、虚假信息与检测", "Trustworthy LLM", "bias/misinformation/detection"
        if has_any(text, ("reasoning", "chain of thought", "cot", "math", "mathematical", "question", "answering", "commonsense", "multi hop", "multihop", "logic")):
            if has_any(text, ("visual", "vision", "multimodal", "image", "video", "vlm", "mllm")):
                return "大模型推理：多模态/视觉推理", "LLM Reasoning", "multimodal reasoning"
            if has_any(text, ("reinforcement", "reward", "policy", "rl")):
                return "大模型推理：RL驱动推理与奖励学习", "LLM Reasoning", "RL for LLM reasoning"
            return "大模型推理：问答、常识与思维链", "LLM Reasoning", "QA/commonsense/chain-of-thought reasoning"
        if has_any(kw_text, ("graph", "graphs", "gnn", "gnns", "node", "nodes", "knowledge graph")):
            return "图基础模型：LLM增强图学习与节点表示", "Graph Foundation Models", "LLM-enhanced graph learning"
        if has_any(kw_text, ("time series", "timeseries", "forecasting", "forecast")):
            return "时序基础模型：时间序列预测与跨域泛化", "Foundation Models", "time-series foundation models"
        if has_any(kw_text, ("mamba", "vits", "vision transformers", "vision transformer", "linear attention")):
            return "视觉基础模型：Mamba/ViT与高效视觉语言建模", "Vision Foundation Models", "efficient vision-language modeling"
        if has_any(text, ("preference", "preferences", "rlhf", "dpo", "reward", "human feedback")):
            return "大模型对齐：偏好优化、RLHF与奖励建模", "Alignment", "preference alignment/RLHF"
        if has_any(text, ("safety", "harmful", "jailbreak", "guardrail", "unsafe", "red teaming")):
            return "大模型安全：越狱、防护与安全对齐", "Trustworthy AI", "LLM safety/jailbreak"
        if has_any(text, ("hallucination", "factual", "truthfulness", "fact")):
            return "大模型可信：幻觉检测与事实性", "Trustworthy AI", "hallucination/factuality"
        if has_any(text, ("embedding", "embeddings", "classification", "text embedding", "autoregressive")):
            return "大模型表征：Embedding、分类与文本表征", "LLM", "LLM embeddings/text representations"
        if has_any(text, ("visual", "vision", "multimodal", "image", "video", "vlm", "mllm")):
            if has_any(text, ("video", "temporal", "action", "motion")):
                return "多模态大模型：视频理解与时序推理", "Multimodal LLM", "video-language large models"
            return "多模态大模型：视觉语言理解与跨模态推理", "Multimodal LLM", "multimodal/VLM understanding"
        if has_any(text, ("multilingual", "translation", "crosslingual", "low resource", "languages")):
            return "多语言大模型：翻译、跨语言与低资源", "NLP", "multilingual LLMs"
        if has_any(text, ("code", "program", "software", "repository")):
            return "代码大模型：代码生成、程序理解与软件工程", "Code Intelligence", "code LLMs"
        if has_any(kw_text, ("culture", "cultural", "moral", "personality", "persona", "empathy", "mental health", "psychological", "counseling")):
            return "大模型人文社会：价值观、文化与心理行为评测", "LLM / Society", "values/culture/psychology evaluation"
        if has_any(text, ("instruction", "prompt", "benchmark", "evaluation", "finetuning", "fine tuning", "adaptation")):
            return "大模型应用：指令微调、评测与任务适配", "LLM", "instruction tuning/evaluation/adaptation"
        return "大模型应用：训练、评测与任务适配", "LLM", "LLM training/evaluation/adaptation"

    if has_any(kw_text, ("sat", "combinatorial", "solver", "solvers", "integer programming", "satisfiability", "boolean", "constraint")):
        return "组合优化：SAT求解、整数规划与搜索", "Optimization", "SAT/combinatorial optimization"

    if has_any(kw_text, ("recommendation", "recommender", "collaborative filtering", "ctr", "click through", "user item", "users", "items")):
        if has_any(kw_text, ("sequential", "session", "sequence")):
            return "推荐系统：序列/会话推荐与用户行为建模", "Recommendation", "sequential/session recommendation"
        if has_any(kw_text, ("multimodal", "image", "video", "content")):
            return "推荐系统：多模态内容推荐", "Recommendation", "multimodal recommendation"
        if has_any(kw_text, ("graph", "knowledge graph", "gnn", "heterogeneous")):
            return "推荐系统：图推荐与知识增强推荐", "Recommendation", "graph/knowledge-enhanced recommendation"
        if has_any(kw_text, ("fair", "debias", "bias", "privacy")):
            return "推荐系统：公平性、偏差与隐私", "Recommendation", "fair/private recommendation"
        return "推荐系统：排序、召回与点击率预测", "Recommendation", "ranking/retrieval/CTR recommendation"

    if has_any(kw_text, ("graph", "graphs", "gnn", "gnns", "node", "nodes", "link prediction", "subgraph", "graph clustering", "graph representation")):
        if has_any(kw_text, ("knowledge graph", "entity", "relation")):
            return "知识图谱：实体关系、推理与补全", "Knowledge Graph", "knowledge graph reasoning/completion"
        if has_any(kw_text, ("clustering", "community", "multiplex", "attributed", "multiview")):
            return "图学习：图聚类、表示学习与结构匹配", "Graph Learning", "graph clustering/representation/matching"
        if has_any(kw_text, ("pretraining", "pre training", "prompt", "condensation", "topology")):
            return "图学习：图预训练、提示与结构压缩", "Graph Learning", "graph pretraining/prompting/condensation"
        return "图学习：GNN、节点分类与链接预测", "Graph Learning", "GNN/node/link prediction"

    if has_any(kw_text, BANDIT_TERMS):
        return "在线决策：Bandit、后悔界与探索", "Reinforcement Learning", "bandits/regret/exploration"

    if has_any(text, RL_TERMS):
        if has_any(text, ("offline", "batch", "dataset", "datasets")):
            return "强化学习：离线策略、奖励建模与控制", "Reinforcement Learning", "offline RL/reward/control"
        if has_any(text, ("multi agent", "multiagent", "game", "games", "agent", "agents")):
            return "多智能体强化学习：博弈、协作与策略学习", "Reinforcement Learning", "multi-agent RL/games/policies"
        return "强化学习：策略优化、奖励学习与控制", "Reinforcement Learning", "policy optimization/reward/control"

    if has_any(kw_text, ("kernel", "ntk", "tangent kernel", "tangent", "relu", "generalization bound")):
        return "理论机器学习：核方法、NTK与泛化分析", "ML Theory", "kernel methods/NTK/generalization"

    if has_any(kw_text, ("federated", "clients", "client", "server", "aggregation")):
        return "联邦学习：异构客户端、隐私与分布式优化", "Federated Learning", "federated optimization/privacy"

    if has_any(kw_text, ("adversarial", "attack", "attacks", "backdoor", "watermark", "privacy")):
        return "可信安全：对抗攻击、后门、水印与隐私", "Security / Privacy", "adversarial/backdoor/watermark/privacy"

    if has_any(kw_text, ("pde", "pdes", "differential equation", "differential", "operator", "physics", "fluid")):
        return "科学计算：神经算子、PDE与物理建模", "AI for Science", "neural operators/PDEs"

    if has_any(kw_text, OPTIMIZATION_TERMS):
        if has_any(kw_text, ("convex", "nonconvex", "stochastic", "convergence", "rates")):
            return "优化理论：随机/非凸优化与收敛率", "Optimization", "stochastic/non-convex optimization"
        return "优化理论：梯度方法、收敛性与训练动力学", "Optimization", "gradient optimization/training dynamics"

    if has_any(kw_text, ("ood", "outofdistribution", "out of distribution", "label", "labels", "noisy", "semisupervised", "semi supervised")):
        return "鲁棒泛化：OOD检测、校准与噪声标签", "Robust / Weakly Supervised ML", "OOD/calibration/noisy labels"

    if has_any(text, BAYESIAN_TERMS):
        if has_any(text, ("gaussian process", "gp")):
            return "概率机器学习：高斯过程、不确定性与贝叶斯推断", "Probabilistic ML", "Gaussian processes/Bayesian inference"
        return "概率机器学习：变分推断、不确定性与后验建模", "Probabilistic ML", "variational inference/uncertainty/posteriors"

    if has_any(kw_text, ("ood", "out of distribution", "calibration", "calibrated", "label", "labels", "noisy", "semisupervised", "semi supervised")):
        return "鲁棒泛化：OOD检测、校准与噪声标签", "Robust / Weakly Supervised ML", "OOD/calibration/noisy labels"

    if has_any(kw_text, ("bound", "bounds", "risk", "sample complexity", "statistical", "decision tree", "trees")):
        if has_any(kw_text, ("decision tree", "trees")):
            return "统计学习理论：决策树、风险界与泛化分析", "ML Theory", "decision trees/risk bounds"
        return "统计学习理论：泛化界、风险分析与样本复杂度", "ML Theory", "generalization/risk/sample complexity"

    if has_any(text, ("vision transformer", "vision transformers", "vit")) or (
        has_any(text, ("transformer", "transformers", "selfattention", "self attention"))
        and has_any(text, ("vision", "image", "images", "visual"))
    ):
        return "视觉Transformer：ViT、自注意力与视觉表征", "Computer Vision", "vision transformers/self-attention"

    if has_any(kw_text, ("point cloud", "lidar", "3d detection", "depth", "camera")):
        if has_any(text, ("autonomous", "driving", "trajectory", "lidar")):
            return "自动驾驶感知：LiDAR、轨迹与三维检测", "Autonomous Driving", "LiDAR/trajectory/3D detection"
        return "三维视觉：点云、深度估计与相机姿态", "3D Vision", "point cloud/depth/pose"

    if has_any(kw_text, ("restoration", "image restoration", "superresolution", "super resolution", "denoising", "deblurring", "lowlight", "low light")):
        return "底层视觉：超分、去噪与图像复原", "Computer Vision", "image restoration/super-resolution"

    if has_any(kw_text, ("segmentation", "semantic segmentation", "instance segmentation", "panoptic", "object detection", "detector", "detection", "mask", "rcnn", "open vocabulary")):
        if has_any(kw_text, ("semantic segmentation", "instance segmentation", "panoptic")):
            return "视觉感知：语义/实例/全景分割", "Computer Vision", "segmentation"
        if has_any(kw_text, ("open vocabulary", "openvocabulary")):
            return "视觉感知：开放词汇检测与分割", "Computer Vision", "open-vocabulary detection/segmentation"
        return "视觉感知：目标检测、定位与分割", "Computer Vision", "detection/localization/segmentation"

    if has_any(kw_text, ("diffusion", "score", "denoising diffusion", "text to image", "texttoimage", "image generation")):
        if has_any(kw_text, ("video", "text to video", "texttovideo")):
            return "生成模型：视频扩散生成与编辑", "Generative AI", "video diffusion/generation"
        if has_any(kw_text, ("3d", "nerf", "gaussian", "splatting", "multiview", "multi view")):
            return "生成模型：三维生成、新视角与Gaussian Splatting", "3D Generation", "3D generation/view synthesis"
        return "生成模型：文生图、扩散采样与图像编辑", "Generative AI", "text-to-image diffusion/editing"

    if has_any(kw_text, ("video", "videos", "video understanding", "long video", "action recognition", "temporal")):
        return "视频理解：长视频、多帧时序与事件理解", "Video Understanding", "long-video/temporal understanding"

    if has_any(text, ("shape", "scene", "object", "objects", "reconstruction")) and has_any(text, ("reconstruction", "3d", "shape", "scene")):
        return "三维视觉：形状、场景与物体重建", "3D Vision", "shape/scene/object reconstruction"

    if has_any(kw_text, ("architecture search", "neural architecture search", "nas", "hyperparameter", "automl")):
        return "自动机器学习：神经架构搜索与超参数优化", "AutoML", "neural architecture search/hyperparameter optimization"

    if has_any(text, ("kernel", "ntk", "tangent kernel", "gaussian process", "gp", "relu", "generalization bound")):
        return "理论机器学习：核方法、NTK与泛化分析", "ML Theory", "kernel methods/NTK/generalization"

    if has_any(kw_text, ("brain", "neuroscience", "neural activity", "eeg", "spiking", "synaptic", "neural response")):
        return "神经科学AI：脑活动建模、EEG与脉冲网络", "AI for Neuroscience", "brain activity/EEG/spiking networks"

    if has_any(kw_text, ("person reidentification", "person re identification", "reid", "vehicle reidentification")) or (
        has_any(kw_text, ("tracking",)) and has_any(kw_text, ("person", "vehicle", "pedestrian"))
    ):
        return "行人车辆重识别：ReID、跟踪与域适应", "Computer Vision", "person/vehicle re-identification"

    if has_any(kw_text, ("factchecking", "fact checking", "claims", "factuality", "evidence", "verification")):
        return "可信信息：事实核查、证据检索与声明验证", "Trustworthy AI / IR", "fact checking/evidence verification"
    if has_any(kw_text, ("knowledge graph", "knowledge graphs", "kgs")):
        return "知识图谱：实体关系、推理与补全", "Knowledge Graph", "knowledge graph reasoning/completion"
    if has_any(kw_text, ("relation extraction", "entity", "entities", "relation", "relations")):
        return "知识抽取：实体识别、关系抽取与事件理解", "Information Extraction", "entity/relation extraction"

    if has_any(text, MODEL_EFFICIENCY_TERMS) or (
        has_any(text, ("inference", "attention"))
        and has_any(text, ("llm", "llms", "language model", "transformer", "longcontext", "long context", "kv cache", "decoding", "tokens"))
    ):
        if has_any(text, ("lora", "adapter", "adapters", "finetuning", "fine tuning", "parameter efficient", "peft")):
            return "高效大模型：参数高效微调与LoRA适配", "Efficient LLMs", "PEFT/LoRA/adaptation"
        if has_any(text, ("longcontext", "long context", "context window")):
            return "高效大模型：长上下文推理与注意力压缩", "Efficient LLMs", "long-context inference/attention compression"
        if has_any(text, ("speculative", "decoding", "kv cache", "cache", "tokens", "token")):
            return "高效大模型：解码加速、KV缓存与推理优化", "Efficient LLMs", "decoding/KV-cache/inference optimization"
        if has_any(text, ("quantization", "quantized", "pruning", "lowrank", "low rank")):
            return "模型压缩：量化、剪枝与低秩加速", "Efficient ML", "quantization/pruning/low-rank compression"
        return "高效大模型：推理加速、压缩与资源优化", "Efficient LLMs", "inference acceleration/compression"

    if has_any(text, ("incontext", "in context", "icl", "demonstrations", "few shot prompting", "few-shot prompting")):
        return "上下文学习：ICL、示例选择与记忆机制", "In-context Learning", "ICL/demonstrations/memory"

    if has_any(text, MULTILINGUAL_TERMS):
        if has_any(text, ("speech", "asr", "sign language")):
            return "多语言语音：语音识别、翻译与手语理解", "Speech / NLP", "multilingual speech/sign-language"
        return "多语言NLP：机器翻译、跨语言与低资源", "NLP", "machine translation/cross-lingual/low-resource"

    if has_any(text, ("dialogue", "dialog", "conversation", "conversational", "response", "responses", "task oriented")):
        return "对话系统：响应生成、情感支持与任务型对话", "Dialogue", "dialogue/response generation/task-oriented systems"

    if has_any(text, ("summarization", "summary", "summaries", "scientific", "factual", "generation", "controllable generation")):
        return "文本生成：摘要、事实性与可控生成", "Text Generation", "summarization/factuality/controllable generation"

    if has_any(text, ("question answering", "answering", "reading comprehension", "commonsense", "multi hop", "multihop", "cot", "reasoning")):
        return "语言推理：问答、常识与多跳推理", "NLP Reasoning", "QA/commonsense/multi-hop reasoning"

    if has_any(text, ("auction", "allocation", "welfare", "mechanism", "market", "voting")):
        return "多智能体：机制设计、拍卖与资源分配", "Agents / Game Theory", "mechanism design/auctions"

    if has_any(text, ("bias", "fairness", "stereotype", "stereotypes", "hate", "toxic", "toxicity", "stance", "social media", "political")):
        return "社会计算与安全NLP：偏见、仇恨与立场检测", "Trustworthy NLP", "bias/hate/stance/social NLP"

    if has_any(text, ("parsing", "semantic parsing", "syntactic", "syntax", "morphological", "inflection", "coreference", "discourse")):
        return "语言结构：句法、语义解析与篇章建模", "NLP Structure", "syntax/semantic parsing/discourse"

    if has_any(text, ("speech", "asr", "audio", "spoken", "voice")):
        return "语音语言：ASR、语音理解与音频建模", "Speech / Audio", "ASR/spoken language/audio"

    if has_any(text, ("code", "program", "programming", "software", "repository", "bug")):
        return "代码智能：代码生成、程序理解与软件工程", "Code Intelligence", "code generation/program analysis/software engineering"

    # Put recommendation before graph: SIGIR/KDD topics often mention graph but
    # are fundamentally recsys topics.
    if has_any(text, ("recommendation", "recommender", "collaborative filtering", "ctr", "click through", "user item")):
        if has_any(text, ("sequential", "session", "sequence")):
            return "推荐系统：序列/会话推荐与用户行为建模", "Recommendation", "sequential/session recommendation"
        if has_any(text, ("multimodal", "image", "video", "content")):
            return "推荐系统：多模态内容推荐", "Recommendation", "multimodal recommendation"
        if has_any(text, ("graph", "knowledge graph", "gnn", "heterogeneous")):
            return "推荐系统：图推荐与知识增强推荐", "Recommendation", "graph/knowledge-enhanced recommendation"
        if has_any(text, ("fair", "debias", "bias", "privacy")):
            return "推荐系统：公平性、偏差与隐私", "Recommendation", "fair/private recommendation"
        return "推荐系统：排序、召回与点击率预测", "Recommendation", "ranking/retrieval/CTR recommendation"

    if has_any(kw_text, ("video retrieval", "moment retrieval", "image retrieval", "cross modal retrieval", "cir")):
        if has_any(kw_text, ("video retrieval", "moment retrieval")):
            return "多媒体检索：视频检索与时刻定位", "Multimedia Retrieval", "video/moment retrieval"
        return "多媒体检索：图像检索与跨模态检索", "Multimedia Retrieval", "image/cross-modal retrieval"
    if has_any(kw_text, ("factchecking", "fact checking", "claims", "factuality", "evidence", "verification")):
        return "可信信息：事实核查、证据检索与声明验证", "Trustworthy AI / IR", "fact checking/evidence verification"
    if has_any(kw_text, ("knowledge graph", "knowledge graphs", "kgs")):
        return "知识图谱：实体关系、推理与补全", "Knowledge Graph", "knowledge graph reasoning/completion"
    if has_any(kw_text, ("relation extraction", "entity", "entities", "relation", "relations")):
        return "知识抽取：实体识别、关系抽取与事件理解", "Information Extraction", "entity/relation extraction"

    if has_any(text, ("graph", "graphs", "gnn", "gnns", "node", "nodes", "link prediction", "subgraph", "graph clustering", "graph representation")):
        if has_any(text, ("knowledge graph", "entity", "relation")):
            return "知识图谱：实体关系、推理与补全", "Knowledge Graph", "knowledge graph reasoning/completion"
        if has_any(text, ("clustering", "community", "multiplex", "attributed")):
            return "图学习：图聚类、表示学习与结构匹配", "Graph Learning", "graph clustering/representation/matching"
        if has_any(text, ("pretraining", "pre training", "prompt", "condensation", "topology")):
            return "图学习：图预训练、提示与结构压缩", "Graph Learning", "graph pretraining/prompting/condensation"
        return "图学习：GNN、节点分类与链接预测", "Graph Learning", "GNN/node/link prediction"

    if has_any(text, ("retrieval augmented", "rag", "retrievalaugmented")):
        return "检索增强生成：RAG、知识注入与问答", "LLM / IR", "retrieval-augmented generation"
    if has_any(text, ("dense retrieval", "neural retrieval", "vector retrieval", "retriever", "reranking", "re ranking")):
        if has_any(text, ("rerank", "reranking", "ranking")):
            return "信息检索：检索排序与重排", "Information Retrieval", "retrieval ranking/reranking"
        return "信息检索：密集检索与向量召回", "Information Retrieval", "dense/vector retrieval"
    if has_any(text, ("query", "search", "web search", "document retrieval", "information retrieval")):
        return "信息检索：查询理解、搜索与文档检索", "Information Retrieval", "query/search/document retrieval"

    if has_any(kw_text, ("sat", "combinatorial", "solver", "solvers", "integer programming", "satisfiability")):
        return "组合优化：SAT求解、整数规划与搜索", "Optimization", "SAT/combinatorial optimization"
    if has_any(kw_text, ("anomaly", "anomaly detection", "outlier", "fraud")):
        return "异常检测：图异常、欺诈检测与时序异常", "Anomaly Detection", "graph/fraud/time-series anomaly detection"
    if has_any(kw_text, ("medical", "clinical", "patient", "healthcare", "disease", "biomedical")):
        if has_any(kw_text, ("segmentation", "image", "imaging", "tumor")):
            return "医疗AI：医学影像分割、肿瘤与临床影像", "AI for Health", "medical image segmentation"
        return "医疗AI：临床预测、医学影像与生物医学NLP", "AI for Health", "clinical/medical AI"
    if has_any(kw_text, ("label", "labels", "multilabel", "multi label", "noisy", "semisupervised", "semi supervised")):
        return "弱监督学习：噪声标签、半监督与多标签分类", "Weak/Semi-supervised Learning", "noisy/semi-supervised/multi-label learning"

    if has_any(kw_text, ("point cloud", "lidar", "3d detection", "depth", "camera")):
        if has_any(text, ("autonomous", "driving", "trajectory", "lidar")):
            return "自动驾驶感知：LiDAR、轨迹与三维检测", "Autonomous Driving", "LiDAR/trajectory/3D detection"
        return "三维视觉：点云、深度估计与相机姿态", "3D Vision", "point cloud/depth/pose"
    if has_any(kw_text, ("restoration", "image restoration", "superresolution", "super resolution", "denoising", "deblurring", "lowlight", "low light")):
        return "底层视觉：超分、去噪与图像复原", "Computer Vision", "image restoration/super-resolution"
    if has_any(kw_text, ("human motion", "pose", "hand", "skeleton", "hoi", "action recognition", "motion")):
        if has_any(text, ("action", "video", "temporal")):
            return "视频理解：动作识别、时序建模与行为分析", "Video Understanding", "action/temporal video understanding"
        return "人体理解：姿态、手部与人体运动建模", "Human-Centric Vision", "pose/hand/human motion"
    if has_any(kw_text, ("video", "videos", "video understanding", "long video")):
        return "视频理解：长视频、多帧时序与事件理解", "Video Understanding", "long-video/temporal understanding"

    if has_any(kw_text, ("llm", "llms", "large language model", "language models")):
        if has_any(kw_text, ("agent", "agents", "multiagent")) or has_any(text, ("tool", "workflow", "planning")):
            return "LLM智能体：工具调用、规划与工作流", "LLM Agents", "tool/planning agents"
        if has_any(text, ("reasoning", "chain of thought", "cot", "math", "mathematical")):
            if has_any(text, ("visual", "vision", "multimodal", "image", "video")):
                return "大模型推理：多模态/视觉推理", "LLM Reasoning", "multimodal reasoning"
            if has_any(text, ("reinforcement", "reward", "policy", "rl")):
                return "大模型推理：RL驱动推理与奖励学习", "LLM Reasoning", "RL for LLM reasoning"
            return "大模型推理：数学、逻辑与思维链", "LLM Reasoning", "math/logical chain-of-thought reasoning"
        if has_any(text, ("alignment", "preference", "rlhf", "dpo", "reward")):
            return "大模型对齐：偏好优化、RLHF与奖励建模", "Alignment", "preference alignment/RLHF"
        if has_any(text, ("safety", "harmful", "jailbreak", "guardrail", "unsafe", "red teaming")):
            return "大模型安全：越狱、防护与安全对齐", "Trustworthy AI", "LLM safety/jailbreak"
        if has_any(text, ("hallucination", "factual", "truthfulness", "fact")):
            return "大模型可信：幻觉检测与事实性", "Trustworthy AI", "hallucination/factuality"
        if has_any(text, ("multilingual", "translation", "crosslingual", "low resource", "languages")):
            return "多语言大模型：翻译、跨语言与低资源", "NLP", "multilingual LLMs"
        if has_any(text, ("code", "program", "software", "repository")):
            return "代码大模型：代码生成、程序理解与软件工程", "Code Intelligence", "code LLMs"
        return "大模型应用：训练、评测与任务适配", "LLM", "LLM training/evaluation/adaptation"

    if has_any(text, ("agent", "agents", "multiagent", "multi agent")):
        if has_any(text, ("auction", "allocation", "welfare", "mechanism", "market")):
            return "多智能体：机制设计、拍卖与资源分配", "Agents / Game Theory", "mechanism design/auctions"
        if has_any(text, ("navigation", "embodied", "robot", "environment", "planning")):
            return "具身智能体：导航、规划与交互", "Embodied AI", "embodied navigation/planning"
        return "智能体系统：协作、规划与多智能体学习", "Agents", "agent collaboration/planning"

    if has_any(text, ("multimodal", "vision language", "visionlanguage", "vlm", "vqa", "mllm")):
        if has_any(text, ("video", "temporal", "action", "motion")):
            return "多模态理解：视频语言、动作与时序推理", "Multimodal AI", "video-language understanding"
        if has_any(text, ("grounding", "referring", "localization", "phrase")):
            return "多模态理解：视觉定位与指代表达", "Multimodal AI", "visual grounding"
        if has_any(text, ("vqa", "question answering", "question")):
            return "多模态理解：视觉问答与图文推理", "Multimodal AI", "visual question answering"
        return "多模态理解：视觉语言表征与跨模态对齐", "Multimodal AI", "vision-language representation/alignment"

    if has_any(kw_text, ("molecular", "molecule", "protein", "drug", "chemical", "biology")):
        if has_any(kw_text, ("protein", "folding")):
            return "AI4Science：蛋白结构、序列与功能建模", "AI for Science", "protein modeling"
        return "AI4Science：分子生成、药物发现与化学建模", "AI for Science", "molecular/drug discovery"

    if has_any(text, ("diffusion", "score", "denoising diffusion", "text to image", "texttoimage", "image generation")):
        if has_any(text, ("video", "text to video", "texttovideo")):
            return "生成模型：视频扩散生成与编辑", "Generative AI", "video diffusion/generation"
        if has_any(text, ("3d", "nerf", "gaussian", "splatting", "multiview", "multi view")):
            return "生成模型：三维生成、新视角与Gaussian Splatting", "3D Generation", "3D generation/view synthesis"
        return "生成模型：文生图、扩散采样与图像编辑", "Generative AI", "text-to-image diffusion/editing"
    if has_any(text, ("gaussian", "splatting", "nerf", "novel view", "view synthesis", "rendering", "reconstruction")):
        return "三维视觉：Gaussian Splatting、新视角合成与重建", "3D Vision", "Gaussian Splatting/view synthesis/reconstruction"
    if has_any(text, ("gan", "image to image", "imagetoimage", "inpainting", "super resolution", "superresolution")):
        if has_any(text, ("super resolution", "superresolution", "restoration", "denoising", "deblurring")):
            return "底层视觉：超分、去噪与图像复原", "Computer Vision", "image restoration/super-resolution"
        return "生成模型：GAN、图像翻译与修复", "Generative AI", "GAN/image translation/inpainting"

    if has_any(text, ("segmentation", "object detection", "detector", "mask", "rcnn", "open vocabulary")):
        if has_any(text, ("semantic segmentation", "instance segmentation", "panoptic")):
            return "视觉感知：语义/实例/全景分割", "Computer Vision", "segmentation"
        if has_any(text, ("open vocabulary", "openvocabulary")):
            return "视觉感知：开放词汇检测与分割", "Computer Vision", "open-vocabulary detection/segmentation"
        return "视觉感知：目标检测、定位与分割", "Computer Vision", "detection/localization/segmentation"
    if has_any(text, ("face", "facial", "deepfake", "expression")):
        if has_any(text, ("attack", "adversarial", "deepfake", "spoof")):
            return "人脸与安全：识别、Deepfake与攻击防御", "Security / Vision", "face recognition/deepfake/security"
        return "人脸视觉：表情、身份与人脸生成", "Computer Vision", "face/expression/identity"

    if has_any(text, ("graph", "graphs", "gnn", "gnns", "node", "link prediction", "subgraph")):
        if has_any(text, ("condensation", "topology", "prompt", "pre training", "pretraining")):
            return "图学习：图预训练、提示与结构压缩", "Graph Learning", "graph pretraining/prompting/condensation"
        if has_any(text, ("knowledge graph", "entity", "relation")):
            return "知识图谱：实体关系、推理与补全", "Knowledge Graph", "knowledge graph reasoning/completion"
        return "图学习：GNN、节点分类与链接预测", "Graph Learning", "GNN/node/link prediction"

    if has_any(text, ("federated", "clients", "server", "distributed", "heterogeneity")):
        return "联邦学习：异构客户端、隐私与分布式优化", "Federated Learning", "federated optimization/privacy"
    if has_any(text, ("adversarial", "attack", "backdoor", "watermark", "privacy")):
        return "可信安全：对抗攻击、后门、水印与隐私", "Security / Privacy", "adversarial/backdoor/watermark/privacy"
    if has_any(text, ("continual", "catastrophic forgetting", "class incremental", "incremental")):
        return "持续学习：增量分类与灾难性遗忘", "Continual Learning", "continual/incremental learning"
    if has_any(text, ("few shot", "fewshot", "meta learning", "metalearning", "in context")):
        return "小样本学习：元学习、类增量与表征迁移", "Few-shot Learning", "few-shot/meta-learning"

    if has_any(text, ("medical", "clinical", "patient", "healthcare", "disease", "biomedical")):
        return "医疗AI：临床预测、医学影像与生物医学NLP", "AI for Health", "clinical/medical AI"
    if has_any(text, ("pde", "differential equation", "operator", "physics", "fluid")):
        return "科学计算：神经算子、PDE与物理建模", "AI for Science", "neural operators/PDEs"

    if has_any(text, ("causal", "treatment", "counterfactual", "causal discovery")):
        return "因果学习：因果发现、反事实与处理效应", "Causal ML", "causal discovery/counterfactuals"
    if has_any(kw_text, ("trajectory", "trajectories", "road", "traffic", "spatiotemporal", "spatio temporal")):
        return "时空数据挖掘：轨迹预测、交通建模与道路网络", "Spatiotemporal ML", "trajectory/traffic/spatiotemporal modeling"
    if has_any(kw_text, ("news", "fake news", "misinformation", "rumor", "rumour")):
        return "可信信息：虚假新闻、谣言与错误信息检测", "Trustworthy AI / IR", "fake news/misinformation detection"
    if has_any(text, ("time series", "timeseries", "forecasting", "forecast")):
        return "时间序列：预测、异常检测与时序表征", "Time Series", "forecasting/anomaly/time-series representation"
    if has_any(kw_text, ("preference", "rlhf", "dpo", "preferences")):
        return "大模型对齐：偏好优化、RLHF与奖励建模", "Alignment", "preference alignment/RLHF"
    if has_any(kw_text, ("manipulation", "robotic", "robot", "world model", "world models", "embodied")):
        return "机器人学习：操作、世界模型与具身控制", "Robotics", "robot manipulation/world models"
    if has_any(kw_text, ("reinforcement", "policy", "offline", "reward", "mdp", "bandit")):
        if has_any(kw_text, ("bandit", "regret")):
            return "在线决策：Bandit、后悔界与探索", "Reinforcement Learning", "bandits/regret/exploration"
        return "强化学习：离线策略、奖励建模与控制", "Reinforcement Learning", "offline RL/reward/control"
    if has_any(text, ("optimization", "gradient", "convex", "sgd", "adam")):
        return "优化理论：梯度方法、收敛性与训练动力学", "Optimization", "gradient optimization/training dynamics"
    if has_any(kw_text, ("auction", "auctions", "welfare", "allocation", "revenue", "pricing")):
        return "机制设计：拍卖、定价与资源分配", "Agents / Economics", "auction/pricing/allocation"

    if has_any(text, ("hci", "participants", "user study", "interaction", "interface", "virtual reality", "vr")):
        if has_any(text, ("virtual reality", "vr", "augmented reality", "ar")):
            return "HCI：VR/AR交互与沉浸式体验", "HCI", "VR/AR interaction"
        return "HCI：用户研究、界面交互与协作系统", "HCI", "user studies/interfaces/collaboration"

    phrase = pick_phrase(text, keywords)
    return f"细分主题：{phrase}", "Other", phrase


def representative_titles_for_year(year_dir: Path, year: int, model_safe: str, top_n: int) -> pd.DataFrame:
    papers_path = year_dir / "papers_with_topics.csv"
    embeddings_path = year_dir / f"embeddings_{model_safe}.npy"
    papers = pd.read_csv(papers_path)
    embeddings = np.load(embeddings_path)
    if len(papers) != len(embeddings):
        raise ValueError(f"{year}: papers/embeddings length mismatch: {len(papers)} vs {len(embeddings)}")

    rows = []
    topics = sorted(topic for topic in papers["topic"].dropna().astype(int).unique() if topic >= 0)
    for topic_id in topics:
        idx = papers.index[papers["topic"].astype(int) == topic_id].to_numpy()
        topic_embeddings = embeddings[idx]
        centroid = topic_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        scores = topic_embeddings @ centroid
        order = np.argsort(scores)[::-1][:top_n]
        selected = papers.iloc[idx[order]]
        rows.append(
            {
                "year": int(year),
                "topic": int(topic_id),
                "year_topic_id": f"{int(year)}_T{int(topic_id):03d}",
                "representative_titles": " || ".join(str(title) for title in selected["title"].fillna("").tolist()),
                "representative_venues": " || ".join(str(venue) for venue in selected["venue"].fillna("").tolist()),
                "title_terms": title_terms(selected["title"].fillna("").tolist()),
            }
        )
    return pd.DataFrame(rows)


def refine_results(results_dir: Path, model_safe: str, top_n_titles: int, top_k_per_venue_year: int) -> None:
    topic_summary = pd.read_csv(results_dir / "topic_summary_by_year.csv")
    topic_by_venue = pd.read_csv(results_dir / "topic_by_venue_yearly.csv")

    rep_frames = []
    for year_dir in sorted(results_dir.glob("year=*")):
        year = int(year_dir.name.split("=", 1)[1])
        rep_frames.append(representative_titles_for_year(year_dir, year, model_safe, top_n_titles))
    representatives = pd.concat(rep_frames, ignore_index=True)

    refined = topic_summary.merge(representatives, on=["year", "topic", "year_topic_id"], how="left")
    labels = refined.apply(
        lambda row: specific_label_cn(
            row.get("topic_keywords", ""),
            str(row.get("representative_titles", "")).split(" || ") if pd.notna(row.get("representative_titles", "")) else [],
        ),
        axis=1,
    )
    refined["specific_label_cn"] = [item[0] for item in labels]
    refined["specific_parent_category"] = [item[1] for item in labels]
    refined["specific_label_en"] = [item[2] for item in labels]
    refined["naming_evidence"] = refined.apply(
        lambda row: f"keywords={row.get('topic_keywords', '')}; title_terms={row.get('title_terms', '')}",
        axis=1,
    )
    refined.to_csv(results_dir / "topic_summary_by_year_refined.csv", index=False)

    join_cols = [
        "year",
        "topic",
        "year_topic_id",
        "specific_label_cn",
        "specific_parent_category",
        "specific_label_en",
        "representative_titles",
        "representative_venues",
        "title_terms",
    ]
    venue_refined = topic_by_venue.merge(refined[join_cols], on=["year", "topic", "year_topic_id"], how="left")
    venue_refined["share_pct"] = venue_refined["share"] * 100
    venue_refined.sort_values(["venue", "year", "count"], ascending=[True, True, False], inplace=True)
    venue_refined.to_csv(results_dir / "topic_by_venue_yearly_refined.csv", index=False)

    top_rows = []
    for (venue, year), group in venue_refined.groupby(["venue", "year"], sort=True):
        for rank, (_, row) in enumerate(group.sort_values("count", ascending=False).head(top_k_per_venue_year).iterrows(), 1):
            top_rows.append(
                {
                    "venue": venue,
                    "year": int(year),
                    "rank": rank,
                    "year_topic_id": row["year_topic_id"],
                    "count": int(row["count"]),
                    "venue_year_total": int(row["venue_year_total"]),
                    "share_pct": round(float(row["share_pct"]), 2),
                    "specific_label_cn": row["specific_label_cn"],
                    "short_keywords": ", ".join(split_keywords(row.get("topic_keywords", ""))[:5]),
                    "representative_titles": row.get("representative_titles", ""),
                }
            )
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(results_dir / f"top{top_k_per_venue_year}_topics_by_venue_year_refined.csv", index=False)

    lines = ["# Refined Top Topics By Venue-Year", ""]
    for (venue, year), group in top_df.groupby(["venue", "year"], sort=True):
        lines.append(f"## {venue} {int(year)}")
        lines.append("")
        display = group[["rank", "specific_label_cn", "count", "venue_year_total", "share_pct", "short_keywords"]].copy()
        display.rename(
            columns={
                "rank": "排名",
                "specific_label_cn": "细主题名",
                "count": "篇数",
                "venue_year_total": "当年会议篇数",
                "share_pct": "占比%",
                "short_keywords": "关键词",
            },
            inplace=True,
        )
        lines.append(display.to_markdown(index=False))
        lines.append("")
    (results_dir / f"top{top_k_per_venue_year}_topics_by_venue_year_refined.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create more specific topic names for yearly BERTopic outputs.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-safe", default=DEFAULT_MODEL_SAFE)
    parser.add_argument("--top-n-titles", type=int, default=5)
    parser.add_argument("--top-k-per-venue-year", type=int, default=10)
    args = parser.parse_args()
    refine_results(Path(args.results_dir), args.model_safe, args.top_n_titles, args.top_k_per_venue_year)
    print(f"Refined topic names written under: {args.results_dir}")


if __name__ == "__main__":
    main()
