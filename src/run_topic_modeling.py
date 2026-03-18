from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.utils import build_processed_output_path, ensure_directories

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_and_preprocess_data(file_path: Path) -> tuple[Optional[pd.DataFrame], List[str]]:
    """Loads and preprocesses data from a JSONL file."""
    print(f"Loading data from {file_path}...")
    if not file_path.exists():
        print(f"Error: Input file not found at {file_path}")
        return None, []

    df = pd.read_json(file_path, lines=True)
    required_columns = {"title", "abstract"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        print(f"Error: Input data is missing required columns: {', '.join(missing_columns)}")
        return None, []

    if "keywords" not in df.columns:
        df["keywords"] = [[] for _ in range(len(df))]

    df.dropna(subset=["title", "abstract"], inplace=True)
    df["title"] = df["title"].astype(str).str.strip()
    df["abstract"] = df["abstract"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["abstract"] != "")]
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print("Error: No valid papers remain after preprocessing.")
        return None, []

    df["keywords_str"] = df["keywords"].apply(
        lambda keywords: " ".join(keywords) if isinstance(keywords, list) else str(keywords or "")
    )
    df["text_for_analysis"] = (
        df["title"] + ". " + df["keywords_str"].fillna("") + ". " + df["abstract"]
    ).str.strip()

    print(f"Data loaded successfully. Found {len(df)} valid papers.")
    return df, df["text_for_analysis"].tolist()


def download_embedding_model(model_id: str, cache_dir: Path) -> Path:
    """Resolves a local model path or downloads it into the cache directory."""
    candidate_path = Path(model_id).expanduser()
    if candidate_path.exists():
        print(f"Using local embedding model path: {candidate_path.resolve()}")
        return candidate_path.resolve()

    print(f"\n--- Checking for embedding model: {model_id} ---")
    expected_model_path = cache_dir / model_id
    if expected_model_path.exists():
        print("Local model already exists, skipping download.")
        return expected_model_path

    print("Local model not found. Downloading...")
    from modelscope.hub.snapshot_download import snapshot_download

    downloaded_path = snapshot_download(model_id, cache_dir=str(cache_dir))
    downloaded_model_path = Path(downloaded_path)
    print(f"Model downloaded successfully to: {downloaded_model_path}")
    return downloaded_model_path


def build_topic_labels(topic_model: Any, topics: List[int], top_n_words: int = 3) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        topic_terms = topic_model.get_topic(topic_id) or []
        label_words = [word for word, _score in topic_terms[:top_n_words]]
        labels[topic_id] = ", ".join(label_words) if label_words else f"Topic {topic_id}"
    return labels


def save_topic_labels(topic_labels: Dict[int, str], output_dir: Path) -> Path:
    labels_path = output_dir / "topic_labels.yaml"
    with labels_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(topic_labels, file, sort_keys=True, allow_unicode=True)
    print(f"Topic labels saved to: {labels_path}")
    return labels_path


def save_single_topic_output(
    df: pd.DataFrame,
    input_path: Path,
    processed_data_dir: Path,
    output_dir: Path,
) -> Path:
    """BERTopic needs at least two documents, so keep the pipeline usable for tiny datasets."""
    print("Only one valid paper was found. Skipping BERTopic and assigning a single fallback topic.")
    output_data_path = build_processed_output_path(input_path, processed_data_dir)
    df = df.copy()
    df["Topic"] = 0
    df.to_csv(output_data_path, index=False, encoding="utf-8-sig")
    print(f"Paper data with fallback topic saved to: {output_data_path}")
    save_topic_labels({0: "Single Topic"}, output_dir)
    return output_data_path


def main(config: Dict[str, Any], input_path: Path, processed_data_dir: Path, output_dir: Path) -> Path | None:
    """Main function to be called as a module for topic modeling."""
    tm_config = config.get("topic_modeling", {})
    model_id = tm_config.get("model_id", "sentence-transformers/all-mpnet-base-v2")
    min_topic_size = tm_config.get("min_topic_size", 30)
    label_word_count = tm_config.get("label_word_count", 3)

    ensure_directories(processed_data_dir, output_dir)

    df, docs = load_and_preprocess_data(input_path)
    if df is None:
        return None

    if len(docs) == 1:
        return save_single_topic_output(df, input_path, processed_data_dir, output_dir)

    effective_min_topic_size = min_topic_size
    if len(docs) < effective_min_topic_size:
        effective_min_topic_size = min(min_topic_size, max(2, len(docs) // 2))
        print(
            f"\nWarning: Number of documents ({len(docs)}) is less than min_topic_size "
            f"({min_topic_size})."
        )
        print(f"   --> Auto-adjusting min_topic_size to {effective_min_topic_size}.")

    models_cache_dir = PROJECT_ROOT / "models"
    ensure_directories(models_cache_dir)
    local_model_path = download_embedding_model(model_id, models_cache_dir)

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model from local path: {local_model_path}")
    embedding_model = SentenceTransformer(str(local_model_path))

    print("\n--- Starting BERTopic topic modeling ---")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=effective_min_topic_size,
        verbose=True,
        language="english",
    )

    topics, _ = topic_model.fit_transform(docs)
    print("\n--- Topic modeling complete ---")

    base_name = input_path.stem
    output_model_path = output_dir / f"bertopic_model_{base_name}"
    topic_model.save(str(output_model_path), serialization="safetensors")
    print(f"BERTopic model saved to: {output_model_path}")

    output_data_path = build_processed_output_path(input_path, processed_data_dir)
    df["Topic"] = topics
    df.to_csv(output_data_path, index=False, encoding="utf-8-sig")
    print(f"Paper data with topics saved to: {output_data_path}")

    topic_labels = build_topic_labels(topic_model, topics, top_n_words=label_word_count)
    save_topic_labels(topic_labels, output_dir)

    return output_data_path
