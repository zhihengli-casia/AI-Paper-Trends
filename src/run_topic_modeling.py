import os
from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from modelscope.hub.snapshot_download import snapshot_download

def load_and_preprocess_data(file_path: Path) -> (pd.DataFrame, List[str]):
    """Loads and preprocesses data from a JSONL file."""
    print(f"Loading data from {file_path}...")
    if not file_path.exists():
        print(f"Error: Input file not found at {file_path}")
        return None, None
    
    df = pd.read_json(file_path, lines=True)
    df.dropna(subset=['title', 'abstract'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['keywords_str'] = df['keywords'].apply(lambda kw: ' '.join(kw) if isinstance(kw, list) else '')
    df['text_for_analysis'] = df['title'] + ". " + df['keywords_str'] + ". " + df['abstract']
    
    print(f"Data loaded successfully. Found {len(df)} valid papers.")
    return df, df['text_for_analysis'].tolist()

def download_embedding_model(model_id: str, cache_dir: Path) -> Path:
    """Checks for a local model and downloads it if not present."""
    print(f"\n--- Checking for embedding model: {model_id} ---")
    expected_model_path = cache_dir / model_id

    if not expected_model_path.exists():
        print(f"Local model not found. Downloading...")
        snapshot_download(model_id, cache_dir=str(cache_dir))
        print("Model downloaded successfully.")
    else:
        print("Local model already exists, skipping download.")
    return expected_model_path

def main(config: Dict[str, Any], input_path: Path, processed_data_dir: Path, output_dir: Path) -> Path:
    """Main function to be called as a module for topic modeling."""
    tm_config = config.get('topic_modeling', {})
    model_id = tm_config.get('model_id', 'sentence-transformers/all-mpnet-base-v2')
    min_topic_size = tm_config.get('min_topic_size', 30)
    
    df, docs = load_and_preprocess_data(input_path)
    if df is None: 
        return None

    effective_min_topic_size = min_topic_size
    if len(docs) < effective_min_topic_size:
        effective_min_topic_size = max(2, len(docs) // 2) 
        print(f"\nWarning: Number of documents ({len(docs)}) is less than min_topic_size ({min_topic_size}).")
        print(f"   --> Auto-adjusting min_topic_size to {effective_min_topic_size}.")

    models_cache_dir = Path('./models')
    models_cache_dir.mkdir(exist_ok=True)
    local_model_path = download_embedding_model(model_id, models_cache_dir)
    
    print(f"Loading embedding model from local path: {local_model_path}")
    embedding_model = SentenceTransformer(str(local_model_path))
    
    print("\n--- Starting BERTopic topic modeling ---")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=effective_min_topic_size,
        verbose=True,
        language="english"
    )
    
    topics, _ = topic_model.fit_transform(docs)
    print("\n--- Topic modeling complete ---")
    
    base_name = input_path.stem
    
    output_model_path = output_dir / f"bertopic_model_{base_name}"
    topic_model.save(str(output_model_path), serialization="safetensors")
    print(f"BERTopic model saved to: {output_model_path}")
    
    output_data_filename = f"{base_name}_with_topics.csv"
    output_data_path = processed_data_dir / output_data_filename
    df['Topic'] = topics
    df.to_csv(output_data_path, index=False, encoding='utf-8-sig')
    print(f"Paper data with topics saved to: {output_data_path}")

    return output_data_path