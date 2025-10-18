import argparse
import yaml
from pathlib import Path
import sys

# Ensure the 'src' directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from src import get_papers, run_topic_modeling, analyze

def load_config(config_path: str) -> dict:
    """Loads and parses the YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_expected_filepaths(config: dict, raw_dir: Path, processed_dir: Path) -> dict:
    """Pre-builds all expected input and output file paths based on the configuration."""
    conf_id = config['conference_id'].replace('/', '_').replace('.', '')
    limit = config.get('limit', None)
    fetch_reviews = config.get('fetch_reviews', False)

    suffix = "_reviews" if fetch_reviews else ""
    limit_suffix = f"_limit{limit}" if limit else ""
    raw_filename = f"{conf_id}_papers{suffix}{limit_suffix}.jsonl"
    raw_path = raw_dir / raw_filename

    processed_filename = f"{raw_path.stem}_with_topics.csv"
    processed_path = processed_dir / processed_filename
    
    return {"raw": raw_path, "processed": processed_path}

def main():
    """Main entry point and scheduler for the project."""
    parser = argparse.ArgumentParser(description="Main entry point for the academic paper trend analysis framework.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file in the configs/ directory.")
    parser.add_argument('--force-rerun', action='store_true', help="Force rerunning all steps, even if cached files exist.")
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    output_folder_name = config.get('output_folder_name', 'default_analysis')
    output_dir = Path("results") / output_folder_name
    data_raw_dir = Path("data/raw")
    data_processed_dir = Path("data/processed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    expected_paths = get_expected_filepaths(config, data_raw_dir, data_processed_dir)
    raw_data_path = expected_paths['raw']
    processed_data_path = expected_paths['processed']
    
    # Step 1: Data Fetching
    print("\n--- [Step 1/3] Data Fetching ---")
    if not args.force_rerun and raw_data_path.exists():
        print(f"File already exists, skipping data fetching.")
        print(f"   --> Using local file: {raw_data_path}")
    else:
        print("   --> Starting data fetching (local file not found or --force-rerun)...")
        returned_path = get_papers.main(config=config, raw_data_dir=data_raw_dir)
        if not returned_path:
            print("Data fetching failed. Terminating process.")
            return
    print("--- [Step 1/3] Data Fetching Complete ---\n")

    # Step 2: Topic Modeling
    if config.get('topic_modeling', {}).get('enabled', False):
        print("--- [Step 2/3] Topic Modeling ---")
        if not args.force_rerun and processed_data_path.exists():
            print(f"File already exists, skipping topic modeling.")
            print(f"   --> Using local file: {processed_data_path}")
        else:
            print("   --> Starting topic modeling (local file not found or --force-rerun)...")
            if not raw_data_path.exists():
                 print(f"Input file {raw_data_path} does not exist. Cannot perform topic modeling.")
                 return
            returned_path = run_topic_modeling.main(
                config=config, input_path=raw_data_path, 
                processed_data_dir=data_processed_dir, output_dir=output_dir
            )
            if not returned_path:
                print("Topic modeling failed. Terminating process.")
                return
    else:
        print("--- [Step 2/3] Topic modeling is disabled in the configuration. ---")
    print("--- [Step 2/3] Topic Modeling Complete ---\n")

    # Step 3: Analysis and Visualization
    if config.get('analysis', {}).get('enabled', False):
        print("--- [Step 3/3] Analysis and Visualization ---")
        if processed_data_path.exists():
            analyze.main(config=config, input_path=processed_data_path, output_dir=output_dir)
        else:
            print(f"Cannot perform analysis because the input file {processed_data_path} does not exist.")
    else:
        print("--- [Step 3/3] Analysis and visualization is disabled in the configuration. ---")
    print("--- [Step 3/3] Analysis and Visualization Complete ---\n")

    print("All processes completed successfully.")

if __name__ == "__main__":
    main()