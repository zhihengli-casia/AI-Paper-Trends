import argparse
from importlib import import_module
from pathlib import Path

import yaml

from src.utils import build_pipeline_paths, ensure_directories, validate_config

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_config_path(config_path: str) -> Path:
    """Allows config paths relative to the current directory or the project root."""
    candidate = Path(config_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    project_relative_candidate = PROJECT_ROOT / candidate
    if project_relative_candidate.exists():
        return project_relative_candidate.resolve()

    return candidate.resolve()


def load_config(config_path: Path) -> dict:
    """Loads, parses, and validates the YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    return validate_config(raw_config)


def load_step_module(module_name: str):
    return import_module(f"src.{module_name}")


def main() -> int:
    """Main entry point and scheduler for the project."""
    parser = argparse.ArgumentParser(
        description="Main entry point for the academic paper trend analysis framework."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file in the configs/ directory.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerunning all steps, even if cached files exist.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return 1

    print(f"Loading configuration from: {config_path}")
    try:
        config = load_config(config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Failed to load configuration: {exc}")
        return 1

    data_raw_dir = PROJECT_ROOT / "data" / "raw"
    data_processed_dir = PROJECT_ROOT / "data" / "processed"
    results_root = PROJECT_ROOT / "results"

    paths = build_pipeline_paths(
        config=config,
        raw_data_dir=data_raw_dir,
        processed_data_dir=data_processed_dir,
        results_root=results_root,
    )
    ensure_directories(data_raw_dir, data_processed_dir, paths.output_dir)

    raw_data_path = paths.raw_data_path
    processed_data_path = paths.processed_data_path

    print(f"Results will be saved to: {paths.output_dir}")

    print("\n--- [Step 1/3] Data Fetching ---")
    if not args.force_rerun and raw_data_path.exists():
        print("File already exists, skipping data fetching.")
        print(f"   --> Using local file: {raw_data_path}")
    else:
        print("   --> Starting data fetching (local file not found or --force-rerun)...")
        get_papers = load_step_module("get_papers")
        returned_path = get_papers.main(config=config, raw_data_dir=data_raw_dir)
        if not returned_path:
            print("Data fetching failed. Terminating process.")
            return 1
        raw_data_path = returned_path
    print("--- [Step 1/3] Data Fetching Complete ---\n")

    if config.get("topic_modeling", {}).get("enabled", False):
        print("--- [Step 2/3] Topic Modeling ---")
        if not args.force_rerun and processed_data_path.exists():
            print("File already exists, skipping topic modeling.")
            print(f"   --> Using local file: {processed_data_path}")
        else:
            print("   --> Starting topic modeling (local file not found or --force-rerun)...")
            if not raw_data_path.exists():
                print(f"Input file {raw_data_path} does not exist. Cannot perform topic modeling.")
                return 1
            run_topic_modeling = load_step_module("run_topic_modeling")
            returned_path = run_topic_modeling.main(
                config=config,
                input_path=raw_data_path,
                processed_data_dir=data_processed_dir,
                output_dir=paths.output_dir,
            )
            if not returned_path:
                print("Topic modeling failed. Terminating process.")
                return 1
            processed_data_path = returned_path
    else:
        print("--- [Step 2/3] Topic modeling is disabled in the configuration. ---")
    print("--- [Step 2/3] Topic Modeling Complete ---\n")

    if config.get("analysis", {}).get("enabled", False):
        print("--- [Step 3/3] Analysis and Visualization ---")
        if processed_data_path.exists():
            analyze = load_step_module("analyze")
            analyze.main(config=config, input_path=processed_data_path, output_dir=paths.output_dir)
        else:
            print(
                f"Cannot perform analysis because the input file {processed_data_path} does not exist."
            )
            return 1
    else:
        print("--- [Step 3/3] Analysis and visualization is disabled in the configuration. ---")
    print("--- [Step 3/3] Analysis and Visualization Complete ---\n")

    print("All processes completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
