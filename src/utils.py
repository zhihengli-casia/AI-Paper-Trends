from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

VALID_ANALYSIS_TASKS = {
    "plot_paper_count",
    "plot_avg_rating",
    "plot_decision_breakdown",
    "generate_summary_table",
}

DECISION_TYPES = ["Oral", "Spotlight", "Poster", "Reject", "N/A"]


@dataclass(frozen=True)
class PipelinePaths:
    raw_data_path: Path
    processed_data_path: Path
    output_dir: Path


def extract_content_value(content: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Reads an OpenReview content field regardless of whether it is wrapped in {'value': ...}."""
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value.get("value", default)
    if value is None:
        return default
    return value


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def sanitize_conference_id(conference_id: str) -> str:
    return conference_id.replace("/", "_").replace(".", "")


def normalize_limit(limit: Any) -> int | None:
    if limit in (None, "", 0):
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("'limit' must be a positive integer or null.")
    return limit


def build_raw_output_path(
    conference_id: str,
    raw_data_dir: Path,
    fetch_reviews: bool = False,
    limit: int | None = None,
) -> Path:
    suffix = "_reviews" if fetch_reviews else ""
    limit_suffix = f"_limit{limit}" if limit else ""
    output_filename = f"{sanitize_conference_id(conference_id)}_papers{suffix}{limit_suffix}.jsonl"
    return raw_data_dir / output_filename


def build_processed_output_path(raw_data_path: Path, processed_data_dir: Path) -> Path:
    return processed_data_dir / f"{raw_data_path.stem}_with_topics.csv"


def build_pipeline_paths(
    config: Mapping[str, Any],
    raw_data_dir: Path,
    processed_data_dir: Path,
    results_root: Path,
) -> PipelinePaths:
    raw_data_path = build_raw_output_path(
        conference_id=config["conference_id"],
        raw_data_dir=raw_data_dir,
        fetch_reviews=config.get("fetch_reviews", False),
        limit=config.get("limit"),
    )
    processed_data_path = build_processed_output_path(raw_data_path, processed_data_dir)
    output_dir = results_root / config.get("output_folder_name", "default_analysis")
    return PipelinePaths(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        output_dir=output_dir,
    )


def validate_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalizes user config and validates the fields the pipeline depends on."""
    if not isinstance(config, Mapping):
        raise ValueError("Configuration file must define a YAML mapping.")

    normalized = dict(config)
    conference_id = normalized.get("conference_id")
    if not isinstance(conference_id, str) or not conference_id.strip():
        raise ValueError("'conference_id' is required and must be a non-empty string.")

    normalized["conference_id"] = conference_id.strip()
    normalized["fetch_reviews"] = bool(normalized.get("fetch_reviews", False))
    normalized["limit"] = normalize_limit(normalized.get("limit"))

    topic_modeling = dict(normalized.get("topic_modeling") or {})
    analysis = dict(normalized.get("analysis") or {})

    topic_modeling["enabled"] = bool(topic_modeling.get("enabled", False))
    if topic_modeling["enabled"]:
        min_topic_size = topic_modeling.get("min_topic_size", 30)
        if isinstance(min_topic_size, bool) or not isinstance(min_topic_size, int) or min_topic_size < 2:
            raise ValueError("'topic_modeling.min_topic_size' must be an integer >= 2.")
        label_word_count = topic_modeling.get("label_word_count", 3)
        if isinstance(label_word_count, bool) or not isinstance(label_word_count, int) or label_word_count < 1:
            raise ValueError("'topic_modeling.label_word_count' must be an integer >= 1.")
        topic_modeling["min_topic_size"] = min_topic_size
        topic_modeling["label_word_count"] = label_word_count

    analysis["enabled"] = bool(analysis.get("enabled", False))
    tasks = analysis.get("tasks", [])
    if analysis["enabled"]:
        if not isinstance(tasks, list):
            raise ValueError("'analysis.tasks' must be a list.")
        top_n = analysis.get("top_n", 65)
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("'analysis.top_n' must be an integer >= 1.")
        analysis["top_n"] = top_n
        unknown_tasks = [task for task in tasks if task not in VALID_ANALYSIS_TASKS]
        if unknown_tasks:
            raise ValueError(
                "Unsupported analysis task(s): " + ", ".join(sorted(set(unknown_tasks)))
            )
    analysis["tasks"] = tasks

    output_folder_name = normalized.get("output_folder_name", "default_analysis")
    if not isinstance(output_folder_name, str) or not output_folder_name.strip():
        raise ValueError("'output_folder_name' must be a non-empty string when provided.")

    normalized["output_folder_name"] = output_folder_name.strip()
    normalized["topic_modeling"] = topic_modeling
    normalized["analysis"] = analysis
    return normalized
