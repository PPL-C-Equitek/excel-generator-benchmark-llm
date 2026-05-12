"""CLI entrypoint for running batch LLM benchmark examples."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections.abc import Iterable
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))  # pragma: no cover

from src.data_extractor import (  # noqa: E402
    SUPPORTED_FILE_EXTENSIONS,
    extract_text_from_file,
)
from src.category_accuracy_report import (  # noqa: E402
    generate_category_accuracy_reports,
)
from src.recommendation_improvements import (  # noqa: E402
    generate_recommendation_improvements,
)
from src.runner import BenchmarkRunner  # noqa: E402


DEFAULT_MODEL_NAMES = (
    "claude-sonnet-4-6",
    "gpt-5.2-codex",
    "gemini-3.1-pro-preview",
    "deepseek-v3-2",
)
DEFAULT_EXAMPLE_DIRS = (
    "synthetic.examples",
    "synthetic.examples.lanjutan",
    "like-real.examples",
)
DEFAULT_DATA_DIR = "data"
DEFAULT_REPORT_DIR = "benchmark_reports"
DEFAULT_RUNTIME_DATASET_DIR = "benchmark_runtime_datasets"
OVERALL_REPORT_FILENAME = "overall_benchmark_report.csv"
OVERALL_TEXT_REPORT_FILENAME = "overall_benchmark_report.txt"
PRESERVE_OVERALL_REPORTS_ENV = "PRESERVE_OVERALL_REPORTS"
CSV_ENCODING = "utf-8-sig"
JSON_ENCODING = "utf-8"
SUPPORTED_INPUT_EXTENSIONS = SUPPORTED_FILE_EXTENSIONS
REPORT_HEADERS = ["unit", "item", "num_type", "status_type", "value"]
OVERALL_REPORT_FIELDNAMES = [
    "section",
    "case_id",
    "model",
    "best_model",
    "best_score",
    "fastest_model",
    "fastest_seconds",
    "average_score",
    "average_seconds",
    "total_runs",
    "completed_runs",
]
# Default source filter for benchmark-report ingestion:
# include only `synthetic_examples` unless caller overrides it explicitly.
DEFAULT_BENCHMARK_REPORT_SOURCES = ("synthetic_examples",)
REPORT_COMPLETED_STATUS = "completed"
MODEL_SUMMARY_TITLE = "Model Summary"
OVERALL_WINNERS_TITLE = "Overall Winners"
REPORT_SECTIONS_WITH_TABLES = (MODEL_SUMMARY_TITLE, OVERALL_WINNERS_TITLE)
OVERALL_WINNER_FASTEST_LABEL = "Fastest average runtime"
DERIVED_TEXT_REPORT_FILENAME = "overall_benchmark_report_source_augmented.txt"
SYSTEM_PROMPT = """
You are a document parsing assistant.

Return ONLY a valid JSON object with no markdown, no code fences, and no
explanation.

The object must have exactly three top-level keys: document_info, summary, and
content_data.

document_info must be an object with source_type (must be exactly "Excel" or
"PDF", case-sensitive) and filename (non-empty string).

summary must be an object with non-empty string keys and scalar values only
(string, number, boolean, or null), no nested objects or arrays.

content_data must be a non-empty array of table objects, each with table_name
(non-empty string, unique), headers (non-empty array of unique non-empty
strings), and rows (array of objects where each object keys match headers
exactly with scalar values only, no nested objects or arrays).

For Excel files, if the file contains multiple sheets, each sheet must be
represented as a separate table object in content_data with table_name set to
the sheet name. Do not merge sheets together.

For PDF files, all extracted content must be combined into a single table
object in content_data regardless of page count.

If the input data contains columns that represent categorical groupings such as
department names, regions, or units, unpivot those columns into rows to produce
a normalized long format table where each row represents a single observation,
each categorical column header becomes a value in a new column called "unit",
and its corresponding cell value becomes a separate column called "value", with
all other columns repeated for each unpivoted row.

The resulting long format table must always use these exact column names: unit,
item, num_type, status_type, value. Never use translated or alternative column
names such as "Nilai", "Tipe", "Status", "Item", or any other language
variant.

Exclude any rows where the value is 0 or null after unpivoting.

The following is an example of the REQUIRED FORMAT ONLY. Do NOT return this
example data; always parse and return the actual uploaded file content:
{"document_info":{"source_type":"Excel","filename":"example.xlsx"},"summary":{"total_sheets":1,"total_rows":2,"total_columns":5},"content_data":[{"table_name":"Sheet1","headers":["unit","item","num_type","status_type","value"],"rows":[{"unit":"IT","item":"Laptop","num_type":"cost","status_type":"target","value":15000000},{"unit":"IT","item":"Laptop","num_type":"cost","status_type":"actual","value":14000000}]}]}
""".strip()


@dataclass(frozen=True)
class ExampleCase:
    """Input and expected-output files for one benchmark example."""

    case_id: str
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class SkippedCase:
    """Example file that cannot be benchmarked by this text CLI."""

    input_path: Path
    reason: str


def main() -> int:
    """Run all supported example files against all configured models.

    Environment variables:
        MODEL_NAMES: Optional comma-separated model list. When unset, all
            default SUMOPOD models are used.
        EXAMPLE_DIRS: Optional comma-separated example folders.
        REPORT_DIR: Optional destination folder for per-run CSV reports.
        BENCHMARK_DATASET_DIR: Optional folder for generated runtime datasets.
        PRESERVE_OVERALL_REPORTS: Optional boolean-like flag. When enabled,
            overall CSV/TXT reports use non-overwriting filename suffixes such
            as ``overall_benchmark_report_1.csv`` when base filenames exist.

    Returns:
        Process exit code. ``0`` means the batch runner completed.
    """
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    model_names = _model_names_from_env()
    example_dirs = _example_dirs_from_env()
    input_extensions = _input_extensions_from_env()
    merge_base_overall_report = _merge_base_overall_report_from_env()
    preserve_overall_reports = _preserve_overall_reports_from_env()
    data_dir = _project_path(DEFAULT_DATA_DIR)
    report_dir = _project_output_path_from_env("REPORT_DIR", DEFAULT_REPORT_DIR)
    runtime_dataset_dir = _project_output_path_from_env(
        "BENCHMARK_DATASET_DIR",
        DEFAULT_RUNTIME_DATASET_DIR,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    runtime_dataset_dir.mkdir(parents=True, exist_ok=True)

    cases, skipped_cases = _discover_example_cases(example_dirs, input_extensions)

    print("========================================")
    print("LLM Benchmark Batch Runner")
    print("========================================")
    print(f"Models     : {', '.join(model_names)}")
    print(f"Extensions : {', '.join(sorted(input_extensions))}")
    print(f"Examples   : {len(cases)} supported")
    print(f"Skipped    : {len(skipped_cases)} unsupported")
    print(f"Report dir : {report_dir}")
    print("")

    if skipped_cases:
        print("Skipped files")
        print("-------------")
        for skipped_case in skipped_cases:
            print(
                f"- {_display_path(skipped_case.input_path)}: "
                f"{skipped_case.reason}"
            )
        print("")

    batch_results = _run_batch(
        cases=cases,
        model_names=model_names,
        report_dir=report_dir,
        runtime_dataset_dir=runtime_dataset_dir,
    )
    if merge_base_overall_report is None:
        overall_report_path = _write_overall_report(
            batch_results,
            report_dir,
            preserve_existing=preserve_overall_reports,
        )
    else:
        overall_report_path = _write_overall_report_with_merge(
            batch_results,
            report_dir,
            merge_base_overall_report,
            preserve_existing=preserve_overall_reports,
        )
    overall_text_report_path = _write_overall_text_report(
        batch_results,
        report_dir,
        overall_report_path=overall_report_path,
        preserve_existing=preserve_overall_reports,
    )
    category_accuracy_paths = _write_category_accuracy_reports(
        report_dir=report_dir,
        runtime_dataset_dir=runtime_dataset_dir,
    )
    _print_batch_summary(
        batch_results,
        overall_report_path,
        overall_text_report_path,
        category_accuracy_paths,
    )

    return 0


def _write_category_accuracy_reports(
    *,
    report_dir: Path,
    runtime_dataset_dir: Path,
) -> dict[str, Path]:
    """Write category-accuracy JSON and TXT reports from existing artifacts."""
    result = generate_category_accuracy_reports(
        report_dir=report_dir,
        runtime_dir=runtime_dataset_dir,
        output_dir=report_dir,
    )
    json_path = Path(result["json_path"])
    txt_path = Path(result["txt_path"])
    _append_recommendations_to_category_text_report(
        category_json_path=json_path,
        category_txt_path=txt_path,
    )
    return {
        "json_path": json_path,
        "txt_path": txt_path,
    }


def _append_recommendations_to_category_text_report(
    *,
    category_json_path: Path,
    category_txt_path: Path,
) -> None:
    """Append recommendation improvements section to category TXT report."""
    safe_json_path = _ensure_project_child(
        category_json_path,
        "category accuracy json path",
    )
    safe_txt_path = _ensure_project_child(
        category_txt_path,
        "category accuracy txt path",
    )

    try:
        payload = json.loads(safe_json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return

    if not isinstance(payload, dict):
        return

    category_scores = _extract_category_scores_for_recommendation(payload)
    recommendation_text = generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )

    try:
        existing_text = safe_txt_path.read_text(encoding="utf-8")
    except OSError:
        return

    separator = "\n" if existing_text.endswith("\n") else "\n\n"
    try:
        safe_txt_path.write_text(
            f"{existing_text}{separator}{recommendation_text}",
            encoding="utf-8",
        )
    except OSError:
        return


def _extract_category_scores_for_recommendation(
    report_payload: dict[str, Any],
) -> dict[str, float]:
    """Extract 0..1 category exact-accuracy scores from report payload."""
    raw_by_category = report_payload.get("by_category")
    if not isinstance(raw_by_category, dict):
        return {}

    category_scores: dict[str, float] = {}
    for category_name, summary in raw_by_category.items():
        if not isinstance(summary, dict):
            continue
        raw_percent = summary.get("exact_accuracy_percent", 0.0)
        try:
            percent_value = float(raw_percent)
        except (TypeError, ValueError):
            continue
        normalized_score = (
            percent_value / 100.0 if percent_value > 1.0 else percent_value
        )
        category_scores[str(category_name)] = max(0.0, min(1.0, normalized_score))

    return category_scores


def _model_names_from_env() -> list[str]:
    """Read benchmark model names from ``MODEL_NAMES`` or defaults.

    Returns:
        Ordered list of model names to benchmark.
    """
    raw_model_names = os.getenv("MODEL_NAMES")
    if raw_model_names:
        return _csv_env_values(raw_model_names)

    return list(DEFAULT_MODEL_NAMES)


def _input_extensions_from_env() -> set[str]:
    """Read allowed benchmark input extensions from ``INPUT_EXTENSIONS``.

    Returns:
        Normalized lowercase extension set (for example ``{".png", ".pdf"}``).

    Raises:
        ValueError: If an extension token is empty after normalization.
    """
    raw_extensions = os.getenv("INPUT_EXTENSIONS")
    if not raw_extensions:
        return set(SUPPORTED_INPUT_EXTENSIONS)

    extensions: set[str] = set()
    for value in _csv_env_values(raw_extensions):
        normalized_value = value.lower()
        if not normalized_value.startswith("."):
            normalized_value = f".{normalized_value}"
        if normalized_value == ".":
            raise ValueError("INPUT_EXTENSIONS contains an invalid extension token.")
        if normalized_value not in SUPPORTED_INPUT_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension in INPUT_EXTENSIONS: {normalized_value}"
            )
        extensions.add(normalized_value)

    return extensions


def _merge_base_overall_report_from_env() -> Path | None:
    """Read optional base overall report path for merged aggregation.

    Returns:
        Resolved report path when ``MERGE_BASE_OVERALL_REPORT`` is set,
        otherwise ``None``.
    """
    raw_path = os.getenv("MERGE_BASE_OVERALL_REPORT")
    if not raw_path:
        return None

    return _ensure_project_child(
        _project_path(raw_path),
        "MERGE_BASE_OVERALL_REPORT",
    )


def _preserve_overall_reports_from_env() -> bool:
    """Read the optional overall-report preservation flag from env."""
    raw_value = os.getenv(PRESERVE_OVERALL_REPORTS_ENV, "")
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _example_dirs_from_env() -> list[Path]:
    """Read example folders from ``EXAMPLE_DIRS`` or defaults.

    Returns:
        Ordered list of absolute example folder paths.
    """
    raw_example_dirs = os.getenv("EXAMPLE_DIRS")
    raw_values = (
        _csv_env_values(raw_example_dirs)
        if raw_example_dirs
        else list(DEFAULT_EXAMPLE_DIRS)
    )
    return [
        _ensure_project_child(_project_path(value), "EXAMPLE_DIRS")
        for value in raw_values
    ]


def _csv_env_values(raw_value: str) -> list[str]:
    """Split a comma-separated environment value into non-empty entries.

    Args:
        raw_value: Comma-separated environment value.

    Returns:
        Trimmed non-empty values.
    """
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _project_path_from_env(env_name: str, default: str) -> Path:
    """Resolve a path environment variable relative to the project root.

    Args:
        env_name: Environment variable name.
        default: Default relative path.

    Returns:
        Absolute path.
    """
    return _project_path(os.getenv(env_name, default))


def _project_output_path_from_env(env_name: str, default: str) -> Path:
    """Resolve and validate an output directory from an environment variable.

    Args:
        env_name: Environment variable name.
        default: Default project-relative path.

    Returns:
        Resolved output directory path under ``PROJECT_ROOT``.

    Raises:
        ValueError: If the configured path escapes ``PROJECT_ROOT``.
    """
    return _ensure_project_child(
        _project_path_from_env(env_name, default),
        env_name,
    )


def _project_path(raw_path: str | Path) -> Path:
    """Resolve a path relative to the project root.

    Args:
        raw_path: Absolute or project-relative path.

    Returns:
        Absolute path.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _ensure_project_child(path: Path, label: str) -> Path:
    """Validate that a path remains inside the project root.

    Args:
        path: Candidate path.
        label: Human-readable source label for error messages.

    Returns:
        Resolved path under ``PROJECT_ROOT``.

    Raises:
        ValueError: If ``path`` resolves outside ``PROJECT_ROOT``.
    """
    project_root = PROJECT_ROOT.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(project_root):
        raise ValueError(
            f"{label} must stay inside the project directory: {resolved_path}"
        )

    return resolved_path


def _discover_example_cases(
    example_dirs: list[Path],
    input_extensions: set[str],
) -> tuple[list[ExampleCase], list[SkippedCase]]:
    """Discover supported input/output pairs in example folders.

    Args:
        example_dirs: Folders that contain ``*_input.*`` files.

    Returns:
        Supported cases and skipped unsupported inputs.
    """
    cases: list[ExampleCase] = []
    skipped_cases: list[SkippedCase] = []

    for example_dir in example_dirs:
        for input_path in sorted(example_dir.glob("*_input.*")):
            if input_path.suffix.lower() not in input_extensions:
                skipped_cases.append(
                    SkippedCase(
                        input_path=input_path,
                        reason=f"unsupported input format {input_path.suffix}",
                    )
                )
                continue

            output_path = _matching_output_path(input_path)
            if output_path is None:
                skipped_cases.append(
                    SkippedCase(
                        input_path=input_path,
                        reason="matching *_output.csv file was not found",
                    )
                )
                continue

            cases.append(
                ExampleCase(
                    case_id=_case_id(input_path),
                    input_path=input_path,
                    output_path=output_path,
                )
            )

    return cases, skipped_cases


def _matching_output_path(input_path: Path) -> Path | None:
    """Find the expected output CSV for an input file.

    Args:
        input_path: Example input file.

    Returns:
        Matching output CSV path, or ``None`` when no match exists.
    """
    prefix = input_path.stem.removesuffix("_input")
    extension_name = input_path.suffix.lower().lstrip(".")
    candidates = [
        input_path.with_name(f"{prefix}_{extension_name}_output.csv"),
        input_path.with_name(f"{prefix}_output.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _case_id(input_path: Path) -> str:
    """Build a stable case id for report filenames.

    Args:
        input_path: Example input file.

    Returns:
        Stable case identifier.
    """
    folder_name = input_path.parent.name.replace(".", "_")
    return f"{folder_name}__{input_path.stem.removesuffix('_input')}"


def _run_batch(
    *,
    cases: list[ExampleCase],
    model_names: list[str],
    report_dir: Path,
    runtime_dataset_dir: Path,
) -> list[dict[str, Any]]:
    """Run every supported example case against every model.

    Args:
        cases: Example cases to benchmark.
        model_names: Model names to run.
        report_dir: Destination folder for result CSVs.
        runtime_dataset_dir: Destination folder for generated datasets.

    Returns:
        Lightweight batch summaries for terminal output.
    """
    batch_results: list[dict[str, Any]] = []
    total_runs = len(cases) * len(model_names)
    run_index = 0

    for model_name in model_names:
        for example_case in cases:
            run_index += 1
            print(
                f"[{run_index}/{total_runs}] "
                f"{model_name} -> {example_case.case_id}"
            )
            try:
                result = _run_single_case(
                    example_case=example_case,
                    model_name=model_name,
                    report_dir=report_dir,
                    runtime_dataset_dir=runtime_dataset_dir,
                )
            except Exception as exc:
                print(f"  warning: {exc}")
                result = _failed_preprocess_result(
                    example_case=example_case,
                    model_name=model_name,
                    report_dir=report_dir,
                    error_message=str(exc),
                )
            batch_results.append(result)
            print(
                "  "
                f"status={result['status']} "
                f"score={result['average_score']:.4f} "
                f"report={result['report_path']}"
            )

    return batch_results


def _failed_preprocess_result(
    *,
    example_case: ExampleCase,
    model_name: str,
    report_dir: Path,
    error_message: str,
) -> dict[str, Any]:
    """Build a synthetic batch result for cases that fail before runner setup.

    Args:
        example_case: Example case that failed during preprocessing.
        model_name: Model that was being evaluated.
        report_dir: Destination folder for benchmark reports.
        error_message: Captured preprocessing error.

    Returns:
        Batch summary row with a failed preprocess status.
    """
    safe_model_name = _safe_filename(model_name)
    safe_case_id = _safe_filename(example_case.case_id)
    report_path = _safe_output_path(
        report_dir,
        f"{safe_model_name}__{safe_case_id}.csv",
    )
    return {
        "model": model_name,
        "case_id": example_case.case_id,
        "input_path": _display_path(example_case.input_path),
        "output_path": _display_path(example_case.output_path),
        "report_path": _display_path(report_path),
        "elapsed_seconds": 0.0,
        "status": "failed_preprocess",
        "total_rows": 0,
        "successful_evaluations": 0,
        "average_score": 0.0,
        "error_message": error_message,
    }


def _run_single_case(
    *,
    example_case: ExampleCase,
    model_name: str,
    report_dir: Path,
    runtime_dataset_dir: Path,
) -> dict[str, Any]:
    """Run one example case against one model.

    Args:
        example_case: Example case to benchmark.
        model_name: Model name to run.
        report_dir: Destination folder for the CSV report.
        runtime_dataset_dir: Destination folder for the generated dataset.

    Returns:
        Batch summary row.
    """
    safe_model_name = _safe_filename(model_name)
    safe_case_id = _safe_filename(example_case.case_id)
    report_path = _safe_output_path(
        report_dir,
        f"{safe_model_name}__{safe_case_id}.csv",
    )
    prompt = _build_prompt(
        example_case.input_path,
        extract_text_from_file(example_case.input_path),
    )
    expected_output = _expected_output_from_ground_truth(
        example_case.input_path,
        example_case.output_path,
    )
    runtime_dataset_path = _write_runtime_dataset(
        runtime_dataset_dir,
        f"{safe_model_name}__{safe_case_id}.json",
        prompt,
        expected_output,
    )

    runner = BenchmarkRunner(
        dataset_path=runtime_dataset_path,
        model=model_name,
        report_path=report_path,
    )
    start_time = time.perf_counter()
    summary = runner.run()
    elapsed_seconds = time.perf_counter() - start_time
    return {
        "model": model_name,
        "case_id": example_case.case_id,
        "input_path": _display_path(example_case.input_path),
        "output_path": _display_path(example_case.output_path),
        "report_path": _display_path(report_path),
        "elapsed_seconds": elapsed_seconds,
        **summary,
    }


def _display_path(path: Path) -> str:
    """Format a path relative to the project when possible.

    Args:
        path: Path to format for display.

    Returns:
        Project-relative path when ``path`` is under ``PROJECT_ROOT``,
        otherwise the original path string.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return os.path.relpath(path, PROJECT_ROOT)
        except ValueError:
            return str(path)


def _safe_filename(value: str) -> str:
    """Convert a model name or case id into a safe filename segment.

    Args:
        value: Raw value.

    Returns:
        Filesystem-safe value.
    """
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe_value or "unnamed"


def _safe_output_path(base_dir: Path, filename: str) -> Path:
    """Build a validated child path for generated benchmark artifacts.

    Args:
        base_dir: Controlled output directory.
        filename: Requested filename segment.

    Returns:
        Resolved path guaranteed to stay inside ``base_dir``.

    Raises:
        ValueError: If the resolved path escapes ``base_dir``.
    """
    resolved_base_dir = _ensure_project_child(base_dir, "output directory")
    candidate = (resolved_base_dir / _safe_filename(filename)).resolve()
    if not candidate.is_relative_to(resolved_base_dir):
        raise ValueError("Generated output path escapes the output directory.")

    return candidate


def _next_available_output_path(base_dir: Path, filename: str) -> Path:
    """Resolve a non-overwriting output path inside ``base_dir``.

    Args:
        base_dir: Controlled output directory.
        filename: Preferred filename.

    Returns:
        Preferred output path when it does not exist, otherwise a suffixed
        variant such as ``name_1.ext``.
    """
    preferred_path = _safe_output_path(base_dir, filename)
    if not preferred_path.exists():
        return preferred_path

    stem = preferred_path.stem
    suffix = preferred_path.suffix
    index = 1
    while True:
        candidate_name = f"{stem}_{index}{suffix}"
        candidate_path = _safe_output_path(base_dir, candidate_name)
        if not candidate_path.exists():
            return candidate_path
        index += 1


def _read_ground_truth_rows(path: Path) -> list[dict[str, Any]]:
    """Read normalized ground-truth rows from a CSV file.

    Args:
        path: Ground-truth CSV file path.

    Returns:
        List of normalized row dictionaries.
    """
    safe_path = _ensure_project_child(path, "ground truth csv path")
    rows: list[dict[str, Any]] = []
    with safe_path.open("r", newline="", encoding=CSV_ENCODING) as file:
        for row in csv.DictReader(file):
            clean_row = {key.strip(): value.strip() for key, value in row.items()}
            clean_row["value"] = int(clean_row["value"])
            rows.append(clean_row)

    return rows


def _expected_output_from_ground_truth(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build expected output matching the main document-parser schema.

    Args:
        input_path: Source file used in the prompt.
        output_path: Ground-truth CSV file.

    Returns:
        Expected JSON object for strict evaluation.
    """
    expected_rows = _read_ground_truth_rows(output_path)
    return {
        "document_info": {
            "source_type": _source_type(input_path),
            "filename": input_path.name,
        },
        "summary": {
            "total_sheets": 1,
            "total_rows": len(expected_rows),
            "total_columns": len(REPORT_HEADERS),
        },
        "content_data": [
            {
                "table_name": "Sheet1",
                "headers": REPORT_HEADERS,
                "rows": expected_rows,
            }
        ],
    }


def _source_type(path: Path) -> str:
    """Map a source path to prompt source-type labels.

    Args:
        path: Source file path.

    Returns:
        ``PDF`` for ``.pdf`` and ``.png`` inputs, otherwise ``Excel``.
    """
    if path.suffix.lower() in {".pdf", ".png"}:
        return "PDF"

    return "Excel"


def _build_prompt(input_path: Path, source_text: str) -> str:
    """Build the final LLM prompt for one benchmark input.

    Args:
        input_path: Source file path.
        source_text: Text extracted from the source file.

    Returns:
        Prompt sent to the LLM.
    """
    return f"""
{SYSTEM_PROMPT}

SOURCE FILENAME:
{input_path.name}

SOURCE:
{source_text}
""".strip()


def _write_runtime_dataset(
    directory: Path,
    filename: str,
    prompt: str,
    expected_output: dict[str, Any],
) -> Path:
    """Write a one-row runtime dataset used by ``BenchmarkRunner``.

    Args:
        directory: Controlled runtime dataset directory.
        filename: Runtime dataset filename.
        prompt: Prompt generated from the source input.
        expected_output: Expected output dictionary used for evaluation.

    Returns:
        Written runtime dataset path.
    """
    path = _safe_output_path(directory, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": [
            {
                "prompt": prompt,
                "expected_output": expected_output,
            }
        ]
    }
    with path.open("w", encoding=JSON_ENCODING) as dataset_file:
        json.dump(payload, dataset_file, ensure_ascii=False, indent=2)
    return path


def _write_overall_report(
    batch_results: list[dict[str, Any]],
    report_dir: Path,
    *,
    preserve_existing: bool = False,
) -> Path:
    """Write the overall benchmark comparison report.

    The report contains per-file winners, per-model averages, and overall
    winner rows for best average score and fastest average runtime.

    Args:
        batch_results: Batch result dictionaries.
        report_dir: Destination report folder.

    Returns:
        Path to the written aggregate CSV report.
    """
    if preserve_existing:
        report_path = _next_available_output_path(
            report_dir,
            OVERALL_REPORT_FILENAME,
        )
    else:
        report_path = _safe_output_path(report_dir, OVERALL_REPORT_FILENAME)
    rows = [
        *_case_comparison_rows(batch_results),
        *_model_summary_rows(batch_results),
        *_overall_winner_rows(batch_results),
    ]

    with report_path.open("w", newline="", encoding=CSV_ENCODING) as file:
        writer = csv.DictWriter(file, fieldnames=OVERALL_REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return report_path


def _write_overall_report_with_merge(
    batch_results: list[dict[str, Any]],
    report_dir: Path,
    base_overall_report_path: Path,
    *,
    preserve_existing: bool = False,
) -> Path:
    """Write an overall report merged with a previous aggregate report.

    Args:
        batch_results: Fresh batch result dictionaries.
        report_dir: Destination report folder.
        base_overall_report_path: Existing overall report to merge with.

    Returns:
        Path to the merged overall report.
    """
    if preserve_existing:
        report_path = _next_available_output_path(
            report_dir,
            OVERALL_REPORT_FILENAME,
        )
    else:
        report_path = _safe_output_path(report_dir, OVERALL_REPORT_FILENAME)
    if not base_overall_report_path.exists():
        print(
            "warning: merge base overall report was not found, "
            "writing a fresh aggregate report."
        )
        return _write_overall_report(
            batch_results,
            report_dir,
            preserve_existing=preserve_existing,
        )

    base_rows = _read_overall_report_rows(base_overall_report_path)
    rows = _merge_overall_rows(base_rows, batch_results)

    with report_path.open("w", newline="", encoding=CSV_ENCODING) as file:
        writer = csv.DictWriter(file, fieldnames=OVERALL_REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return report_path


def _read_overall_report_rows(path: Path) -> list[dict[str, str]]:
    """Read an existing overall benchmark report."""
    safe_path = _ensure_project_child(path, "overall report path")
    with safe_path.open("r", newline="", encoding=CSV_ENCODING) as file:
        return list(csv.DictReader(file))


def _default_benchmark_report_sources() -> tuple[str, ...]:
    """Return default benchmark report sources to include.

    The default is intentionally narrow: only ``synthetic_examples``.
    """
    return DEFAULT_BENCHMARK_REPORT_SOURCES


def _read_benchmark_report_rows(path: Path) -> list[dict[str, Any]]:
    """Read benchmark CSV rows and parse ``llm_output`` safely."""
    safe_path = _ensure_project_child(path, "benchmark report path")
    rows: list[dict[str, Any]] = []
    with safe_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            llm_output = (row.get("llm_output") or "").strip()
            parsed_llm_output: dict[str, Any] = {}
            if llm_output:
                try:
                    parsed = json.loads(llm_output)
                    if isinstance(parsed, dict):
                        parsed_llm_output = parsed
                except json.JSONDecodeError:
                    parsed_llm_output = {}

            rows.append(
                {
                    **row,
                    "row_index": int(row.get("row_index", 0) or 0),
                    "parsed_llm_output": parsed_llm_output,
                }
            )

    return rows


def _collect_benchmark_report_rows(
    report_dir: Path,
    *,
    included_sources: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Collect filtered benchmark rows from ``benchmark_reports`` CSV files."""
    selected_sources = included_sources or _default_benchmark_report_sources()
    selected_set = set(selected_sources)
    collected_rows: list[dict[str, Any]] = []

    for path in sorted(report_dir.glob("*.csv")):
        name_parts = _model_and_source_from_report_filename(path)
        if name_parts is None:
            continue
        model_name, source_name = name_parts
        if source_name not in selected_set:
            continue

        for row in _read_benchmark_report_rows(path):
            if not _is_completed_row_with_output(row):
                continue
            collected_rows.append(
                {
                    **row,
                    "source_name": source_name,
                    "model": model_name,
                }
            )

    return collected_rows


def _average_scores_by_model_for_source(
    report_dir: Path,
    source_name: str,
) -> dict[str, float]:
    """Calculate per-model average scores for one benchmark source."""
    rows = _collect_benchmark_report_rows(
        report_dir,
        included_sources=(source_name,),
    )
    by_model: dict[str, list[float]] = {}
    for row in rows:
        model_name = str(row.get("model", ""))
        if not model_name:
            continue
        by_model.setdefault(model_name, []).append(float(row.get("score", 0.0)))

    return {
        model_name: _average_float(scores)
        for model_name, scores in by_model.items()
    }


def _append_source_column_to_text_report(
    existing_txt_path: Path,
    *,
    report_dir: Path,
    source_name: str,
) -> Path:
    """Create an additive TXT report variant with one extra source column.

    This function never overwrites the original TXT report. It always writes to
    a derived filename based on the original report stem with a fixed suffix
    (for example ``overall_benchmark_report_source_augmented.txt``) so
    historical reports stay intact.
    """
    project_root_real = os.path.realpath(str(PROJECT_ROOT.resolve()))
    existing_path_real = os.path.realpath(str(existing_txt_path))
    try:
        inside_project = (
            os.path.commonpath([project_root_real, existing_path_real])
            == project_root_real
        )
    except ValueError:
        inside_project = False
    if not inside_project:
        raise ValueError(
            "existing text report path must stay inside the project directory: "
            f"{existing_path_real}"
        )
    safe_existing_txt_path = Path(existing_path_real)
    if not safe_existing_txt_path.exists():
        raise FileNotFoundError(safe_existing_txt_path)

    existing_content = safe_existing_txt_path.read_text(encoding="utf-8")
    lines = existing_content.splitlines()
    lines = _with_source_column_added(
        lines=lines,
        report_dir=report_dir,
        source_name=source_name,
    )
    return _write_derived_text_report(
        report_dir=safe_existing_txt_path.parent,
        content="\n".join(lines) + "\n",
    )


def _derived_text_report_path(
    *,
    report_dir: Path,
) -> Path:
    """Build a validated derived TXT path for additive report output."""
    project_root_real = os.path.realpath(str(PROJECT_ROOT.resolve()))
    base_dir_real = os.path.realpath(str(report_dir))
    try:
        base_inside_project = (
            os.path.commonpath([project_root_real, base_dir_real]) == project_root_real
        )
    except ValueError:
        base_inside_project = False
    if not base_inside_project:
        raise ValueError(
            f"output directory must stay inside the project directory: "
            f"{base_dir_real}"
        )
    derived_path_real = os.path.realpath(
        str(Path(base_dir_real) / DERIVED_TEXT_REPORT_FILENAME)
    )
    try:
        derived_inside_base = (
            os.path.commonpath([base_dir_real, derived_path_real]) == base_dir_real
        )
    except ValueError:
        derived_inside_base = False
    if not derived_inside_base:
        raise ValueError("Derived text report path escapes the output directory.")
    return Path(derived_path_real)


def _write_derived_text_report(
    *,
    report_dir: Path,
    content: str,
) -> Path:
    """Write derived additive TXT output to a fixed safe report path."""
    project_root_real = os.path.realpath(str(PROJECT_ROOT.resolve()))
    base_dir_real = os.path.realpath(str(report_dir))
    try:
        base_inside_project = (
            os.path.commonpath([project_root_real, base_dir_real]) == project_root_real
        )
    except ValueError:
        base_inside_project = False
    if not base_inside_project:
        raise ValueError(
            "output directory must stay inside the project directory: "
            f"{base_dir_real}"
        )

    derived_path_real = os.path.realpath(
        str(Path(base_dir_real) / DERIVED_TEXT_REPORT_FILENAME)
    )
    try:
        derived_inside_base = (
            os.path.commonpath([base_dir_real, derived_path_real]) == base_dir_real
        )
    except ValueError:
        derived_inside_base = False
    if not derived_inside_base:
        raise ValueError("Derived text report path escapes the output directory.")

    with Path(derived_path_real).open("w", encoding="utf-8") as report_file:
        report_file.write(content)
    return Path(derived_path_real)


def _with_source_column_added(
    *,
    lines: list[str],
    report_dir: Path,
    source_name: str,
) -> list[str]:
    """Add a source-specific column to report text lines."""
    source_model_scores = _average_scores_by_model_for_source(report_dir, source_name)

    if not lines:
        return [f"Model | {source_name}"]
    if not _has_full_report_sections(lines):
        return _append_source_column_to_simple_lines(
            lines,
            source_name,
            source_model_scores=source_model_scores,
        )

    return _append_source_column_to_full_report(
        lines=lines,
        source_name=source_name,
        source_model_scores=source_model_scores,
    )


def _append_source_column_to_simple_lines(
    lines: list[str],
    source_name: str,
    *,
    source_model_scores: dict[str, float],
) -> list[str]:
    """Fallback strategy for legacy/plain one-table text output."""
    updated_lines = list(lines)
    header = updated_lines[0]
    if source_name not in header:
        updated_lines[0] = f"{header} | {source_name}"
    for index in range(1, len(updated_lines)):
        if "|" in updated_lines[index] and updated_lines[index].strip():
            row_cells = _table_cells(updated_lines[index])
            model_name = row_cells[0] if row_cells else updated_lines[index].split("|")[0].strip()
            score = source_model_scores.get(model_name)
            score_value = f"{score:.4f}" if score is not None else ""
            updated_lines[index] = f"{updated_lines[index]} | {score_value}"
    return updated_lines


def _append_source_column_to_full_report(
    *,
    lines: list[str],
    source_name: str,
    source_model_scores: dict[str, float],
) -> list[str]:
    """Add source column for sectioned benchmark reports."""
    updated_lines = _add_column_to_section_table(
        lines,
        section_title=MODEL_SUMMARY_TITLE,
        column_name=source_name,
        value_by_row=lambda cells: _model_summary_source_score(
            row_cells=cells,
            source_model_scores=source_model_scores,
        ),
    )
    return _add_column_to_section_table(
        updated_lines,
        section_title=OVERALL_WINNERS_TITLE,
        column_name=source_name,
        value_by_row=lambda cells: _overall_winner_source_score(
            row_cells=cells,
            source_model_scores=source_model_scores,
        ),
    )


def _model_summary_source_score(
    *,
    row_cells: list[str],
    source_model_scores: dict[str, float],
) -> str:
    """Compute source-column value for one Model Summary row."""
    if not row_cells:
        return ""
    model_name = row_cells[0]
    score = source_model_scores.get(model_name)
    if score is None:
        return ""
    return f"{score:.4f}"


def _overall_winner_source_score(
    *,
    row_cells: list[str],
    source_model_scores: dict[str, float],
) -> str:
    """Compute source-column value for one Overall Winners row."""
    if not row_cells:
        return ""
    category = row_cells[0]
    if category == OVERALL_WINNER_FASTEST_LABEL:
        return ""
    if len(row_cells) <= 1:
        return ""

    winner_model = row_cells[1]
    winner_score = source_model_scores.get(winner_model)
    if winner_score is None:
        return ""
    return f"{winner_score:.4f}"


def _add_column_to_section_table(
    lines: list[str],
    *,
    section_title: str,
    column_name: str,
    value_by_row: Callable[[list[str]], str],
) -> list[str]:
    """Add one column to a section table and re-render it for clean alignment."""
    section_index = _find_line_index(lines, section_title)
    if section_index is None:
        return lines

    table_start = _find_table_border_line(lines, section_index + 1)
    if table_start is None or table_start + 2 >= len(lines):
        return lines

    header_cells = _table_cells(lines[table_start + 1])
    if not header_cells:
        return lines

    headers = list(header_cells)
    if column_name in headers:
        column_index = headers.index(column_name)
    else:
        headers.append(column_name)
        column_index = len(headers) - 1

    row_start = table_start + 3
    table_end = _find_table_border_line(lines, row_start)
    if table_end is None:
        return lines

    rows: list[list[str]] = []
    for raw_line in lines[row_start:table_end]:
        cells = _table_cells(raw_line)
        if not cells:
            continue
        row_cells = list(cells)
        while len(row_cells) < len(headers):
            row_cells.append("")
        row_cells[column_index] = str(value_by_row(cells))
        rows.append(row_cells[: len(headers)])

    rendered = _format_table(headers, rows).splitlines()
    return [*lines[:table_start], *rendered, *lines[table_end + 1 :]]


def _find_line_index(lines: list[str], target: str) -> int | None:
    for index, line in enumerate(lines):
        if line.strip() == target:
            return index
    return None


def _find_table_border_line(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip().startswith("+-"):
            return index
    return None


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _model_and_source_from_report_filename(
    path: Path,
) -> tuple[str, str] | None:
    """Extract model and source name from benchmark-report CSV filename."""
    parts = path.stem.split("__")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _is_completed_row_with_output(row: dict[str, Any]) -> bool:
    """Check whether one row should be included in source-score aggregation."""
    return row.get("status") == REPORT_COMPLETED_STATUS and bool(
        str(row.get("llm_output", "")).strip()
    )


def _has_full_report_sections(lines: list[str]) -> bool:
    """Return whether the text report contains both key table sections."""
    return all(
        any(section in line for line in lines)
        for section in REPORT_SECTIONS_WITH_TABLES
    )


def _merge_overall_rows(
    base_rows: list[dict[str, str]],
    batch_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge previous overall rows with fresh batch results."""
    base_case_rows = {
        row["case_id"]: row
        for row in base_rows
        if row.get("section") == "case_comparison" and row.get("case_id")
    }
    base_model_rows = {
        row["model"]: row
        for row in base_rows
        if row.get("section") == "model_summary" and row.get("model")
    }

    new_case_rows = {
        row["case_id"]: row for row in _case_comparison_rows(batch_results)
    }
    merged_case_rows = {**base_case_rows, **new_case_rows}

    new_model_rows = {
        row["model"]: row for row in _model_summary_rows(batch_results)
    }
    merged_model_rows = _merge_model_summary_rows(base_model_rows, new_model_rows)

    model_rows_list = [
        _normalized_model_summary_row(model_name, row)
        for model_name, row in sorted(merged_model_rows.items())
    ]
    winner_rows = _winner_rows_from_model_summaries(model_rows_list)

    combined_rows: list[dict[str, Any]] = []
    for case_id, row in sorted(merged_case_rows.items()):
        combined_rows.append(_normalized_case_comparison_row(case_id, row))
    combined_rows.extend(model_rows_list)
    combined_rows.extend(winner_rows)
    return combined_rows


def _merge_model_summary_rows(
    base_rows: dict[str, dict[str, Any]],
    new_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge model summary rows using weighted averages by run count."""
    merged: dict[str, dict[str, Any]] = {}
    all_models = set(base_rows) | set(new_rows)
    for model_name in all_models:
        base_row = base_rows.get(model_name)
        new_row = new_rows.get(model_name)
        if base_row is None:
            merged[model_name] = dict(new_row or {})
            continue
        if new_row is None:
            merged[model_name] = dict(base_row)
            continue

        base_total_runs = int(base_row["total_runs"])
        new_total_runs = int(new_row["total_runs"])
        merged_total_runs = base_total_runs + new_total_runs
        if merged_total_runs == 0:
            merged_average_score = 0.0
            merged_average_seconds = 0.0
        else:
            merged_average_score = (
                (float(base_row["average_score"]) * base_total_runs)
                + (float(new_row["average_score"]) * new_total_runs)
            ) / merged_total_runs
            merged_average_seconds = (
                (float(base_row["average_seconds"]) * base_total_runs)
                + (float(new_row["average_seconds"]) * new_total_runs)
            ) / merged_total_runs

        merged[model_name] = _model_summary_row(
            model_name=model_name,
            average_score=merged_average_score,
            average_seconds=merged_average_seconds,
            total_runs=merged_total_runs,
            completed_runs=int(base_row["completed_runs"])
            + int(new_row["completed_runs"]),
        )

    return merged


def _normalized_case_comparison_row(
    case_id: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a case-comparison row to the overall report schema."""
    return _case_comparison_row(
        case_id=case_id,
        best_model=str(row.get("best_model", "")),
        best_score=float(row.get("best_score", 0.0)),
        fastest_model=str(row.get("fastest_model", "")),
        fastest_seconds=float(row.get("fastest_seconds", 0.0)),
        total_runs=int(row.get("total_runs", 0)),
        completed_runs=int(row.get("completed_runs", 0)),
    )


def _normalized_model_summary_row(
    model_name: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a model-summary row to the overall report schema."""
    return _model_summary_row(
        model_name=model_name,
        average_score=float(row.get("average_score", 0.0)),
        average_seconds=float(row.get("average_seconds", 0.0)),
        total_runs=int(row.get("total_runs", 0)),
        completed_runs=int(row.get("completed_runs", 0)),
    )


def _winner_rows_from_model_summaries(
    model_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build overall winner rows from model-summary rows."""
    if not model_rows:
        return []

    best_average_row = max(
        model_rows,
        key=lambda row: (
            float(row["average_score"]),
            -float(row["average_seconds"]),
        ),
    )
    fastest_average_row = min(
        model_rows,
        key=lambda row: (
            float(row["average_seconds"]),
            -float(row["average_score"]),
        ),
    )
    return _overall_winner_rows_from_best_and_fastest(
        best_average_row,
        fastest_average_row,
    )


def _write_overall_text_report(
    batch_results: list[dict[str, Any]],
    report_dir: Path,
    *,
    overall_report_path: Path | None = None,
    preserve_existing: bool = False,
) -> Path:
    """Write a human-readable text report with summary tables.

    Args:
        batch_results: Batch result dictionaries.
        report_dir: Destination report folder.

    Returns:
        Path to the written text report.
    """
    if preserve_existing:
        report_path = _next_available_output_path(
            report_dir,
            OVERALL_TEXT_REPORT_FILENAME,
        )
    else:
        report_path = _safe_output_path(report_dir, OVERALL_TEXT_REPORT_FILENAME)
    lines = [
        "LLM Benchmark Report",
        "====================",
        "",
        "Run Results",
        "-----------",
    ]

    if batch_results:
        run_rows = [
            [
                str(result["model"]),
                str(result["case_id"]),
                str(result["status"]),
                f"{float(result['average_score']):.4f}",
                f"{float(result['elapsed_seconds']):.4f}",
                f"{int(result['successful_evaluations'])}/"
                f"{int(result['total_rows'])}",
            ]
            for result in batch_results
        ]
        lines.append(
            _format_table(
                ["Model", "Case", "Status", "Score", "Seconds", "Success"],
                run_rows,
            )
        )
    else:
        lines.append("No benchmark runs were executed.")

    overall_rows: list[dict[str, Any]] | None = None
    if overall_report_path is not None and overall_report_path.exists():
        overall_rows = _read_overall_report_rows(overall_report_path)

    case_rows = (
        _case_rows_from_overall_rows(overall_rows)
        if overall_rows is not None
        else _case_comparison_rows(batch_results)
    )
    lines.extend(["", "Case Comparison", "---------------"])
    if case_rows:
        lines.append(
            _format_table(
                [
                    "Case",
                    "Best Model",
                    "Best Score",
                    "Fastest Model",
                    "Fastest Seconds",
                    "Completed/Total",
                ],
                [
                    [
                        str(row.get("case_id", "")),
                        str(row.get("best_model", "")),
                        f"{float(row.get('best_score', 0.0)):.4f}",
                        str(row.get("fastest_model", "")),
                        f"{float(row.get('fastest_seconds', 0.0)):.4f}",
                        f"{int(row.get('completed_runs', 0))}/"
                        f"{int(row.get('total_runs', 0))}",
                    ]
                    for row in case_rows
                ],
            )
        )
    else:
        lines.append("No case comparison data.")

    model_rows = (
        _model_rows_from_overall_rows(overall_rows)
        if overall_rows is not None
        else _model_summary_rows(batch_results)
    )
    lines.extend(["", "Model Summary", "-------------"])
    if model_rows:
        lines.append(
            _format_table(
                [
                    "Model",
                    "Average Score",
                    "Average Seconds",
                    "Completed/Total",
                ],
                [
                    [
                        str(row.get("model", "")),
                        f"{float(row.get('average_score', 0.0)):.4f}",
                        f"{float(row.get('average_seconds', 0.0)):.4f}",
                        f"{int(row.get('completed_runs', 0))}/"
                        f"{int(row.get('total_runs', 0))}",
                    ]
                    for row in model_rows
                ],
            )
        )
    else:
        lines.append("No model summary data.")

    winner_rows = (
        _winner_rows_from_overall_rows(overall_rows)
        if overall_rows is not None
        else _overall_winner_rows(batch_results)
    )
    lines.extend(["", OVERALL_WINNERS_TITLE, "---------------"])
    if winner_rows:
        best_average_row, fastest_row = winner_rows
        lines.append(
            _format_table(
                ["Category", "Model", "Average Score", "Average Seconds"],
                [
                    [
                        "Best average score",
                        str(best_average_row.get("model", "")),
                        f"{float(best_average_row.get('average_score', 0.0)):.4f}",
                        f"{float(best_average_row.get('average_seconds', 0.0)):.4f}",
                    ],
                    [
                        "Fastest average runtime",
                        str(fastest_row.get("model", "")),
                        f"{float(fastest_row.get('average_score', 0.0)):.4f}",
                        f"{float(fastest_row.get('average_seconds', 0.0)):.4f}",
                    ],
                ],
            )
        )
    else:
        lines.append("No overall winner data.")

    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path


def _case_rows_from_overall_rows(
    overall_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Extract case-comparison rows from an overall report payload."""
    if not overall_rows:
        return []
    return [
        row
        for row in overall_rows
        if row.get("section") == "case_comparison"
    ]


def _model_rows_from_overall_rows(
    overall_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Extract model-summary rows from an overall report payload."""
    if not overall_rows:
        return []
    return [
        row
        for row in overall_rows
        if row.get("section") == "model_summary"
    ]


def _winner_rows_from_overall_rows(
    overall_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Extract winner rows from an overall report payload."""
    if not overall_rows:
        return []

    by_section = {
        row.get("section", ""): row
        for row in overall_rows
        if row.get("section") in {"overall_best_average", "overall_fastest"}
    }
    best_average_row = by_section.get("overall_best_average")
    fastest_row = by_section.get("overall_fastest")
    if best_average_row is None or fastest_row is None:
        return []
    return [best_average_row, fastest_row]


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an ASCII table for text reports.

    Args:
        headers: Header row labels.
        rows: Body rows as string cells.

    Returns:
        Formatted table text.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def _format_row(cells: list[str]) -> str:
        padded = [
            value.ljust(widths[index])
            for index, value in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    table_lines = [separator, _format_row(headers), separator]
    table_lines.extend(_format_row(row) for row in rows)
    table_lines.append(separator)
    return "\n".join(table_lines)


def _case_comparison_rows(
    batch_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build aggregate rows showing best and fastest model per file.

    Args:
        batch_results: Batch result dictionaries.

    Returns:
        Overall report rows with ``section`` set to ``case_comparison``.
    """
    rows: list[dict[str, Any]] = []
    for case_id, case_results in _group_results(batch_results, "case_id").items():
        best_result = _best_score_result(case_results)
        fastest_result = _fastest_result(case_results)
        rows.append(
            _case_comparison_row(
                case_id=case_id,
                best_model=str(best_result["model"]),
                best_score=float(best_result["average_score"]),
                fastest_model=str(fastest_result["model"]),
                fastest_seconds=float(fastest_result["elapsed_seconds"]),
                total_runs=len(case_results),
                completed_runs=_completed_run_count(case_results),
            )
        )

    return rows


def _model_summary_rows(
    batch_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build aggregate rows showing average score and runtime per model.

    Args:
        batch_results: Batch result dictionaries.

    Returns:
        Overall report rows with ``section`` set to ``model_summary``.
    """
    rows: list[dict[str, Any]] = []
    for model_name, model_results in _group_results(batch_results, "model").items():
        rows.append(
            _model_summary_row(
                model_name=model_name,
                average_score=_average_float(
                    result["average_score"] for result in model_results
                ),
                average_seconds=_average_float(
                    result["elapsed_seconds"] for result in model_results
                ),
                total_runs=len(model_results),
                completed_runs=_completed_run_count(model_results),
            )
        )

    return rows


def _overall_winner_rows(
    batch_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build rows for the overall best-score and fastest-runtime models.

    Args:
        batch_results: Batch result dictionaries.

    Returns:
        Overall winner rows.
    """
    model_rows = _model_summary_rows(batch_results)
    if not model_rows:
        return []

    best_average_row = max(
        model_rows,
        key=lambda row: (
            float(row["average_score"]),
            -float(row["average_seconds"]),
        ),
    )
    fastest_average_row = min(
        model_rows,
        key=lambda row: (
            float(row["average_seconds"]),
            -float(row["average_score"]),
        ),
    )
    return _overall_winner_rows_from_best_and_fastest(
        best_average_row,
        fastest_average_row,
    )


def _case_comparison_row(
    *,
    case_id: str,
    best_model: str,
    best_score: float,
    fastest_model: str,
    fastest_seconds: float,
    total_runs: int,
    completed_runs: int,
) -> dict[str, Any]:
    """Build one case-comparison row in overall report schema."""
    return {
        "section": "case_comparison",
        "case_id": case_id,
        "model": "",
        "best_model": best_model,
        "best_score": best_score,
        "fastest_model": fastest_model,
        "fastest_seconds": fastest_seconds,
        "average_score": "",
        "average_seconds": "",
        "total_runs": total_runs,
        "completed_runs": completed_runs,
    }


def _model_summary_row(
    *,
    model_name: str,
    average_score: float,
    average_seconds: float,
    total_runs: int,
    completed_runs: int,
) -> dict[str, Any]:
    """Build one model-summary row in overall report schema."""
    return {
        "section": "model_summary",
        "case_id": "",
        "model": model_name,
        "best_model": "",
        "best_score": "",
        "fastest_model": "",
        "fastest_seconds": "",
        "average_score": average_score,
        "average_seconds": average_seconds,
        "total_runs": total_runs,
        "completed_runs": completed_runs,
    }


def _overall_winner_rows_from_best_and_fastest(
    best_average_row: dict[str, Any],
    fastest_average_row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build overall winner rows from best-score and fastest rows."""
    return [
        {
            "section": "overall_best_average",
            "case_id": "",
            "model": best_average_row["model"],
            "best_model": "",
            "best_score": "",
            "fastest_model": "",
            "fastest_seconds": "",
            "average_score": best_average_row["average_score"],
            "average_seconds": best_average_row["average_seconds"],
            "total_runs": best_average_row["total_runs"],
            "completed_runs": best_average_row["completed_runs"],
        },
        {
            "section": "overall_fastest",
            "case_id": "",
            "model": fastest_average_row["model"],
            "best_model": "",
            "best_score": "",
            "fastest_model": "",
            "fastest_seconds": "",
            "average_score": fastest_average_row["average_score"],
            "average_seconds": fastest_average_row["average_seconds"],
            "total_runs": fastest_average_row["total_runs"],
            "completed_runs": fastest_average_row["completed_runs"],
        },
    ]


def _group_results(
    batch_results: list[dict[str, Any]],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    """Group batch result dictionaries by a string key.

    Args:
        batch_results: Batch result dictionaries.
        key: Result dictionary key to group by.

    Returns:
        Grouped batch results.
    """
    grouped_results: dict[str, list[dict[str, Any]]] = {}
    for result in batch_results:
        grouped_results.setdefault(str(result[key]), []).append(result)

    return grouped_results


def _best_score_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the best-scoring result, using runtime as tie-breaker.

    Args:
        results: Batch results for one case.

    Returns:
        Best result dictionary.
    """
    return max(
        results,
        key=lambda result: (
            float(result["average_score"]),
            -float(result["elapsed_seconds"]),
        ),
    )


def _fastest_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the fastest result, using score as tie-breaker.

    Args:
        results: Batch results for one case.

    Returns:
        Fastest result dictionary.
    """
    return min(
        results,
        key=lambda result: (
            float(result["elapsed_seconds"]),
            -float(result["average_score"]),
        ),
    )


def _completed_run_count(results: list[dict[str, Any]]) -> int:
    """Count completed runs in a result collection.

    Args:
        results: Batch result dictionaries.

    Returns:
        Number of completed runs.
    """
    return sum(1 for result in results if result["status"] == "completed")


def _average_float(values: Iterable[Any]) -> float:
    """Calculate a safe floating-point average.

    Args:
        values: Iterable numeric values.

    Returns:
        Average value, or ``0.0`` when empty.
    """
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return 0.0

    return sum(numeric_values) / len(numeric_values)


def _print_batch_summary(
    batch_results: list[dict[str, Any]],
    overall_report_path: Path,
    overall_text_report_path: Path,
    category_accuracy_paths: dict[str, Path] | None = None,
) -> None:
    """Print a readable summary table for batch results.

    Args:
        batch_results: Batch result dictionaries.
        overall_report_path: Aggregate report path.
    """
    print("")
    print("Batch Summary")
    print("-------------")
    if not batch_results:
        print("No supported benchmark cases were found.")
        return

    for result in batch_results:
        print(
            f"{result['model']} | {result['case_id']} | "
            f"status={result['status']} | "
            f"score={result['average_score']:.4f} | "
            f"success={result['successful_evaluations']}/"
            f"{result['total_rows']}"
        )

    winner_rows = _overall_winner_rows(batch_results)
    if winner_rows:
        best_average_row, fastest_row = winner_rows
        print("")
        print(OVERALL_WINNERS_TITLE)
        print("---------------")
        print(
            "Best average score: "
            f"{best_average_row['model']} "
            f"({float(best_average_row['average_score']):.4f})"
        )
        print(
            "Fastest average runtime: "
            f"{fastest_row['model']} "
            f"({float(fastest_row['average_seconds']):.4f}s)"
        )

    print("")
    print(f"Overall report: {overall_report_path}")
    print(f"Text report   : {overall_text_report_path}")
    if category_accuracy_paths:
        print(f"Category JSON : {category_accuracy_paths['json_path']}")
        print(f"Category TXT  : {category_accuracy_paths['txt_path']}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
