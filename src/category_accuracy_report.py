"""Build category-accuracy reports from existing benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.category_accuracy import calculate_category_accuracy_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Generate category-accuracy JSON and TXT reports from saved artifacts."""
    args = _parse_args()
    result = generate_category_accuracy_reports(
        report_dir=Path(args.report_dir),
        runtime_dir=Path(args.runtime_dir),
        output_dir=Path(args.output_dir),
        source_filter=args.source,
        model_filter=args.model,
    )
    print(f"Saved JSON: {result['json_path']}")
    print(f"Saved TXT : {result['txt_path']}")
    print(f"Total evaluations: {result['total_evaluations']}")
    return 0


def generate_category_accuracy_reports(
    *,
    report_dir: Path,
    runtime_dir: Path,
    output_dir: Path,
    source_filter: str = "",
    model_filter: str = "",
) -> dict[str, Any]:
    """Build category-accuracy artifacts from existing run outputs.

    Returns:
        Metadata with output paths and evaluation count.
    """
    safe_report_dir = _ensure_project_child(report_dir, "report directory")
    safe_runtime_dir = _ensure_project_child(
        runtime_dir,
        "runtime dataset directory",
    )
    safe_output_dir = _ensure_project_child(output_dir, "output directory")
    safe_output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = _collect_evaluations(
        report_dir=safe_report_dir,
        runtime_dir=safe_runtime_dir,
        source_filter=source_filter,
        model_filter=model_filter,
    )
    report = calculate_category_accuracy_report(evaluations)
    by_model = _group_summary(evaluations, "model")
    by_source = _group_summary(evaluations, "source")

    suffix = _suffix_from_filters(source_filter, model_filter)
    json_path = safe_output_dir / f"category_accuracy_report{suffix}.json"
    txt_path = safe_output_dir / f"category_accuracy_report{suffix}.txt"
    final_payload = {
        **report,
        "by_model": by_model,
        "by_source": by_source,
    }

    json_path.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(
        _format_text_report(
            final_payload,
            total_evaluations=len(evaluations),
        ),
        encoding="utf-8",
    )

    return {
        "json_path": json_path,
        "txt_path": txt_path,
        "total_evaluations": len(evaluations),
    }


def _ensure_project_child(path: Path, label: str) -> Path:
    """Validate that a path is inside ``PROJECT_ROOT``.

    Args:
        path: Candidate path (relative or absolute).
        label: Human-readable label for error messaging.

    Returns:
        Resolved path guaranteed to be within ``PROJECT_ROOT``.

    Raises:
        ValueError: If the resolved path escapes ``PROJECT_ROOT``.
    """
    project_root = PROJECT_ROOT.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(project_root):
        raise ValueError(
            f"{label} must stay inside the project directory: {resolved_path}"
        )
    return resolved_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate category accuracy from benchmark_reports and "
            "benchmark_runtime_datasets without rerunning LLM."
        )
    )
    parser.add_argument(
        "--report-dir",
        default="benchmark_reports",
        help="Directory containing per-run CSV reports.",
    )
    parser.add_argument(
        "--runtime-dir",
        default="benchmark_runtime_datasets",
        help="Directory containing per-run runtime dataset JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_reports",
        help="Directory where category accuracy JSON/TXT are saved.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Optional source filter, e.g. like-real_examples or single_case.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional model filter, e.g. gpt-5.2-codex.",
    )
    return parser.parse_args()


def _collect_evaluations(
    *,
    report_dir: Path,
    runtime_dir: Path,
    source_filter: str,
    model_filter: str,
) -> list[dict[str, Any]]:
    """Collect normalized evaluation rows from report/runtime artifacts."""
    evaluations: list[dict[str, Any]] = []
    source_filter = source_filter.strip()
    model_filter = model_filter.strip()

    for report_path in sorted(report_dir.glob("*.csv")):
        context = _resolve_report_context(
            report_path=report_path,
            source_filter=source_filter,
            model_filter=model_filter,
        )
        if context is None:
            continue

        model_name, source_name = context
        runtime_path = runtime_dir / f"{report_path.stem}.json"
        if not runtime_path.exists():
            continue

        ground_truth = _read_ground_truth(runtime_path)
        category = _extract_category(ground_truth)
        evaluations.extend(
            _completed_rows_to_evaluations(
                report_path=report_path,
                model_name=model_name,
                source_name=source_name,
                category=category,
                ground_truth=ground_truth,
            )
        )

    return evaluations


def _resolve_report_context(
    *,
    report_path: Path,
    source_filter: str,
    model_filter: str,
) -> tuple[str, str] | None:
    """Return ``(model_name, source_name)`` when report matches filters."""
    if report_path.name.startswith("overall_"):
        return None

    parsed_name = _parse_report_filename(report_path.stem)
    if parsed_name is None:
        return None
    model_name, source_name = parsed_name

    if source_filter and source_name != source_filter:
        return None
    if model_filter and model_name != model_filter:
        return None

    return model_name, source_name


def _completed_rows_to_evaluations(
    *,
    report_path: Path,
    model_name: str,
    source_name: str,
    category: str,
    ground_truth: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert completed CSV rows into evaluation payload dictionaries."""
    evaluations: list[dict[str, Any]] = []
    with report_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("status") != "completed":
                continue
            evaluations.append(
                {
                    "category": category,
                    "model": model_name,
                    "source": source_name,
                    "ground_truth": ground_truth,
                    "llm_output": row.get("llm_output", ""),
                }
            )
    return evaluations


def _parse_report_filename(stem: str) -> tuple[str, str] | None:
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _read_ground_truth(runtime_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {}

    first_row = rows[0]
    if not isinstance(first_row, dict):
        return {}

    expected_output = first_row.get("expected_output", {})
    if not isinstance(expected_output, dict):
        return {}
    return expected_output


def _extract_category(ground_truth: dict[str, Any]) -> str:
    """Extract category label from runtime expected output metadata.

    Priority:
    1. ``document_info.filename`` extension (for example ``png``, ``csv``, ``txt``).
    2. ``document_info.source_type`` legacy value (for example ``PDF``/``Excel``).
    3. ``unknown``.
    """
    document_info = ground_truth.get("document_info")
    if not isinstance(document_info, dict):
        return "unknown"

    raw_filename = document_info.get("filename", "")
    if isinstance(raw_filename, str) and raw_filename.strip():
        suffix = Path(raw_filename).suffix.lower().lstrip(".")
        if suffix:
            return suffix

    source_type = document_info.get("source_type", "unknown")
    return str(source_type)


def _suffix_from_filters(source: str, model: str) -> str:
    chunks: list[str] = []
    if source.strip():
        chunks.append(source.strip())
    if model.strip():
        chunks.append(model.strip())
    if not chunks:
        return ""
    safe = "_".join(_safe_label(chunk) for chunk in chunks)
    return f"_{safe}"


def _safe_label(value: str) -> str:
    output = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    output = output.strip("_")
    return output or "all"


def _group_summary(
    evaluations: list[dict[str, Any]],
    field_name: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        key = str(evaluation.get(field_name, "unknown"))
        grouped.setdefault(key, []).append(evaluation)

    result: dict[str, dict[str, Any]] = {}
    for key, group_evaluations in grouped.items():
        result[key] = calculate_category_accuracy_report(group_evaluations)[
            "overall"
        ]
    return result


def _format_text_report(report: dict[str, Any], *, total_evaluations: int) -> str:
    overall = report.get("overall", {})
    lines = [
        "Category Accuracy Report",
        "========================",
        "",
        f"Total evaluations: {total_evaluations}",
        "",
        "Overall",
        "-------",
        f"- Ground truths counted : {int(overall.get('ground_truth_count', 0))}",
        f"- Exact matches         : {int(overall.get('exact_match_count', 0))}",
        f"- Parse errors          : {int(overall.get('error_count', 0))}",
        f"- Exact accuracy        : {float(overall.get('exact_accuracy_percent', 0.0)):.2f}%",
        f"- Partial accuracy      : {float(overall.get('partial_accuracy_percent', 0.0)):.2f}%",
        "",
        "By Category",
        "-----------",
    ]

    by_category = report.get("by_category", {})
    if not isinstance(by_category, dict) or not by_category:
        lines.append("- No category data found.")
    else:
        for category_name in sorted(by_category):
            summary = by_category[category_name]
            lines.extend(
                [
                    f"{category_name}",
                    f"  ground_truth_count   : {int(summary.get('ground_truth_count', 0))}",
                    f"  exact_match_count    : {int(summary.get('exact_match_count', 0))}",
                    f"  error_count          : {int(summary.get('error_count', 0))}",
                    f"  exact_accuracy_pct   : {float(summary.get('exact_accuracy_percent', 0.0)):.2f}%",
                    f"  partial_accuracy_pct : {float(summary.get('partial_accuracy_percent', 0.0)):.2f}%",
                    "",
                ]
            )

    lines.extend(["By Model", "--------"])
    _append_summary_block(lines, report.get("by_model"), title_field="model")
    lines.extend(["", "By Source", "---------"])
    _append_summary_block(lines, report.get("by_source"), title_field="source")

    return "\n".join(lines).rstrip() + "\n"


def _append_summary_block(
    lines: list[str],
    raw_data: Any,
    *,
    title_field: str,
) -> None:
    if not isinstance(raw_data, dict) or not raw_data:
        lines.append(f"- No {title_field} data found.")
        return

    for name in sorted(raw_data):
        summary = raw_data[name]
        lines.extend(
            [
                f"{name}",
                f"  ground_truth_count   : {int(summary.get('ground_truth_count', 0))}",
                f"  exact_match_count    : {int(summary.get('exact_match_count', 0))}",
                f"  error_count          : {int(summary.get('error_count', 0))}",
                f"  exact_accuracy_pct   : {float(summary.get('exact_accuracy_percent', 0.0)):.2f}%",
                f"  partial_accuracy_pct : {float(summary.get('partial_accuracy_percent', 0.0)):.2f}%",
                "",
            ]
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
