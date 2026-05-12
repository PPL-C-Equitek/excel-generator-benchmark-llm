"""Category-level accuracy aggregation for LLM evaluation outputs."""

from __future__ import annotations

import json
from typing import Any, TypedDict


class CategorySummary(TypedDict):
    """Per-scope aggregation payload for category accuracy reporting."""

    ground_truth_count: int
    exact_match_count: int
    partial_score_sum: float
    error_count: int
    exact_accuracy_percent: float
    partial_accuracy_percent: float


class CategoryAccuracyReport(TypedDict):
    """Top-level category accuracy report payload."""

    overall: CategorySummary
    by_category: dict[str, CategorySummary]


def calculate_category_accuracy_report(
    evaluations: list[dict[str, Any]],
) -> CategoryAccuracyReport:
    """Aggregate exact and partial accuracy percentages by category.

    Args:
        evaluations: Evaluation rows containing ``category``, ``ground_truth``,
            and ``llm_output`` fields.

    Returns:
        JSON-serializable dictionary with ``overall`` and ``by_category`` keys.
    """
    by_category: dict[str, CategorySummary] = {}
    overall = _new_summary()

    for evaluation in evaluations:
        category = _normalize_category(evaluation.get("category"))
        ground_truth = _normalize_ground_truth(evaluation.get("ground_truth"))
        parsed_output, parse_error = _parse_json_object(evaluation.get("llm_output", ""))

        category_summary = by_category.setdefault(category, _new_summary())
        _accumulate_evaluation(
            summary=category_summary,
            parsed_output=parsed_output,
            ground_truth=ground_truth,
            parse_error=parse_error,
        )
        _accumulate_evaluation(
            summary=overall,
            parsed_output=parsed_output,
            ground_truth=ground_truth,
            parse_error=parse_error,
        )

    _finalize_summary(overall)
    for summary in by_category.values():
        _finalize_summary(summary)

    return {
        "overall": overall,
        "by_category": by_category,
    }


def _normalize_category(value: Any) -> str:
    """Normalize category names, falling back to ``uncategorized``."""
    if value in {None, ""}:
        return "uncategorized"
    return str(value)


def _normalize_ground_truth(value: Any) -> dict[str, Any]:
    """Normalize ground truth payload to dictionary form."""
    if isinstance(value, dict):
        return value
    return {}


def _parse_json_object(value: Any) -> tuple[dict[str, Any], bool]:
    """Parse a JSON object from a raw value.

    Returns:
        Parsed object and a boolean indicating whether parsing failed.
    """
    if isinstance(value, dict):
        return value, False
    if not isinstance(value, str):
        return {}, True

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}, True

    if not isinstance(parsed, dict):
        return {}, True

    return parsed, False


def _accumulate_evaluation(
    *,
    summary: CategorySummary,
    parsed_output: dict[str, Any],
    ground_truth: dict[str, Any],
    parse_error: bool,
) -> None:
    """Accumulate one parsed evaluation into a mutable summary bucket."""
    if parse_error:
        summary["error_count"] += 1

    if not ground_truth:
        return

    summary["ground_truth_count"] += 1
    if parsed_output == ground_truth:
        summary["exact_match_count"] += 1
    summary["partial_score_sum"] += _partial_score(parsed_output, ground_truth)


def _partial_score(
    parsed_output: dict[str, Any],
    ground_truth: dict[str, Any],
) -> float:
    """Compute per-item partial score safely.

    Partial score is based on correct key-value matches across the ground truth
    schema. Extra LLM keys do not increase the score.
    """
    denominator = len(ground_truth)
    if denominator == 0:
        return 0.0

    matches = sum(
        1
        for key, expected in ground_truth.items()
        if parsed_output.get(key, object()) == expected
    )
    return matches / denominator


def _new_summary() -> CategorySummary:
    """Create a mutable summary accumulator."""
    return {
        "ground_truth_count": 0,
        "exact_match_count": 0,
        "partial_score_sum": 0.0,
        "error_count": 0,
        "exact_accuracy_percent": 0.0,
        "partial_accuracy_percent": 0.0,
    }


def _finalize_summary(summary: CategorySummary) -> None:
    """Finalize percentage fields with zero-division safety."""
    denominator = int(summary["ground_truth_count"])
    if denominator == 0:
        summary["exact_accuracy_percent"] = 0.0
        summary["partial_accuracy_percent"] = 0.0
        return

    summary["exact_accuracy_percent"] = (
        float(summary["exact_match_count"]) / denominator
    ) * 100.0
    summary["partial_accuracy_percent"] = (
        float(summary["partial_score_sum"]) / denominator
    ) * 100.0
