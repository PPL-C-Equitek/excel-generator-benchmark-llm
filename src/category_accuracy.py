"""Category-level accuracy aggregation for LLM evaluation outputs."""

from __future__ import annotations

import json
from typing import Any


def calculate_category_accuracy_report(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact and partial accuracy percentages by category.

    Args:
        evaluations: Evaluation rows containing ``category``, ``ground_truth``,
            and ``llm_output`` fields.

    Returns:
        JSON-serializable dictionary with ``overall`` and ``by_category`` keys.
    """
    by_category: dict[str, dict[str, Any]] = {}
    overall = _new_summary()

    for evaluation in evaluations:
        category = str(evaluation.get("category", "uncategorized"))
        ground_truth = evaluation.get("ground_truth")
        llm_output = evaluation.get("llm_output", "")

        if not isinstance(ground_truth, dict):
            ground_truth = {}

        parsed_output, parse_error = _parse_json_object(llm_output)

        category_summary = by_category.setdefault(category, _new_summary())

        if parse_error:
            category_summary["error_count"] += 1
            overall["error_count"] += 1

        if not ground_truth:
            continue

        category_summary["ground_truth_count"] += 1
        overall["ground_truth_count"] += 1

        if parsed_output == ground_truth:
            category_summary["exact_match_count"] += 1
            overall["exact_match_count"] += 1

        partial_score = _partial_score(parsed_output, ground_truth)
        category_summary["partial_score_sum"] += partial_score
        overall["partial_score_sum"] += partial_score

    _finalize_summary(overall)
    for summary in by_category.values():
        _finalize_summary(summary)

    return {
        "overall": overall,
        "by_category": by_category,
    }


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


def _new_summary() -> dict[str, Any]:
    """Create a mutable summary accumulator."""
    return {
        "ground_truth_count": 0,
        "exact_match_count": 0,
        "partial_score_sum": 0.0,
        "error_count": 0,
        "exact_accuracy_percent": 0.0,
        "partial_accuracy_percent": 0.0,
    }


def _finalize_summary(summary: dict[str, Any]) -> None:
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
