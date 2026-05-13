import json
from typing import Any

import pytest

from src.category_accuracy import (
    _accumulate_evaluation,
    _finalize_summary,
    _new_summary,
    _normalize_category,
    _normalize_ground_truth,
    _parse_json_object,
    _partial_score,
    calculate_category_accuracy_report,
)


def _evaluation_case(
    *,
    category: Any,
    ground_truth: Any,
    llm_output: Any,
) -> dict[str, Any]:
    return {
        "category": category,
        "ground_truth": ground_truth,
        "llm_output": llm_output,
    }


def test_calculate_category_accuracy_groups_results_and_computes_80_percent():
    evaluations = [
        _evaluation_case(
            category="invoice",
            ground_truth={"invoice_id": "INV-001"},
            llm_output='{"invoice_id": "INV-001"}',
        ),
        _evaluation_case(
            category="invoice",
            ground_truth={"invoice_id": "INV-002"},
            llm_output='{"invoice_id": "INV-002"}',
        ),
        _evaluation_case(
            category="invoice",
            ground_truth={"invoice_id": "INV-003"},
            llm_output='{"invoice_id": "INV-003"}',
        ),
        _evaluation_case(
            category="invoice",
            ground_truth={"invoice_id": "INV-004"},
            llm_output='{"invoice_id": "INV-004"}',
        ),
        _evaluation_case(
            category="invoice",
            ground_truth={"invoice_id": "INV-005"},
            llm_output='{"invoice_id": "WRONG-ID"}',
        ),
        _evaluation_case(
            category="payroll",
            ground_truth={"employee_id": "EMP-001"},
            llm_output='{"employee_id": "EMP-001"}',
        ),
    ]

    report = calculate_category_accuracy_report(evaluations)

    assert isinstance(report, dict)
    assert "overall" in report
    assert "by_category" in report
    assert set(report["by_category"]) == {"invoice", "payroll"}

    invoice = report["by_category"]["invoice"]
    assert invoice["ground_truth_count"] == 5
    assert invoice["exact_accuracy_percent"] == pytest.approx(80.0)
    assert invoice["partial_accuracy_percent"] == pytest.approx(80.0)
    assert invoice["error_count"] == 0

    json.dumps(report)


def test_calculate_category_accuracy_handles_malformed_llm_json_without_crashing():
    evaluations = [
        _evaluation_case(
            category="invoice",
            ground_truth={"invoice_id": "INV-001"},
            llm_output="{this is malformed json}",
        )
    ]

    report = calculate_category_accuracy_report(evaluations)

    invoice = report["by_category"]["invoice"]
    assert invoice["ground_truth_count"] == 1
    assert invoice["exact_accuracy_percent"] == pytest.approx(0.0)
    assert invoice["partial_accuracy_percent"] == pytest.approx(0.0)
    assert invoice["error_count"] == 1


def test_calculate_category_accuracy_avoids_zero_division_when_ground_truth_count_is_zero():
    evaluations = [
        _evaluation_case(
            category="empty_category",
            ground_truth={},
            llm_output='{"hallucinated_field": "value"}',
        )
    ]

    report = calculate_category_accuracy_report(evaluations)

    category_summary = report["by_category"]["empty_category"]
    assert category_summary["ground_truth_count"] == 0
    assert category_summary["exact_accuracy_percent"] == pytest.approx(0.0)
    assert category_summary["partial_accuracy_percent"] == pytest.approx(0.0)
    assert category_summary["error_count"] == 0


def test_calculate_category_accuracy_keeps_partial_score_correct_with_hallucinated_extra_keys():
    evaluations = [
        _evaluation_case(
            category="procurement",
            ground_truth={
                "po_number": "PO-1001",
                "amount": 2500000,
            },
            llm_output=(
                '{"po_number": "PO-1001", "amount": 2500000, '
                '"currency": "IDR", "approver": "Finance Lead"}'
            ),
        )
    ]

    report = calculate_category_accuracy_report(evaluations)

    procurement = report["by_category"]["procurement"]
    assert procurement["ground_truth_count"] == 1
    assert procurement["exact_accuracy_percent"] == pytest.approx(0.0)
    assert procurement["partial_accuracy_percent"] == pytest.approx(100.0)


def test_calculate_category_accuracy_normalizes_missing_or_empty_category_name():
    evaluations = [
        _evaluation_case(
            category=None,
            ground_truth={"id": "A"},
            llm_output='{"id": "A"}',
        ),
        _evaluation_case(
            category="",
            ground_truth={"id": "B"},
            llm_output='{"id": "B"}',
        ),
    ]

    report = calculate_category_accuracy_report(evaluations)

    assert set(report["by_category"]) == {"uncategorized"}
    summary = report["by_category"]["uncategorized"]
    assert summary["ground_truth_count"] == 2
    assert summary["exact_match_count"] == 2


def test_calculate_category_accuracy_handles_non_dict_ground_truth_and_non_string_output():
    evaluations = [
        _evaluation_case(
            category="mixed",
            ground_truth=["not", "a", "dict"],
            llm_output={"already": "dict"},
        ),
        _evaluation_case(
            category="mixed",
            ground_truth={"id": 1},
            llm_output=["not", "json", "string"],
        ),
    ]

    report = calculate_category_accuracy_report(evaluations)

    mixed = report["by_category"]["mixed"]
    assert mixed["ground_truth_count"] == 1
    assert mixed["error_count"] == 1
    assert mixed["partial_accuracy_percent"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("raw_category", "expected"),
    [
        (None, "uncategorized"),
        ("", "uncategorized"),
        ("invoice", "invoice"),
        (123, "123"),
    ],
)
def test_normalize_category(raw_category, expected):
    assert _normalize_category(raw_category) == expected


@pytest.mark.parametrize(
    ("raw_ground_truth", "expected"),
    [
        ({"a": 1}, {"a": 1}),
        (None, {}),
        ("string", {}),
        (["list"], {}),
    ],
)
def test_normalize_ground_truth(raw_ground_truth, expected):
    assert _normalize_ground_truth(raw_ground_truth) == expected


def test_parse_json_object_accepts_dict_input_without_error():
    parsed, parse_error = _parse_json_object({"a": 1})
    assert parsed == {"a": 1}
    assert parse_error is False


def test_parse_json_object_flags_non_string_non_dict_as_error():
    parsed, parse_error = _parse_json_object(123)
    assert parsed == {}
    assert parse_error is True


def test_parse_json_object_flags_malformed_json_as_error():
    parsed, parse_error = _parse_json_object("{broken-json}")
    assert parsed == {}
    assert parse_error is True


def test_parse_json_object_flags_non_object_json_payload_as_error():
    parsed, parse_error = _parse_json_object('["a", "b"]')
    assert parsed == {}
    assert parse_error is True


def test_parse_json_object_parses_valid_json_object():
    parsed, parse_error = _parse_json_object('{"a": 1, "b": 2}')
    assert parsed == {"a": 1, "b": 2}
    assert parse_error is False


def test_partial_score_returns_zero_when_ground_truth_empty():
    assert _partial_score({"a": 1}, {}) == pytest.approx(0.0)


def test_partial_score_counts_only_matching_ground_truth_keys():
    score = _partial_score(
        {"a": 1, "b": 2, "extra": 999},
        {"a": 1, "b": 999},
    )
    assert score == pytest.approx(0.5)


def test_accumulate_evaluation_updates_error_only_when_ground_truth_empty():
    summary = _new_summary()
    _accumulate_evaluation(
        summary=summary,
        parsed_output={},
        ground_truth={},
        parse_error=True,
    )
    assert summary["error_count"] == 1
    assert summary["ground_truth_count"] == 0
    assert summary["partial_score_sum"] == pytest.approx(0.0)


def test_accumulate_evaluation_updates_match_and_partial_for_non_empty_ground_truth():
    summary = _new_summary()
    _accumulate_evaluation(
        summary=summary,
        parsed_output={"id": "A", "extra": "x"},
        ground_truth={"id": "A", "amount": 10},
        parse_error=False,
    )
    assert summary["error_count"] == 0
    assert summary["ground_truth_count"] == 1
    assert summary["exact_match_count"] == 0
    assert summary["partial_score_sum"] == pytest.approx(0.5)


def test_finalize_summary_sets_zero_percentages_when_no_ground_truth_count():
    summary = _new_summary()
    _finalize_summary(summary)
    assert summary["exact_accuracy_percent"] == pytest.approx(0.0)
    assert summary["partial_accuracy_percent"] == pytest.approx(0.0)


def test_finalize_summary_sets_percentages_for_non_zero_ground_truth_count():
    summary = _new_summary()
    summary["ground_truth_count"] = 4
    summary["exact_match_count"] = 1
    summary["partial_score_sum"] = 2.5
    _finalize_summary(summary)
    assert summary["exact_accuracy_percent"] == pytest.approx(25.0)
    assert summary["partial_accuracy_percent"] == pytest.approx(62.5)
