import json

import pytest

from src.category_accuracy import calculate_category_accuracy_report


def _evaluation_case(
    *,
    category: str,
    ground_truth: dict,
    llm_output: str,
) -> dict:
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
