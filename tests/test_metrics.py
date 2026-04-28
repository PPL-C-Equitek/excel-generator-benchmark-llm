import pytest

from src.metrics import calculate_accuracy


def test_calculate_accuracy_returns_perfect_score_for_exact_json_match():
    llm_output = (
        '{"invoice_id": "INV-001", "vendor": "Acme Corp", '
        '"total": 125000, "currency": "IDR"}'
    )
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }

    score = calculate_accuracy(llm_output, ground_truth)

    assert score == pytest.approx(1.0)


def test_calculate_accuracy_returns_partial_score_for_partial_match():
    llm_output = (
        '{"invoice_id": "INV-001", "vendor": "Wrong Vendor", '
        '"total": 125000}'
    )
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }

    score = calculate_accuracy(llm_output, ground_truth)

    assert score == pytest.approx(0.5)


def test_calculate_accuracy_returns_zero_for_malformed_json_output():
    llm_output = "The invoice id is INV-001 and the total is 125000 IDR."
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }

    score = calculate_accuracy(llm_output, ground_truth)

    assert score == pytest.approx(0.0)
