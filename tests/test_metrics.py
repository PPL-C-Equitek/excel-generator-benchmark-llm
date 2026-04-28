import pytest

from src.metrics import calculate_accuracy, parse_llm_output


def test_parse_llm_output_returns_dict_for_valid_json_object():
    llm_output = (
        '{"invoice_id": "INV-001", "vendor": "Acme Corp", '
        '"total": 125000, "currency": "IDR"}'
    )

    parsed_output = parse_llm_output(llm_output)

    assert parsed_output == {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }


@pytest.mark.parametrize(
    "llm_output",
    [
        "not valid json",
        '["invoice_id", "vendor", "total"]',
        '"plain string payload"',
    ],
)
def test_parse_llm_output_returns_empty_dict_for_invalid_or_non_dict_json(
    llm_output,
):
    parsed_output = parse_llm_output(llm_output)

    assert parsed_output == {}


def test_calculate_accuracy_returns_perfect_score_for_exact_dict_match():
    parsed_output = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }

    score = calculate_accuracy(parsed_output, ground_truth)

    assert score == pytest.approx(1.0)


def test_calculate_accuracy_returns_partial_score_for_partial_match():
    parsed_output = {
        "invoice_id": "INV-001",
        "vendor": "Wrong Vendor",
        "total": 125000,
    }
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }

    score = calculate_accuracy(parsed_output, ground_truth)

    assert score == pytest.approx(0.5)


def test_calculate_accuracy_penalizes_extra_hallucinated_keys():
    parsed_output = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
        "payment_terms": "Net 30",
        "approved_by": "Finance Lead",
    }
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }

    score = calculate_accuracy(parsed_output, ground_truth)

    assert score == pytest.approx(4 / 6)


def test_calculate_accuracy_treats_type_mismatch_as_incorrect():
    parsed_output = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": "125000",
        "currency": "IDR",
    }
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }

    score = calculate_accuracy(parsed_output, ground_truth)

    assert score == pytest.approx(0.75)


def test_calculate_accuracy_compares_nested_values_exactly():
    parsed_output = {
        "invoice_id": "INV-001",
        "line_item": {
            "description": "Software license",
            "quantity": 2,
            "unit_price": 50000,
        },
    }
    ground_truth = {
        "invoice_id": "INV-001",
        "line_item": {
            "description": "Software license",
            "quantity": 2,
            "unit_price": 50000,
        },
    }

    score = calculate_accuracy(parsed_output, ground_truth)

    assert score == pytest.approx(1.0)


def test_calculate_accuracy_counts_changed_nested_values_as_incorrect():
    parsed_output = {
        "invoice_id": "INV-001",
        "line_item": {
            "description": "Software license",
            "quantity": 1,
            "unit_price": 50000,
        },
    }
    ground_truth = {
        "invoice_id": "INV-001",
        "line_item": {
            "description": "Software license",
            "quantity": 2,
            "unit_price": 50000,
        },
    }

    score = calculate_accuracy(parsed_output, ground_truth)

    assert score == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("parsed_output", "expected_score"),
    [
        ({}, 1.0),
        ({"invoice_id": "INV-001"}, 0.0),
    ],
)
def test_calculate_accuracy_handles_empty_ground_truth(
    parsed_output,
    expected_score,
):
    score = calculate_accuracy(parsed_output, {})

    assert score == pytest.approx(expected_score)


def test_calculate_accuracy_accepts_raw_json_string_input():
    parsed_output = (
        '{"invoice_id": "INV-001", "vendor": "Acme Corp", '
        '"total": 125000, "currency": "IDR"}'
    )
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
        "total": 125000,
        "currency": "IDR",
    }

    score = calculate_accuracy(parsed_output, ground_truth)

    assert score == pytest.approx(1.0)


def test_calculate_accuracy_treats_invalid_string_input_as_empty_output():
    ground_truth = {
        "invoice_id": "INV-001",
        "vendor": "Acme Corp",
    }

    score = calculate_accuracy("not valid json", ground_truth)

    assert score == pytest.approx(0.0)
