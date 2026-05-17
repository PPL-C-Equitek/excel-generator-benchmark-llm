from __future__ import annotations

import re

import src.recommendation_improvements as recommendation_module


def _extract_recommendation_categories(text: str) -> list[str]:
    return re.findall(r"^- ([^:]+):", text, flags=re.MULTILINE)


def _recommendation_line_for_category(text: str, category: str) -> str:
    pattern = rf"^- {re.escape(category)}:\s*(.+)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match, f"Category line for {category!r} not found in:\n{text}"
    return match.group(1).strip()


def test_generate_recommendations_for_categories_below_threshold_with_appendable_format():
    category_scores = {
        "invoice": 0.95,
        "purchase_order": 0.62,
        "tax_form": 0.68,
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )

    assert isinstance(result, str)
    assert result.startswith("Recommendation Improvements\n")
    assert "purchase_order" in result
    assert "tax_form" in result
    assert "invoice" not in result
    assert "- purchase_order:" in result
    assert "- tax_form:" in result
    assert result.endswith("\n")


def test_generate_recommendations_returns_default_message_when_analyzer_raises_exception():
    category_scores = {
        "invoice": 0.65,
        "receipt": 0.64,
    }

    def failing_analyzer(category: str, score: float) -> str:
        raise RuntimeError(f"Analyzer failed for {category}: {score}")

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
        analyzer=failing_analyzer,
    )

    assert isinstance(result, str)
    assert "Recommendation Improvements" in result
    assert "default" in result.lower() or "tidak tersedia" in result.lower()


def test_generate_recommendations_returns_default_message_when_analyzer_times_out():
    category_scores = {"invoice": 0.69}

    def timeout_analyzer(category: str, score: float) -> str:
        raise TimeoutError(f"Timeout while analyzing {category}: {score}")

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
        analyzer=timeout_analyzer,
    )

    assert isinstance(result, str)
    assert "Recommendation Improvements" in result
    assert "default" in result.lower() or "timeout" in result.lower()


def test_generate_recommendations_returns_max_performance_message_when_all_categories_are_perfect():
    category_scores = {
        "invoice": 1.0,
        "receipt": 1.0,
        "purchase_order": 1.0,
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )

    assert isinstance(result, str)
    assert (
        "Maximum performance reached, no urgent recommendation needed."
        in result
    )


def test_generate_recommendations_sorts_tied_lowest_categories_alphabetically():
    category_scores = {
        "zeta_form": 0.40,
        "alpha_form": 0.40,
        "beta_form": 0.40,
        "invoice": 0.85,
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )

    categories = _extract_recommendation_categories(result)
    assert categories == ["alpha_form", "beta_form", "zeta_form"]


def test_generate_recommendations_returns_no_data_message_for_empty_scores():
    result = recommendation_module.generate_recommendation_improvements(
        category_scores={},
        threshold=0.70,
    )

    assert "Recommendation Improvements" in result
    assert "- No category score data available." in result


def test_generate_recommendations_returns_info_when_no_category_below_threshold():
    category_scores = {
        "invoice": 0.91,
        "receipt": 0.78,
        "purchase_order": 0.70,
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )

    assert "- No categories below threshold." in result


def test_generate_recommendations_sorts_by_score_then_category_name():
    category_scores = {
        "zeta_form": 0.45,
        "alpha_form": 0.40,
        "beta_form": 0.40,
        "invoice": 0.85,
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )

    categories = _extract_recommendation_categories(result)
    assert categories == ["alpha_form", "beta_form", "zeta_form"]


def test_generate_recommendations_default_analyzer_uses_given_threshold_for_gap():
    category_scores = {"invoice": 0.60}

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.80,
    )

    assert "- invoice:" in result
    assert "gap 0.20 from target" in result


def test_generate_recommendations_returns_default_message_for_unexpected_exception():
    category_scores = {"invoice": 0.65}

    def unexpected_failure(category: str, score: float) -> str:
        raise KeyError(f"Unexpected analyzer error for {category}: {score}")

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
        analyzer=unexpected_failure,
    )

    assert "Recommendation Improvements" in result
    assert "Default recommendation was used" in result


def test_generate_recommendations_does_not_mix_partial_and_fallback_on_mid_loop_failure():
    category_scores = {
        "alpha": 0.50,
        "beta": 0.55,
        "gamma": 0.60,
    }
    call_count = {"value": 0}

    def analyzer_fails_on_third(category: str, score: float) -> str:
        call_count["value"] += 1
        if call_count["value"] == 3:
            raise RuntimeError(f"Analyzer failed at {category}: {score}")
        return f"Improve {category}"

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
        analyzer=analyzer_fails_on_third,
    )

    assert "Default recommendation was used" in result
    assert "- alpha: Improve alpha" not in result
    assert "- beta: Improve beta" not in result


def test_generate_recommendations_includes_severity_tiers_based_on_score():
    category_scores = {
        "cat_critical": 0.10,
        "cat_high": 0.35,
        "cat_medium": 0.60,
        "cat_low": 0.80,
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=1.0,
    )

    assert "critical" in _recommendation_line_for_category(
        result,
        "cat_critical",
    ).lower()
    assert "high" in _recommendation_line_for_category(result, "cat_high").lower()
    assert "medium" in _recommendation_line_for_category(
        result,
        "cat_medium",
    ).lower()
    assert "low" in _recommendation_line_for_category(result, "cat_low").lower()


def test_generate_recommendations_are_category_aware_for_filetype_groups():
    category_scores = {
        "docx": 0.05,
        "pdf": 0.06,
        "png": 0.07,
        "csv": 0.08,
        "xlsx": 0.09,
        "txt": 0.10,
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )
    lower_result = result.lower()

    assert "docx" in lower_result
    assert "pdf" in lower_result
    assert "png" in lower_result
    assert "csv" in lower_result
    assert "xlsx" in lower_result
    assert "txt" in lower_result

    for category in ("docx", "pdf", "png"):
        line = _recommendation_line_for_category(result, category).lower()
        assert any(
            keyword in line for keyword in ("ocr", "layout", "vision", "parsing")
        ), line

    for category in ("csv", "xlsx"):
        line = _recommendation_line_for_category(result, category).lower()
        assert any(
            keyword in line for keyword in ("schema", "header", "type", "normalize")
        ), line

    txt_line = _recommendation_line_for_category(result, "txt").lower()
    assert any(
        keyword in txt_line
        for keyword in ("delimiter", "key-value", "regex", "extract")
    ), txt_line


def test_generate_recommendations_prioritizes_worst_categories_first_for_benchmark_like_scores():
    category_scores = {
        "xlsx": 0.00,
        "pdf": 0.00,
        "csv": 0.0833333333333333,
        "docx": 0.00,
        "png": 0.00,
        "txt": 0.00,
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )

    categories = _extract_recommendation_categories(result)
    assert categories == ["docx", "pdf", "png", "txt", "xlsx", "csv"]


def test_generate_recommendations_remain_deterministic_across_input_order():
    scores_a = {
        "txt": 0.00,
        "csv": 0.0833333333333333,
        "pdf": 0.00,
        "docx": 0.00,
        "xlsx": 0.00,
        "png": 0.00,
    }
    scores_b = {
        "png": 0.00,
        "xlsx": 0.00,
        "docx": 0.00,
        "pdf": 0.00,
        "csv": 0.0833333333333333,
        "txt": 0.00,
    }

    result_a = recommendation_module.generate_recommendation_improvements(
        category_scores=scores_a,
        threshold=0.70,
    )
    result_b = recommendation_module.generate_recommendation_improvements(
        category_scores=scores_b,
        threshold=0.70,
    )

    assert result_a == result_b


def test_generate_recommendations_benchmark_payload_has_non_copy_paste_category_guidance():
    benchmark_like_payload = {
        "by_category": {
            "xlsx": {"exact_accuracy_percent": 0.0},
            "pdf": {"exact_accuracy_percent": 0.0},
            "csv": {"exact_accuracy_percent": 8.333333333333332},
            "docx": {"exact_accuracy_percent": 0.0},
            "png": {"exact_accuracy_percent": 0.0},
            "txt": {"exact_accuracy_percent": 0.0},
        }
    }
    category_scores = {
        category: summary["exact_accuracy_percent"] / 100.0
        for category, summary in benchmark_like_payload["by_category"].items()
    }

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )

    lines = {
        category: _recommendation_line_for_category(result, category).lower()
        for category in ("docx", "pdf", "png", "txt", "xlsx", "csv")
    }

    assert len(set(lines.values())) == len(lines), lines
    assert any(keyword in lines["docx"] for keyword in ("ocr", "layout", "vision"))
    assert any(keyword in lines["pdf"] for keyword in ("ocr", "layout", "vision"))
    assert any(keyword in lines["png"] for keyword in ("ocr", "layout", "vision"))
    assert any(
        keyword in lines["txt"]
        for keyword in ("delimiter", "key-value", "regex", "extract")
    )
    assert any(
        keyword in lines["xlsx"]
        for keyword in ("schema", "header", "type", "normalize")
    )
    assert any(
        keyword in lines["csv"]
        for keyword in ("schema", "header", "type", "normalize")
    )


def test_generate_recommendations_category_lookup_is_case_and_whitespace_tolerant():
    category_scores = {" PDF ": 0.05}

    result = recommendation_module.generate_recommendation_improvements(
        category_scores=category_scores,
        threshold=0.70,
    )
    line = _recommendation_line_for_category(result, " PDF ").lower()

    assert any(keyword in line for keyword in ("ocr", "layout", "vision"))


def test_severity_tier_returns_default_for_scores_at_or_above_threshold():
    assert (
        recommendation_module._severity_tier(
            category_score=0.80,
            threshold=0.70,
        )
        == "low"
    )


def test_severity_tier_handles_exact_boundaries():
    assert recommendation_module._severity_tier(
        category_score=0.10,
        threshold=1.0,
    ) == "critical"
    assert recommendation_module._severity_tier(
        category_score=0.40,
        threshold=1.0,
    ) == "high"
    assert recommendation_module._severity_tier(
        category_score=0.70,
        threshold=1.0,
    ) == "medium"


def test_severity_tier_returns_low_above_highest_tier_bound_and_below_threshold():
    assert recommendation_module._severity_tier(
        category_score=0.71,
        threshold=1.0,
    ) == "low"
