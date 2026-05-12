from __future__ import annotations

import re

import src.recommendation_improvements as recommendation_module


def _extract_recommendation_categories(text: str) -> list[str]:
    return re.findall(r"^- ([^:]+):", text, flags=re.MULTILINE)


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
        "Performa maksimal tercapai, tidak ada rekomendasi mendesak."
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

    assert "- Tidak ada kategori di bawah threshold." in result


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
    assert "gap 0.20 dari target" in result


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
    assert "Default recommendation digunakan" in result
