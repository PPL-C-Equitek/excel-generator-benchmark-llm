"""Generate recommendation text from category accuracy scores."""

from __future__ import annotations

from collections.abc import Callable, Mapping


DEFAULT_THRESHOLD = 0.70
REPORT_TITLE = "Recommendation Improvements"
REPORT_UNDERLINE = "==========================="
MAX_PERFORMANCE_MESSAGE = (
    "Performa maksimal tercapai, tidak ada rekomendasi mendesak."
)
NO_DATA_MESSAGE = "- No category score data available."
NO_LOW_SCORE_MESSAGE = "- Tidak ada kategori di bawah threshold."
ANALYSIS_UNAVAILABLE_MESSAGE = (
    "Default recommendation digunakan karena analisis tidak tersedia "
    "(gagal/timeout)."
)

type AnalyzerFn = Callable[[str, float], str]
type RankedCategory = tuple[str, float]


def generate_recommendation_improvements(
    *,
    category_scores: Mapping[str, float],
    threshold: float = DEFAULT_THRESHOLD,
    analyzer: AnalyzerFn | None = None,
) -> str:
    """Build a printable recommendation block for low-accuracy categories.

    Args:
        category_scores: Category-to-accuracy mapping in 0..1 scale.
        threshold: Minimum acceptable score (default ``0.70``).
        analyzer: Optional custom analyzer callable used to build each
            recommendation line.

    Returns:
        Multiline text block that can be printed or appended to report files.
    """
    report_lines: list[str] = [REPORT_TITLE, REPORT_UNDERLINE]
    effective_threshold = float(threshold)

    if not category_scores:
        report_lines.append(NO_DATA_MESSAGE)
        return _render_report(report_lines)

    if _all_max_performance(category_scores):
        report_lines.extend(["", MAX_PERFORMANCE_MESSAGE])
        return _render_report(report_lines)

    ranked_low_scores = _rank_categories_below_threshold(
        category_scores=category_scores,
        threshold=effective_threshold,
    )

    if not ranked_low_scores:
        report_lines.extend(["", NO_LOW_SCORE_MESSAGE])
        return _render_report(report_lines)

    analyzer_fn = analyzer or _build_default_analyzer(effective_threshold)

    try:
        report_lines.append("")
        for category, score in ranked_low_scores:
            recommendation = analyzer_fn(category, score)
            report_lines.append(f"- {category}: {recommendation}")
    except Exception:
        report_lines.extend(["", ANALYSIS_UNAVAILABLE_MESSAGE])

    return _render_report(report_lines)


def _all_max_performance(category_scores: Mapping[str, float]) -> bool:
    """Return ``True`` when all category scores are perfect (>= 1.0)."""
    return all(float(score) >= 1.0 for score in category_scores.values())


def _rank_categories_below_threshold(
    *,
    category_scores: Mapping[str, float],
    threshold: float,
) -> list[RankedCategory]:
    """Filter and sort low-score categories deterministically.

    Sorting rule: score ascending, then category name alphabetically.
    """
    low_score_items: list[RankedCategory] = [
        (category_name, float(category_score))
        for category_name, category_score in category_scores.items()
        if float(category_score) < threshold
    ]
    low_score_items.sort(key=lambda item: (item[1], item[0]))
    return low_score_items


def _build_default_analyzer(threshold: float) -> AnalyzerFn:
    """Build a default analyzer bound to the effective threshold."""

    def _default_analyzer(category_name: str, category_score: float) -> str:
        gap_to_target = max(0.0, threshold - float(category_score))
        return (
            "Fokus validasi field inti pada kategori ini dan tambah sampel "
            f"latihan; skor saat ini {category_score:.2f} "
            f"(gap {gap_to_target:.2f} dari target)."
        )

    return _default_analyzer


def _render_report(lines: list[str]) -> str:
    """Render report lines as newline-terminated text."""
    return "\n".join(lines) + "\n"
