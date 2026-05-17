"""Generate recommendation text from category accuracy scores."""

from __future__ import annotations

from collections.abc import Callable, Mapping


DEFAULT_THRESHOLD = 0.70
REPORT_TITLE = "Recommendation Improvements"
REPORT_UNDERLINE = "==========================="
MAX_PERFORMANCE_MESSAGE = (
    "Maximum performance reached, no urgent recommendation needed."
)
NO_DATA_MESSAGE = "- No category score data available."
NO_LOW_SCORE_MESSAGE = "- No categories below threshold."
ANALYSIS_UNAVAILABLE_MESSAGE = (
    "Default recommendation was used because analysis is unavailable "
    "(failure/timeout)."
)
GENERIC_ACTION_MESSAGE = (
    "Focus on validating core fields for this category and add more training "
    "samples"
)
SEVERITY_THRESHOLDS = (
    ("critical", 0.10),
    ("high", 0.40),
    ("medium", 0.70),
)
SEVERITY_DEFAULT = "low"

CATEGORY_POLICIES = (
    (
        ("docx",),
        (
            "Run block-level OCR, audit multi-column table layout, and verify "
            "vision parsing for headings and section boundaries"
        ),
    ),
    (
        ("pdf",),
        (
            "Enable language-aware OCR, fix layout reading order, and validate "
            "vision parsing on tables and footer regions"
        ),
    ),
    (
        ("png",),
        (
            "Improve OCR image pre-processing, segment text layout regions, and "
            "validate vision parsing on small fields"
        ),
    ),
    (
        ("csv",),
        (
            "Normalize schema columns, standardize header aliases, and lock type "
            "casting for numeric/date values during ingest"
        ),
    ),
    (
        ("xlsx",),
        (
            "Normalize cross-sheet schema, clean merged-cell headers, and enforce "
            "type normalization for formula and date cells"
        ),
    ),
    (
        ("txt",),
        (
            "Set delimiter priority rules, extract explicit key-value pairs, and "
            "prepare regex extraction for free-form text variations"
        ),
    ),
)

type AnalyzerFn = Callable[[str, float], str]
type RankedCategory = tuple[str, float]
type CategoryActionMap = dict[str, str]


def _build_category_action_map() -> CategoryActionMap:
    """Expand grouped category policies into a flat lookup map."""
    action_map: CategoryActionMap = {}
    for categories, action_text in CATEGORY_POLICIES:
        for category in categories:
            action_map[category] = action_text
    return action_map


CATEGORY_ACTIONS = _build_category_action_map()


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

    buffered_recommendation_lines: list[str] = []
    try:
        for category, score in ranked_low_scores:
            recommendation = analyzer_fn(category, score)
            buffered_recommendation_lines.append(f"- {category}: {recommendation}")
    except Exception:
        report_lines.extend(["", ANALYSIS_UNAVAILABLE_MESSAGE])
    else:
        report_lines.append("")
        report_lines.extend(buffered_recommendation_lines)

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
    """Build the default analyzer bound to an effective threshold."""

    def _default_analyzer(category_name: str, category_score: float) -> str:
        severity = _severity_tier(category_score=category_score, threshold=threshold)
        action_text = _action_for_category(category_name)
        gap_to_target = max(0.0, threshold - float(category_score))
        return (
            f"[{severity}] {action_text}; current score {category_score:.2f} "
            f"(gap {gap_to_target:.2f} from target)."
        )

    return _default_analyzer


def _severity_tier(*, category_score: float, threshold: float) -> str:
    """Return a severity tier label based on score bands and threshold."""
    score_value = float(category_score)
    for severity_label, upper_bound in SEVERITY_THRESHOLDS:
        if score_value <= upper_bound:
            return severity_label
    if score_value < float(threshold):
        return SEVERITY_DEFAULT
    return SEVERITY_DEFAULT


def _action_for_category(category_name: str) -> str:
    """Return category-specific action guidance, with generic fallback."""
    normalized_name = _normalize_category_name(category_name)
    return CATEGORY_ACTIONS.get(normalized_name, GENERIC_ACTION_MESSAGE)


def _normalize_category_name(category_name: str) -> str:
    """Normalize category keys to stable lowercase lookup form."""
    return str(category_name).strip().lower()


def _render_report(lines: list[str]) -> str:
    """Render report lines as newline-terminated text."""
    return "\n".join(lines) + "\n"
