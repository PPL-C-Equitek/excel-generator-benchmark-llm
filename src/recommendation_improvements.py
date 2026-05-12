"""Generate recommendation improvements from category accuracy scores."""

from __future__ import annotations

from typing import Callable


DEFAULT_THRESHOLD = 0.70
MAX_PERFORMANCE_MESSAGE = (
    "Performa maksimal tercapai, tidak ada rekomendasi mendesak."
)


def generate_recommendation_improvements(
    *,
    category_scores: dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
    analyzer: Callable[[str, float], str] | None = None,
) -> str:
    """Build recommendation text block for categories below threshold."""
    lines = ["Recommendation Improvements", "==========================="]

    if not category_scores:
        lines.append("- No category score data available.")
        return "\n".join(lines) + "\n"

    if _all_max_performance(category_scores):
        lines.extend(["", MAX_PERFORMANCE_MESSAGE])
        return "\n".join(lines) + "\n"

    low_scores = [
        (category, float(score))
        for category, score in category_scores.items()
        if float(score) < float(threshold)
    ]

    if not low_scores:
        lines.extend(["", "- Tidak ada kategori di bawah threshold."])
        return "\n".join(lines) + "\n"

    low_scores.sort(key=lambda item: (item[1], item[0]))
    analyzer_fn = analyzer or _default_analyzer

    try:
        lines.append("")
        for category, score in low_scores:
            recommendation = analyzer_fn(category, score)
            lines.append(f"- {category}: {recommendation}")
    except Exception:
        lines.extend(
            [
                "",
                (
                    "Default recommendation digunakan karena analisis tidak tersedia "
                    "(gagal/timeout)."
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def _all_max_performance(category_scores: dict[str, float]) -> bool:
    return all(float(score) >= 1.0 for score in category_scores.values())


def _default_analyzer(category: str, score: float) -> str:
    gap = max(0.0, DEFAULT_THRESHOLD - float(score))
    return (
        f"Fokus validasi field inti pada kategori ini dan tambah sampel latihan; "
        f"skor saat ini {score:.2f} (gap {gap:.2f} dari target)."
    )

