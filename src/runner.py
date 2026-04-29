"""Benchmark orchestration for dataset, LLM client, and metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.dataset_loader import load_dataset
from src.llm_client import LLMAuthError, LLMClient
from src.metrics import calculate_accuracy, parse_llm_output


PROMPT_KEY = "prompt"
EXPECTED_OUTPUT_KEY = "expected_output"
DEFAULT_DATASET_SCHEMA: Mapping[str, type[Any]] = {
    PROMPT_KEY: str,
    EXPECTED_OUTPUT_KEY: dict,
}


class BenchmarkRunner:
    """Run a benchmark dataset through an LLM and aggregate scores."""

    def __init__(
        self,
        dataset_path: str | Path,
        model: str | None = None,
        *,
        model_name: str | None = None,
        schema: Mapping[str, type[Any]] = DEFAULT_DATASET_SCHEMA,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            dataset_path: Path to the benchmark dataset.
            model: LLM model name used by the gateway client.
            model_name: Keyword-only alias for ``model``.
            schema: Dataset schema passed to the dataset loader.

        Raises:
            ValueError: If neither ``model`` nor ``model_name`` is provided.
        """
        selected_model = model if model is not None else model_name
        if selected_model is None:
            raise ValueError("A model name must be provided.")

        self.dataset_path = dataset_path
        self.schema = schema
        self.llm_client = LLMClient(model=selected_model)

    def run(self) -> dict[str, Any]:
        """Execute the benchmark pipeline and return summary statistics.

        Returns:
            Summary dictionary with exactly ``status``, ``total_rows``,
            ``successful_evaluations``, and ``average_score`` keys.
        """
        rows = load_dataset(self.dataset_path, self.schema)
        total_rows = 0
        scores: list[float] = []

        for row in rows:
            total_rows += 1
            prompt = str(row[PROMPT_KEY])
            ground_truth = _ground_truth_from_row(row)

            try:
                raw_output = self.llm_client.generate_text(prompt)
            except LLMAuthError as exc:
                print(str(exc))
                return _summary(
                    status="aborted_due_to_auth",
                    total_rows=total_rows,
                    scores=scores,
                )

            parsed_output = parse_llm_output(raw_output)
            scores.append(calculate_accuracy(parsed_output, ground_truth))

        return _summary(
            status="completed",
            total_rows=total_rows,
            scores=scores,
        )


def _ground_truth_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Extract expected output fields from a dataset row.

    Args:
        row: Benchmark dataset row.

    Returns:
        The nested ``expected_output`` dictionary when present, otherwise a
        dictionary containing every row field except ``prompt``.
    """
    expected_output = row.get(EXPECTED_OUTPUT_KEY)
    if isinstance(expected_output, dict):
        return expected_output

    return {
        key: value
        for key, value in row.items()
        if key not in {PROMPT_KEY, EXPECTED_OUTPUT_KEY}
    }


def _summary(
    *,
    status: str,
    total_rows: int,
    scores: list[float],
) -> dict[str, Any]:
    """Build a benchmark summary dictionary.

    Args:
        status: Final benchmark run status.
        total_rows: Number of dataset rows evaluated or attempted.
        scores: Scores collected from successful evaluations.

    Returns:
        Summary dictionary containing exactly the public runner summary keys.
    """
    successful_evaluations = len(scores)
    average_score = sum(scores) / successful_evaluations if scores else 0.0
    result: dict[str, Any] = {
        "status": status,
        "total_rows": total_rows,
        "successful_evaluations": successful_evaluations,
        "average_score": average_score,
    }
    return result
