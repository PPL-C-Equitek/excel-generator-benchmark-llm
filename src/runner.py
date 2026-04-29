"""Benchmark orchestration for dataset, LLM client, and metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from src.dataset_loader import load_dataset
from src.llm_client import LLMAuthError, LLMClient
from src.metrics import calculate_accuracy, parse_llm_output
from src.report_generator import EmptyDataError, ReportGenerator


PROMPT_KEY = "prompt"
EXPECTED_OUTPUT_KEY = "expected_output"
DEFAULT_REPORT_PATH = "benchmark_report.csv"
REPORT_FIELDNAMES = [
    "row_index",
    "status",
    "score",
    "llm_output",
    "error_message",
]
DEFAULT_DATASET_SCHEMA: Mapping[str, type[Any]] = {
    PROMPT_KEY: str,
    EXPECTED_OUTPUT_KEY: dict,
}


class BenchmarkSummary(TypedDict):
    """Public summary schema returned by ``BenchmarkRunner.run``."""

    status: str
    total_rows: int
    successful_evaluations: int
    average_score: float


class BenchmarkRunner:
    """Orchestrate dataset loading, LLM generation, and metric evaluation.

    ``BenchmarkRunner`` loads a benchmark dataset, sends each row's prompt to
    the configured LLM model, scores the generated JSON against the row's
    ground truth, and returns aggregate run statistics.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        model: str | None = None,
        *,
        model_name: str | None = None,
        schema: Mapping[str, type[Any]] = DEFAULT_DATASET_SCHEMA,
        report_path: str | Path = DEFAULT_REPORT_PATH,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            dataset_path: Path to the benchmark dataset.
            model: LLM model name used by the gateway client.
            model_name: Keyword-only alias for ``model``.
            schema: Dataset schema passed to the dataset loader.
            report_path: Destination CSV report path.

        Raises:
            ValueError: If neither ``model`` nor ``model_name`` is provided.
        """
        selected_model = model if model is not None else model_name
        if selected_model is None:
            raise ValueError("A model name must be provided.")

        self.dataset_path = dataset_path
        self.schema = schema
        self.llm_client = LLMClient(model=selected_model)
        self.report_generator = ReportGenerator(
            report_path=report_path,
            fieldnames=REPORT_FIELDNAMES,
        )

    def run(self) -> BenchmarkSummary:
        """Execute the benchmark pipeline and return summary statistics.

        Returns:
            A ``BenchmarkSummary`` dictionary with exactly these keys:
            ``status`` containing the final run state, ``total_rows``
            containing the number of records evaluated or attempted,
            ``successful_evaluations`` containing the number of rows processed
            without exceptions, and ``average_score`` containing the mean score
            across all streamed report rows. Empty datasets return an
            ``average_score`` of ``0.0``.
        """
        rows = load_dataset(self.dataset_path, self.schema)
        successful_scores: list[float] = []

        for row_index, row in enumerate(rows, start=1):
            try:
                successful_scores.append(self._evaluate_single_row(row_index, row))
            except LLMAuthError as exc:
                print(str(exc))
                return _summary(
                    status="aborted_due_to_auth",
                    total_rows=row_index,
                    successful_evaluations=len(successful_scores),
                    average_score=_average_score(successful_scores),
                )
            except Exception as exc:
                print(str(exc))
                self.report_generator.append_row(
                    _failed_row_data(row_index, str(exc))
                )

        try:
            report_summary = self.report_generator.finalize_report()
        except EmptyDataError:
            return _empty_summary()

        return _summary_from_report(
            report_summary,
            successful_evaluations=len(successful_scores),
        )

    def _evaluate_single_row(self, row_index: int, row: dict[str, Any]) -> float:
        """Evaluate a single benchmark row.

        Args:
            row_index: One-based row number used in the streamed report.
            row: Dataset row containing a prompt and expected output fields.

        Returns:
            Accuracy score for the row.

        Raises:
            LLMAuthError: If the LLM client reports invalid credentials.
        """
        prompt = str(row[PROMPT_KEY])
        ground_truth = _ground_truth_from_row(row)
        raw_output = self.llm_client.generate_text(prompt)
        parsed_output = parse_llm_output(raw_output)
        score = calculate_accuracy(parsed_output, ground_truth)
        self.report_generator.append_row(
            _completed_row_data(row_index, score, raw_output)
        )
        return score


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


def _completed_row_data(
    row_index: int,
    score: float,
    llm_output: str,
) -> dict[str, Any]:
    """Build report data for a successful row evaluation.

    Args:
        row_index: One-based row number.
        score: Strict evaluation score for the row.
        llm_output: Raw LLM output text.

    Returns:
        Report row dictionary matching ``REPORT_FIELDNAMES``.
    """
    return {
        "row_index": row_index,
        "status": "completed",
        "score": score,
        "llm_output": llm_output,
        "error_message": "",
    }


def _failed_row_data(row_index: int, error_message: str) -> dict[str, Any]:
    """Build report data for a failed row evaluation.

    Args:
        row_index: One-based row number.
        error_message: Error text captured while processing the row.

    Returns:
        Report row dictionary matching ``REPORT_FIELDNAMES``.
    """
    return {
        "row_index": row_index,
        "status": "failed",
        "score": 0.0,
        "llm_output": "",
        "error_message": error_message,
    }


def _summary(
    *,
    status: str,
    total_rows: int,
    successful_evaluations: int,
    average_score: float,
) -> BenchmarkSummary:
    """Build a benchmark summary dictionary.

    Args:
        status: Final benchmark run status.
        total_rows: Number of dataset rows evaluated or attempted.
        successful_evaluations: Number of rows processed without exceptions.
        average_score: Average score reported for processed rows.

    Returns:
        Summary dictionary containing exactly the public runner summary keys.
    """
    result: BenchmarkSummary = {
        "status": status,
        "total_rows": total_rows,
        "successful_evaluations": successful_evaluations,
        "average_score": average_score,
    }
    return result


def _summary_from_report(
    report_summary: Mapping[str, Any],
    successful_evaluations: int,
) -> BenchmarkSummary:
    """Convert a report summary into the public runner summary schema.

    Args:
        report_summary: Summary returned by ``ReportGenerator.finalize_report``.
        successful_evaluations: Successful row count tracked by the runner.

    Returns:
        Public ``BenchmarkSummary`` with all required keys.
    """
    return _summary(
        status=str(report_summary["status"]),
        total_rows=int(report_summary["total_rows"]),
        successful_evaluations=int(
            report_summary.get(
                "successful_evaluations",
                successful_evaluations,
            )
        ),
        average_score=float(report_summary["average_score"]),
    )


def _average_score(scores: list[float]) -> float:
    """Calculate a safe average for collected scores.

    Args:
        scores: Completed evaluation scores.

    Returns:
        Average score, or ``0.0`` when no scores exist.
    """
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def _empty_summary() -> BenchmarkSummary:
    """Return the public summary for an empty benchmark dataset.

    Returns:
        Empty-run summary with a safe ``0.0`` average score.
    """
    return _summary(
        status="empty",
        total_rows=0,
        successful_evaluations=0,
        average_score=0.0,
    )
