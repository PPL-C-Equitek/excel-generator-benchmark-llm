"""Streaming CSV report generation for benchmark evaluation results."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any


DEFAULT_FIELDNAMES = [
    "row_index",
    "status",
    "score",
    "llm_output",
    "error_message",
]
CSV_ENCODING = "utf-8"
SCORE_FIELD = "score"
STATUS_FIELD = "status"
COMPLETED_STATUS = "completed"


class EmptyDataError(ValueError):
    """Raised when a report is finalized without evaluation rows."""


class ReportGenerator:
    """Generate benchmark reports using append-only CSV streaming."""

    def __init__(
        self,
        report_path: str | Path,
        fieldnames: Sequence[str] = DEFAULT_FIELDNAMES,
    ) -> None:
        """Initialize a CSV report and write its header row.

        Args:
            report_path: Destination CSV report path.
            fieldnames: Ordered CSV header names.
        """
        self.report_path = Path(report_path)
        self.fieldnames = list(fieldnames)
        self.total_rows = 0
        self.successful_evaluations = 0
        self._cumulative_score = 0.0

        with self.report_path.open(
            "w",
            newline="",
            encoding=CSV_ENCODING,
        ) as report_file:
            writer = csv.DictWriter(report_file, fieldnames=self.fieldnames)
            writer.writeheader()

    def append_row(self, row_data: dict[str, Any]) -> None:
        """Append one evaluation result to the CSV report.

        Args:
            row_data: Evaluation result keyed by configured CSV field names.
        """
        with self.report_path.open(
            "a",
            newline="",
            encoding=CSV_ENCODING,
        ) as report_file:
            writer = csv.DictWriter(report_file, fieldnames=self.fieldnames)
            writer.writerow(row_data)
            report_file.flush()

        self.total_rows += 1
        if row_data.get(STATUS_FIELD) == COMPLETED_STATUS:
            self.successful_evaluations += 1
        self._cumulative_score += float(row_data.get(SCORE_FIELD, 0.0))

    def finalize_report(self) -> dict[str, Any]:
        """Finalize the CSV report and return aggregate statistics.

        Returns:
            Summary dictionary containing status, total row count, successful
            evaluation count, and average score.

        Raises:
            EmptyDataError: If no rows were appended before finalization.
        """
        if self.total_rows == 0:
            raise EmptyDataError("No evaluation data was written to the report.")

        return {
            "status": "completed",
            "total_rows": self.total_rows,
            "successful_evaluations": self.successful_evaluations,
            "average_score": self._cumulative_score / self.total_rows,
        }
