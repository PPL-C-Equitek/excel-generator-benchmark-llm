import csv
from pathlib import Path

import pytest

from src.report_generator import EmptyDataError, ReportGenerator


def test_report_generator_initializes_csv_and_appends_rows(tmp_path):
    report_path = tmp_path / "benchmark_report.csv"
    generator = ReportGenerator(
        report_path=report_path,
        fieldnames=["row_id", "status", "score"],
    )

    generator.append_row({"row_id": 1, "status": "completed", "score": 1.0})
    generator.append_row({"row_id": 2, "status": "completed", "score": 0.5})

    summary = generator.finalize_report()

    with report_path.open("r", newline="", encoding="utf-8") as report_file:
        reader = csv.DictReader(report_file)
        rows = list(reader)

    assert reader.fieldnames == ["row_id", "status", "score"]
    assert rows == [
        {"row_id": "1", "status": "completed", "score": "1.0"},
        {"row_id": "2", "status": "completed", "score": "0.5"},
    ]
    assert summary == {
        "status": "completed",
        "total_rows": 2,
        "successful_evaluations": 2,
        "average_score": pytest.approx(0.75),
    }


def test_finalize_report_raises_empty_data_error_for_empty_input(tmp_path):
    report_path = tmp_path / "empty_report.csv"
    generator = ReportGenerator(
        report_path=report_path,
        fieldnames=["row_id", "status", "score"],
    )

    with pytest.raises(EmptyDataError, match="No evaluation data"):
        generator.finalize_report()

    with report_path.open("r", newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))

    assert rows == []


def test_report_generator_streams_rows_in_append_mode_without_result_cache(
    tmp_path,
    mocker,
):
    report_path = tmp_path / "streaming_report.csv"
    original_open = Path.open
    opened_modes = []

    def tracking_open(path, mode="r", *args, **kwargs):
        if path == report_path:
            opened_modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    mocker.patch.object(Path, "open", tracking_open)
    generator = ReportGenerator(
        report_path=report_path,
        fieldnames=["row_id", "status", "score"],
    )

    generator.append_row({"row_id": 1, "status": "completed", "score": 1.0})
    generator.append_row({"row_id": 2, "status": "failed", "score": 0.0})

    with report_path.open("r", newline="", encoding="utf-8") as report_file:
        rows_after_first_append = list(csv.DictReader(report_file))

    generator.append_row({"row_id": 3, "status": "completed", "score": 0.5})
    summary = generator.finalize_report()

    assert rows_after_first_append == [
        {"row_id": "1", "status": "completed", "score": "1.0"},
        {"row_id": "2", "status": "failed", "score": "0.0"},
    ]
    assert summary["successful_evaluations"] == 2
    assert any(mode.startswith("a") for mode in opened_modes)
    assert not any(
        hasattr(generator, attribute_name)
        for attribute_name in (
            "rows",
            "_rows",
            "results",
            "_results",
            "records",
            "_records",
        )
    )
