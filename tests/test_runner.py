import csv

import pytest

from src.llm_client import LLMAuthError
from src.report_generator import EmptyDataError
from src.runner import BenchmarkRunner


EXPECTED_SUMMARY_KEYS = {
    "status",
    "total_rows",
    "successful_evaluations",
    "average_score",
}
REPORT_FIELDNAMES = [
    "row_index",
    "status",
    "score",
    "llm_output",
    "error_message",
]


def test_benchmark_runner_streams_rows_to_report_and_returns_report_summary(
    mocker,
):
    dataset_rows = [
        {
            "prompt": "Extract invoice INV-001",
            "expected_output": {
                "invoice_id": "INV-001",
                "total": 125000,
            },
        },
        {
            "prompt": "Extract invoice INV-002",
            "expected_output": {
                "invoice_id": "INV-002",
                "total": 75000,
            },
        },
    ]
    final_summary = {
        "status": "completed",
        "total_rows": 2,
        "successful_evaluations": 2,
        "average_score": 0.75,
    }
    load_dataset_mock = mocker.patch(
        "src.runner.load_dataset",
        return_value=dataset_rows,
    )
    llm_client = mocker.Mock()
    llm_client.generate_text.side_effect = [
        '{"invoice_id": "INV-001", "total": 125000}',
        '{"invoice_id": "INV-002", "total": 99999}',
    ]
    llm_client_class = mocker.patch(
        "src.runner.LLMClient",
        return_value=llm_client,
    )
    report_generator = mocker.Mock()
    report_generator.finalize_report.return_value = final_summary
    report_generator_class = mocker.patch(
        "src.runner.ReportGenerator",
        return_value=report_generator,
    )

    runner = BenchmarkRunner(
        dataset_path="benchmark.json",
        model="benchmark-model",
        report_path="custom_report.csv",
    )
    summary = runner.run()

    assert set(summary) == EXPECTED_SUMMARY_KEYS
    assert summary == final_summary
    load_dataset_mock.assert_called_once()
    assert load_dataset_mock.call_args.args[0] == "benchmark.json"
    llm_client_class.assert_called_once_with(model="benchmark-model")
    report_generator_class.assert_called_once_with(
        report_path="custom_report.csv",
        fieldnames=REPORT_FIELDNAMES,
    )
    report_generator.append_row.assert_has_calls(
        [
            mocker.call(
                {
                    "row_index": 1,
                    "status": "completed",
                    "score": pytest.approx(1.0),
                    "llm_output": '{"invoice_id": "INV-001", "total": 125000}',
                    "error_message": "",
                }
            ),
            mocker.call(
                {
                    "row_index": 2,
                    "status": "completed",
                    "score": pytest.approx(0.5),
                    "llm_output": '{"invoice_id": "INV-002", "total": 99999}',
                    "error_message": "",
                }
            ),
        ]
    )
    report_generator.finalize_report.assert_called_once_with()


def test_benchmark_runner_accepts_model_name_alias(mocker):
    mocker.patch("src.runner.load_dataset", return_value=[])
    llm_client_class = mocker.patch("src.runner.LLMClient")
    mocker.patch("src.runner.ReportGenerator")

    BenchmarkRunner(
        dataset_path="benchmark.json",
        model_name="alias-model",
    )

    llm_client_class.assert_called_once_with(model="alias-model")


def test_benchmark_runner_requires_a_model_name():
    with pytest.raises(ValueError, match="A model name must be provided"):
        BenchmarkRunner(dataset_path="benchmark.json")


def test_benchmark_runner_reports_row_failure_and_continues(mocker, capsys):
    dataset_rows = [
        {
            "prompt": "Extract invoice INV-001",
            "expected_output": {
                "invoice_id": "INV-001",
                "total": 125000,
            },
        },
        {
            "prompt": "Extract invoice INV-002",
            "expected_output": {
                "invoice_id": "INV-002",
                "total": 75000,
            },
        },
    ]
    final_summary = {
        "status": "completed",
        "total_rows": 2,
        "successful_evaluations": 1,
        "average_score": 0.5,
    }
    mocker.patch("src.runner.load_dataset", return_value=dataset_rows)
    llm_client = mocker.Mock()
    llm_client.generate_text.side_effect = [
        RuntimeError("Temporary provider failure"),
        '{"invoice_id": "INV-002", "total": 75000}',
    ]
    mocker.patch("src.runner.LLMClient", return_value=llm_client)
    report_generator = mocker.Mock()
    report_generator.finalize_report.return_value = final_summary
    mocker.patch("src.runner.ReportGenerator", return_value=report_generator)

    runner = BenchmarkRunner(
        dataset_path="benchmark.json",
        model="benchmark-model",
    )
    summary = runner.run()

    captured = capsys.readouterr()

    assert "Temporary provider failure" in captured.out
    assert summary == final_summary
    report_generator.append_row.assert_has_calls(
        [
            mocker.call(
                {
                    "row_index": 1,
                    "status": "failed",
                    "score": 0.0,
                    "llm_output": "",
                    "error_message": "Temporary provider failure",
                }
            ),
            mocker.call(
                {
                    "row_index": 2,
                    "status": "completed",
                    "score": pytest.approx(1.0),
                    "llm_output": '{"invoice_id": "INV-002", "total": 75000}',
                    "error_message": "",
                }
            ),
        ]
    )


def test_benchmark_runner_uses_flat_row_as_ground_truth(mocker):
    dataset_rows = [
        {
            "prompt": "Extract invoice INV-001",
            "invoice_id": "INV-001",
            "total": 125000,
        },
    ]
    final_summary = {
        "status": "completed",
        "total_rows": 1,
        "successful_evaluations": 1,
        "average_score": 1.0,
    }
    mocker.patch("src.runner.load_dataset", return_value=dataset_rows)
    llm_client = mocker.Mock()
    llm_client.generate_text.return_value = (
        '{"invoice_id": "INV-001", "total": 125000}'
    )
    mocker.patch("src.runner.LLMClient", return_value=llm_client)
    report_generator = mocker.Mock()
    report_generator.finalize_report.return_value = final_summary
    mocker.patch("src.runner.ReportGenerator", return_value=report_generator)

    runner = BenchmarkRunner(
        dataset_path="benchmark.json",
        model="benchmark-model",
    )
    summary = runner.run()

    assert summary == final_summary
    report_generator.append_row.assert_called_once_with(
        {
            "row_index": 1,
            "status": "completed",
            "score": pytest.approx(1.0),
            "llm_output": '{"invoice_id": "INV-001", "total": 125000}',
            "error_message": "",
        }
    )


def test_benchmark_runner_auth_abort_before_success_returns_zero_average(
    mocker,
):
    dataset_rows = [
        {
            "prompt": "Extract invoice INV-001",
            "expected_output": {
                "invoice_id": "INV-001",
                "total": 125000,
            },
        },
    ]
    mocker.patch("src.runner.load_dataset", return_value=dataset_rows)
    llm_client = mocker.Mock()
    llm_client.generate_text.side_effect = LLMAuthError(
        "Invalid LLM API credentials"
    )
    mocker.patch("src.runner.LLMClient", return_value=llm_client)
    report_generator = mocker.Mock()
    mocker.patch("src.runner.ReportGenerator", return_value=report_generator)

    runner = BenchmarkRunner(
        dataset_path="benchmark.json",
        model="benchmark-model",
    )
    summary = runner.run()

    assert summary == {
        "status": "aborted_due_to_auth",
        "total_rows": 1,
        "successful_evaluations": 0,
        "average_score": pytest.approx(0.0),
    }
    report_generator.append_row.assert_not_called()


def test_benchmark_runner_aborts_safely_when_authentication_fails(
    mocker,
    capsys,
):
    dataset_rows = [
        {
            "prompt": "Extract invoice INV-001",
            "expected_output": {
                "invoice_id": "INV-001",
                "total": 125000,
            },
        },
        {
            "prompt": "Extract invoice INV-002",
            "expected_output": {
                "invoice_id": "INV-002",
                "total": 75000,
            },
        },
        {
            "prompt": "Extract invoice INV-003",
            "expected_output": {
                "invoice_id": "INV-003",
                "total": 50000,
            },
        },
    ]
    mocker.patch("src.runner.load_dataset", return_value=dataset_rows)
    llm_client = mocker.Mock()
    llm_client.generate_text.side_effect = [
        '{"invoice_id": "INV-001", "total": 125000}',
        LLMAuthError("Invalid LLM API credentials"),
    ]
    mocker.patch("src.runner.LLMClient", return_value=llm_client)
    report_generator = mocker.Mock()
    mocker.patch("src.runner.ReportGenerator", return_value=report_generator)

    runner = BenchmarkRunner(
        dataset_path="benchmark.json",
        model="benchmark-model",
    )
    summary = runner.run()

    captured = capsys.readouterr()

    assert set(summary) == EXPECTED_SUMMARY_KEYS
    assert summary == {
        "status": "aborted_due_to_auth",
        "total_rows": 2,
        "successful_evaluations": 1,
        "average_score": pytest.approx(1.0),
    }
    assert "Invalid LLM API credentials" in captured.out
    assert llm_client.generate_text.call_count == 2
    report_generator.append_row.assert_called_once_with(
        {
            "row_index": 1,
            "status": "completed",
            "score": pytest.approx(1.0),
            "llm_output": '{"invoice_id": "INV-001", "total": 125000}',
            "error_message": "",
        }
    )
    report_generator.finalize_report.assert_not_called()


def test_benchmark_runner_returns_empty_status_when_report_has_no_data(
    mocker,
):
    mocker.patch("src.runner.load_dataset", return_value=[])
    llm_client = mocker.Mock()
    mocker.patch("src.runner.LLMClient", return_value=llm_client)
    report_generator = mocker.Mock()
    report_generator.finalize_report.side_effect = EmptyDataError(
        "No evaluation data was written to the report."
    )
    mocker.patch("src.runner.ReportGenerator", return_value=report_generator)

    runner = BenchmarkRunner(
        dataset_path="empty-benchmark.json",
        model="benchmark-model",
    )
    summary = runner.run()

    assert set(summary) == EXPECTED_SUMMARY_KEYS
    assert summary == {
        "status": "empty",
        "total_rows": 0,
        "successful_evaluations": 0,
        "average_score": pytest.approx(0.0),
    }
    llm_client.generate_text.assert_not_called()
    report_generator.append_row.assert_not_called()
    report_generator.finalize_report.assert_called_once_with()


def test_runner_integration_mixed_results(mocker, tmp_path):
    dataset_rows = [
        {
            "prompt": "Extract invoice INV-001",
            "expected_output": {
                "invoice_id": "INV-001",
                "total": 125000,
            },
        },
        {
            "prompt": "Extract invoice INV-002",
            "expected_output": {
                "invoice_id": "INV-002",
                "total": 75000,
            },
        },
        {
            "prompt": "Extract invoice INV-003",
            "expected_output": {
                "invoice_id": "INV-003",
                "total": 50000,
            },
        },
    ]
    report_path = tmp_path / "mixed-results.csv"
    mocker.patch("src.runner.load_dataset", return_value=dataset_rows)
    llm_client = mocker.Mock()
    llm_client.generate_text.side_effect = [
        '{"invoice_id": "INV-001", "total": 125000}',
        '{"invoice_id": "INV-002", "total": 99999}',
        RuntimeError("Provider timed out"),
    ]
    mocker.patch("src.runner.LLMClient", return_value=llm_client)

    runner = BenchmarkRunner(
        dataset_path="benchmark.json",
        model="benchmark-model",
        report_path=report_path,
    )
    summary = runner.run()

    with report_path.open("r", newline="", encoding="utf-8") as report_file:
        report_rows = list(csv.DictReader(report_file))

    assert summary == {
        "status": "completed",
        "total_rows": 3,
        "successful_evaluations": 2,
        "average_score": pytest.approx(0.5),
    }
    assert [row["status"] for row in report_rows] == [
        "completed",
        "completed",
        "failed",
    ]
    assert [float(row["score"]) for row in report_rows] == pytest.approx(
        [1.0, 0.5, 0.0]
    )
    assert report_rows[2]["error_message"] == "Provider timed out"
