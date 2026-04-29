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
