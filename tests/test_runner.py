import pytest

from src.llm_client import LLMAuthError
from src.runner import BenchmarkRunner


def test_benchmark_runner_returns_average_score_for_successful_pipeline(
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

    runner = BenchmarkRunner(
        dataset_path="benchmark.json",
        model="benchmark-model",
    )
    summary = runner.run()

    assert summary == {
        "status": "completed",
        "total_rows": 2,
        "successful_evaluations": 2,
        "average_score": pytest.approx(0.75),
    }
    load_dataset_mock.assert_called_once()
    assert load_dataset_mock.call_args.args[0] == "benchmark.json"
    llm_client_class.assert_called_once_with(model="benchmark-model")
    llm_client.generate_text.assert_has_calls(
        [
            mocker.call("Extract invoice INV-001"),
            mocker.call("Extract invoice INV-002"),
        ]
    )


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

    runner = BenchmarkRunner(
        dataset_path="benchmark.json",
        model="benchmark-model",
    )
    summary = runner.run()

    captured = capsys.readouterr()

    assert summary == {
        "status": "failed",
        "total_rows": 3,
        "successful_evaluations": 1,
        "average_score": pytest.approx(1.0),
        "error": "Invalid LLM API credentials",
    }
    assert "Invalid LLM API credentials" in captured.out
    assert llm_client.generate_text.call_count == 2
