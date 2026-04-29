import csv
import json
from pathlib import Path

import pytest

import src.main as main_module


def _write_ground_truth(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "unit,item,num_type,status_type,value",
                "SDM,Seragam Pegawai,cost,target,2000000",
                "SDM,Seragam Pegawai,cost,actual,1800000",
            ]
        ),
        encoding="utf-8",
    )


def test_discover_example_cases_returns_supported_pairs_and_skips_gaps(
    tmp_path,
):
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    supported_input = examples_dir / "ex01_input.csv"
    supported_output = examples_dir / "ex01_output.csv"
    unsupported_input = examples_dir / "ex02_input.gif"
    unsupported_output = examples_dir / "ex02_gif_output.csv"
    missing_output_input = examples_dir / "ex03_input.txt"

    supported_input.write_text("source", encoding="utf-8")
    _write_ground_truth(supported_output)
    unsupported_input.write_text("unsupported image bytes", encoding="utf-8")
    _write_ground_truth(unsupported_output)
    missing_output_input.write_text("source", encoding="utf-8")

    cases, skipped_cases = main_module._discover_example_cases([examples_dir])

    assert cases == [
        main_module.ExampleCase(
            case_id="examples__ex01",
            input_path=supported_input,
            output_path=supported_output,
        )
    ]
    assert [skipped.reason for skipped in skipped_cases] == [
        "unsupported input format .gif",
        "matching *_output.csv file was not found",
    ]


def test_run_single_case_writes_runtime_dataset_and_invokes_runner(
    tmp_path,
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    examples_dir = tmp_path / "examples"
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    examples_dir.mkdir()
    report_dir.mkdir()
    runtime_dir.mkdir()
    input_path = examples_dir / "ex01_input.csv"
    output_path = examples_dir / "ex01_output.csv"
    input_path.write_text(
        "Divisi,Nama Barang,Tipe,Anggaran,Realisasi\n"
        "SDM,Seragam Pegawai,Biaya,2000000,1800000\n",
        encoding="utf-8",
    )
    _write_ground_truth(output_path)
    runner_class = mocker.patch("src.main.BenchmarkRunner")
    runner = runner_class.return_value
    runner.run.return_value = {
        "status": "completed",
        "total_rows": 1,
        "successful_evaluations": 1,
        "average_score": 1.0,
    }
    mocker.patch("src.main.time.perf_counter", side_effect=[10.0, 12.5])

    result = main_module._run_single_case(
        example_case=main_module.ExampleCase(
            case_id="examples__ex01",
            input_path=input_path,
            output_path=output_path,
        ),
        model_name="deepseek/v3:2",
        report_dir=report_dir,
        runtime_dataset_dir=runtime_dir,
    )

    runtime_dataset = runtime_dir / "deepseek_v3_2__examples__ex01.json"
    report_path = report_dir / "deepseek_v3_2__examples__ex01.csv"
    payload = json.loads(runtime_dataset.read_text(encoding="utf-8"))

    runner_class.assert_called_once_with(
        dataset_path=runtime_dataset,
        model="deepseek/v3:2",
        report_path=report_path,
    )
    assert input_path.read_text().strip() in payload["rows"][0]["prompt"]
    assert payload["rows"][0]["expected_output"]["content_data"][0]["rows"][0] == {
        "unit": "SDM",
        "item": "Seragam Pegawai",
        "num_type": "cost",
        "status_type": "target",
        "value": 2000000,
    }
    assert result == {
        "model": "deepseek/v3:2",
        "case_id": "examples__ex01",
        "input_path": str(Path("examples") / "ex01_input.csv"),
        "output_path": str(Path("examples") / "ex01_output.csv"),
        "report_path": str(
            Path("reports") / "deepseek_v3_2__examples__ex01.csv"
        ),
        "elapsed_seconds": 2.5,
        "status": "completed",
        "total_rows": 1,
        "successful_evaluations": 1,
        "average_score": 1.0,
    }


def test_project_output_path_from_env_rejects_path_traversal(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("REPORT_DIR", "../poc_outside_reports")

    with pytest.raises(ValueError, match="REPORT_DIR must stay inside"):
        main_module._project_output_path_from_env(
            "REPORT_DIR",
            main_module.DEFAULT_REPORT_DIR,
        )


def test_safe_output_path_sanitizes_user_controlled_filename(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    output_dir = tmp_path / "reports"

    safe_path = main_module._safe_output_path(output_dir, "../..//evil.json")

    assert safe_path == output_dir.resolve() / "evil.json"
    assert safe_path.is_relative_to(output_dir.resolve())
    assert main_module._safe_filename("..") == "unnamed"


def test_safe_output_path_rejects_paths_that_escape_base_dir(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        main_module,
        "_safe_filename",
        lambda value: "../evil.json",
    )

    with pytest.raises(ValueError, match="escapes the output directory"):
        main_module._safe_output_path(output_dir, "evil.json")


def test_main_runs_configured_models_and_example_dirs(
    tmp_path,
    monkeypatch,
    mocker,
    capsys,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MODEL_NAMES", "model-a, model-b")
    monkeypatch.setenv("EXAMPLE_DIRS", "examples")
    monkeypatch.setenv("REPORT_DIR", "reports")
    monkeypatch.setenv("BENCHMARK_DATASET_DIR", "runtime")
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    input_path = examples_dir / "case01_input.txt"
    output_path = examples_dir / "case01_output.csv"
    skipped_input_path = examples_dir / "case02_input.gif"
    input_path.write_text("plain source text", encoding="utf-8")
    skipped_input_path.write_text("unsupported image source", encoding="utf-8")
    _write_ground_truth(output_path)
    runner_class = mocker.patch("src.main.BenchmarkRunner")
    runner = runner_class.return_value
    runner.run.return_value = {
        "status": "completed",
        "total_rows": 1,
        "successful_evaluations": 1,
        "average_score": 0.75,
    }

    exit_code = main_module.main()

    captured = capsys.readouterr()

    assert exit_code == 0
    assert runner_class.call_count == 2
    assert "Models     : model-a, model-b" in captured.out
    assert "Examples   : 1 supported" in captured.out
    assert "Skipped files" in captured.out
    assert "case02_input.gif: unsupported input format .gif" in captured.out
    assert "Overall report:" in captured.out
    assert (tmp_path / "reports").is_dir()
    assert (tmp_path / "runtime").is_dir()
    assert (tmp_path / "reports" / "overall_benchmark_report.csv").is_file()


def test_write_overall_report_compares_files_and_models(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    batch_results = [
        {
            "model": "model-a",
            "case_id": "file-1",
            "status": "completed",
            "average_score": 0.9,
            "elapsed_seconds": 5.0,
            "total_rows": 1,
            "successful_evaluations": 1,
        },
        {
            "model": "model-b",
            "case_id": "file-1",
            "status": "completed",
            "average_score": 1.0,
            "elapsed_seconds": 8.0,
            "total_rows": 1,
            "successful_evaluations": 1,
        },
        {
            "model": "model-a",
            "case_id": "file-2",
            "status": "completed",
            "average_score": 0.4,
            "elapsed_seconds": 3.0,
            "total_rows": 1,
            "successful_evaluations": 1,
        },
        {
            "model": "model-b",
            "case_id": "file-2",
            "status": "completed",
            "average_score": 0.2,
            "elapsed_seconds": 2.0,
            "total_rows": 1,
            "successful_evaluations": 1,
        },
    ]

    report_path = main_module._write_overall_report(batch_results, tmp_path)

    with report_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    case_rows = {
        row["case_id"]: row
        for row in rows
        if row["section"] == "case_comparison"
    }
    model_rows = {
        row["model"]: row
        for row in rows
        if row["section"] == "model_summary"
    }
    overall_rows = {row["section"]: row for row in rows if row["case_id"] == ""}

    assert report_path == tmp_path / "overall_benchmark_report.csv"
    assert case_rows["file-1"]["best_model"] == "model-b"
    assert case_rows["file-1"]["fastest_model"] == "model-a"
    assert case_rows["file-2"]["best_model"] == "model-a"
    assert case_rows["file-2"]["fastest_model"] == "model-b"
    assert float(model_rows["model-a"]["average_score"]) == pytest.approx(0.65)
    assert float(model_rows["model-a"]["average_seconds"]) == pytest.approx(4.0)
    assert float(model_rows["model-b"]["average_score"]) == pytest.approx(0.6)
    assert float(model_rows["model-b"]["average_seconds"]) == pytest.approx(5.0)
    assert overall_rows["overall_best_average"]["model"] == "model-a"
    assert overall_rows["overall_fastest"]["model"] == "model-a"


def test_default_env_helpers_and_path_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("MODEL_NAMES", raising=False)
    monkeypatch.delenv("EXAMPLE_DIRS", raising=False)

    model_names = main_module._model_names_from_env()
    example_dirs = main_module._example_dirs_from_env()
    absolute_path = main_module._project_path(tmp_path / "already-absolute")

    assert model_names == list(main_module.DEFAULT_MODEL_NAMES)
    assert example_dirs == [
        tmp_path / example_dir for example_dir in main_module.DEFAULT_EXAMPLE_DIRS
    ]
    assert absolute_path == tmp_path / "already-absolute"
    assert main_module.DEFAULT_REPORT_DIR == "data/benchmark_reports"


def test_display_path_prefers_project_relative_and_falls_back_to_absolute(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)

    inside_project = tmp_path / "nested" / "file.csv"
    outside_project = Path("D:/external/file.csv")

    assert main_module._display_path(inside_project) == str(
        Path("nested") / "file.csv"
    )
    assert main_module._display_path(outside_project) == str(outside_project)


def test_source_type_handles_pdf_and_excel_labels():
    assert main_module._source_type(Path("invoice.pdf")) == "PDF"
    assert main_module._source_type(Path("invoice.png")) == "PDF"
    assert main_module._source_type(Path("invoice.xlsx")) == "Excel"


def test_overall_helpers_handle_empty_inputs(capsys, tmp_path):
    assert main_module._overall_winner_rows([]) == []
    assert main_module._average_float([]) == 0.0

    main_module._print_batch_summary([], tmp_path / "overall.csv")
    captured = capsys.readouterr()

    assert "No supported benchmark cases were found." in captured.out
