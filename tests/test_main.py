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

    cases, skipped_cases = main_module._discover_example_cases(
        [examples_dir],
        {".csv", ".txt", ".docx", ".pdf", ".png", ".xlsx"},
    )

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
    assert "Text report" in captured.out
    assert (tmp_path / "reports").is_dir()
    assert (tmp_path / "runtime").is_dir()
    assert (tmp_path / "reports" / "overall_benchmark_report.csv").is_file()
    assert (tmp_path / "reports" / "overall_benchmark_report.txt").is_file()


def test_main_uses_merge_path_when_merge_base_report_is_set(
    tmp_path,
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MERGE_BASE_OVERALL_REPORT", "data/base_overall.csv")
    discover_cases = mocker.patch(
        "src.main._discover_example_cases",
        return_value=([], []),
    )
    run_batch = mocker.patch("src.main._run_batch", return_value=[])
    write_merged = mocker.patch(
        "src.main._write_overall_report_with_merge",
        return_value=tmp_path / "reports" / "overall_benchmark_report.csv",
    )
    write_text = mocker.patch(
        "src.main._write_overall_text_report",
        return_value=tmp_path / "reports" / "overall_benchmark_report.txt",
    )
    print_summary = mocker.patch("src.main._print_batch_summary")

    exit_code = main_module.main()

    assert exit_code == 0
    discover_cases.assert_called_once()
    run_batch.assert_called_once()
    write_merged.assert_called_once()
    write_text.assert_called_once()
    print_summary.assert_called_once()


def test_run_batch_continues_when_single_case_preprocess_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    report_dir.mkdir()
    runtime_dir.mkdir()

    case_a = main_module.ExampleCase(
        case_id="examples__ex01",
        input_path=tmp_path / "examples" / "ex01_input.png",
        output_path=tmp_path / "examples" / "ex01_output.csv",
    )
    case_b = main_module.ExampleCase(
        case_id="examples__ex02",
        input_path=tmp_path / "examples" / "ex02_input.csv",
        output_path=tmp_path / "examples" / "ex02_output.csv",
    )

    def fake_run_single_case(*, example_case, **kwargs):
        if example_case.case_id == "examples__ex01":
            raise ValueError("Tesseract OCR is not available")
        return {
            "model": "model-a",
            "case_id": example_case.case_id,
            "input_path": str(example_case.input_path),
            "output_path": str(example_case.output_path),
            "report_path": "reports/mock.csv",
            "elapsed_seconds": 1.0,
            "status": "completed",
            "total_rows": 1,
            "successful_evaluations": 1,
            "average_score": 1.0,
        }

    monkeypatch.setattr(main_module, "_run_single_case", fake_run_single_case)

    results = main_module._run_batch(
        cases=[case_a, case_b],
        model_names=["model-a"],
        report_dir=report_dir,
        runtime_dataset_dir=runtime_dir,
    )

    assert len(results) == 2
    assert results[0]["case_id"] == "examples__ex01"
    assert results[0]["status"] == "failed_preprocess"
    assert results[0]["average_score"] == 0.0
    assert "Tesseract OCR is not available" in results[0]["error_message"]
    assert results[1]["case_id"] == "examples__ex02"
    assert results[1]["status"] == "completed"


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


def test_write_overall_text_report_contains_readable_tables(tmp_path, monkeypatch):
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
    ]

    report_path = main_module._write_overall_text_report(batch_results, tmp_path)
    content = report_path.read_text(encoding="utf-8")

    assert report_path == tmp_path / "overall_benchmark_report.txt"
    assert "LLM Benchmark Report" in content
    assert "Run Results" in content
    assert "Case Comparison" in content
    assert "Model Summary" in content
    assert "Overall Winners" in content
    assert "| Model" in content
    assert "+-" in content


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
    assert main_module.DEFAULT_REPORT_DIR == "benchmark_reports"


def test_example_dirs_from_env_rejects_path_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    outside_project = tmp_path.parent / "outside_examples"
    monkeypatch.setenv("EXAMPLE_DIRS", str(outside_project))

    with pytest.raises(ValueError, match="EXAMPLE_DIRS"):
        main_module._example_dirs_from_env()


def test_read_ground_truth_rows_rejects_path_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="ground truth csv path"):
        main_module._read_ground_truth_rows(Path("D:/outside/output.csv"))


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
    display = main_module._display_path(outside_project)
    # On different platforms `relpath` may return a relative path that still
    # includes the original drive-like component; accept either the absolute
    # original or any representation that contains the filename.
    assert outside_project.name in display


def test_display_path_returns_original_path_when_relpath_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main_module.os.path,
        "relpath",
        lambda path, start: (_ for _ in ()).throw(ValueError("drive mismatch")),
    )
    outside_project = Path("D:/external/file.csv")

    assert main_module._display_path(outside_project) == str(outside_project)


def test_source_type_handles_pdf_and_excel_labels():
    assert main_module._source_type(Path("invoice.pdf")) == "PDF"
    assert main_module._source_type(Path("invoice.png")) == "PDF"
    assert main_module._source_type(Path("invoice.xlsx")) == "Excel"


def test_input_extensions_from_env_normalizes_values(monkeypatch):
    monkeypatch.setenv("INPUT_EXTENSIONS", "png, .PDF")

    extensions = main_module._input_extensions_from_env()

    assert extensions == {".png", ".pdf"}


def test_input_extensions_from_env_rejects_unsupported_extension(monkeypatch):
    monkeypatch.setenv("INPUT_EXTENSIONS", ".png,.bad")

    with pytest.raises(ValueError, match="Unsupported extension"):
        main_module._input_extensions_from_env()


def test_input_extensions_from_env_rejects_dot_only_token(monkeypatch):
    monkeypatch.setenv("INPUT_EXTENSIONS", ".")

    with pytest.raises(ValueError, match="invalid extension token"):
        main_module._input_extensions_from_env()


def test_merge_base_overall_report_from_env_rejects_path_traversal(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MERGE_BASE_OVERALL_REPORT", "../outside.csv")

    with pytest.raises(ValueError, match="MERGE_BASE_OVERALL_REPORT must stay inside"):
        main_module._merge_base_overall_report_from_env()


def test_merge_base_overall_report_from_env_none_when_unset(monkeypatch):
    monkeypatch.delenv("MERGE_BASE_OVERALL_REPORT", raising=False)

    assert main_module._merge_base_overall_report_from_env() is None


def test_merge_base_overall_report_from_env_resolves_valid_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "reports" / "overall.csv"
    monkeypatch.setenv(
        "MERGE_BASE_OVERALL_REPORT",
        str(path.relative_to(tmp_path)),
    )

    assert main_module._merge_base_overall_report_from_env() == path


def test_write_overall_report_with_merge_combines_previous_and_new_data(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    base_overall_path = tmp_path / "base_overall.csv"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    base_overall_path.write_text(
        "\n".join(
            [
                ",".join(main_module.OVERALL_REPORT_FIELDNAMES),
                (
                    "case_comparison,legacy_case,,model-a,0.7,model-a,3.0,,,"
                    "4,4"
                ),
                "model_summary,,model-a,,,,,0.7,3.0,4,4",
                "overall_best_average,,model-a,,,,,0.7,3.0,4,4",
                "overall_fastest,,model-a,,,,,0.7,3.0,4,4",
            ]
        )
        + "\n",
        encoding="utf-8-sig",
    )
    batch_results = [
        {
            "model": "model-a",
            "case_id": "new_case",
            "status": "completed",
            "average_score": 1.0,
            "elapsed_seconds": 2.0,
            "total_rows": 1,
            "successful_evaluations": 1,
        },
        {
            "model": "model-b",
            "case_id": "new_case",
            "status": "completed",
            "average_score": 0.5,
            "elapsed_seconds": 1.0,
            "total_rows": 1,
            "successful_evaluations": 1,
        },
    ]

    merged_path = main_module._write_overall_report_with_merge(
        batch_results,
        report_dir,
        base_overall_path,
    )

    with merged_path.open("r", newline="", encoding="utf-8-sig") as file:
        merged_rows = list(csv.DictReader(file))

    case_ids = [
        row["case_id"]
        for row in merged_rows
        if row["section"] == "case_comparison"
    ]
    model_rows = {
        row["model"]: row
        for row in merged_rows
        if row["section"] == "model_summary"
    }

    assert "legacy_case" in case_ids
    assert "new_case" in case_ids
    assert float(model_rows["model-a"]["average_score"]) == pytest.approx(0.76)
    assert float(model_rows["model-a"]["average_seconds"]) == pytest.approx(2.8)
    assert int(model_rows["model-a"]["total_runs"]) == 5


def test_write_overall_report_with_merge_falls_back_when_base_missing(
    tmp_path,
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    batch_results = []
    write_overall_report = mocker.spy(main_module, "_write_overall_report")

    merged_path = main_module._write_overall_report_with_merge(
        batch_results,
        report_dir,
        tmp_path / "missing.csv",
    )

    write_overall_report.assert_called_once_with(batch_results, report_dir)
    assert merged_path == report_dir / "overall_benchmark_report.csv"


def test_merge_model_summary_rows_keeps_base_when_new_missing():
    base_rows = {
        "model-a": {
            "section": "model_summary",
            "case_id": "",
            "model": "model-a",
            "average_score": 0.5,
            "average_seconds": 10.0,
            "total_runs": 4,
            "completed_runs": 4,
        }
    }

    merged = main_module._merge_model_summary_rows(base_rows, {})

    assert merged["model-a"]["model"] == "model-a"
    assert merged["model-a"]["total_runs"] == 4


def test_merge_model_summary_rows_handles_zero_total_runs():
    base_rows = {
        "model-a": {
            "section": "model_summary",
            "case_id": "",
            "model": "model-a",
            "average_score": 1.0,
            "average_seconds": 1.0,
            "total_runs": 0,
            "completed_runs": 0,
        }
    }
    new_rows = {
        "model-a": {
            "section": "model_summary",
            "case_id": "",
            "model": "model-a",
            "average_score": 1.0,
            "average_seconds": 1.0,
            "total_runs": 0,
            "completed_runs": 0,
        }
    }

    merged = main_module._merge_model_summary_rows(base_rows, new_rows)

    assert merged["model-a"]["average_score"] == 0.0
    assert merged["model-a"]["average_seconds"] == 0.0


def test_winner_rows_from_model_summaries_returns_ranked_rows():
    model_rows = [
        {
            "model": "model-a",
            "average_score": 0.5,
            "average_seconds": 5.0,
            "total_runs": 4,
            "completed_runs": 4,
        },
        {
            "model": "model-b",
            "average_score": 0.7,
            "average_seconds": 7.0,
            "total_runs": 4,
            "completed_runs": 4,
        },
    ]

    winners = main_module._winner_rows_from_model_summaries(model_rows)

    assert winners[0]["section"] == "overall_best_average"
    assert winners[0]["model"] == "model-b"
    assert winners[1]["section"] == "overall_fastest"
    assert winners[1]["model"] == "model-a"


def test_winner_rows_from_model_summaries_returns_empty_for_no_models():
    assert main_module._winner_rows_from_model_summaries([]) == []


def test_format_table_expands_column_widths():
    table = main_module._format_table(
        ["A", "B"],
        [["1", "long value"]],
    )

    assert "+-" in table
    assert "long value" in table
    assert table.strip().endswith("+")


def test_write_overall_text_report_reads_full_overall_csv(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    overall_path = report_dir / "overall_benchmark_report.csv"
    overall_path.write_text(
        "\n".join(
            [
                ",".join(main_module.OVERALL_REPORT_FIELDNAMES),
                "case_comparison,case-1,,model-a,1.0,model-a,2.0,,,4,4",
                "model_summary,,model-a,,,,,0.5,3.0,4,4",
                "overall_best_average,,model-a,,,,,0.5,3.0,4,4",
                "overall_fastest,,model-a,,,,,0.5,3.0,4,4",
            ]
        )
        + "\n",
        encoding="utf-8-sig",
    )

    txt_path = main_module._write_overall_text_report(
        batch_results=[],
        report_dir=report_dir,
        overall_report_path=overall_path,
    )
    content = txt_path.read_text(encoding="utf-8")

    assert "case-1" in content
    assert "model-a" in content


def test_write_overall_text_report_handles_empty_overall_sections(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    overall_path = report_dir / "overall_benchmark_report.csv"
    overall_path.write_text(
        ",".join(main_module.OVERALL_REPORT_FIELDNAMES) + "\n",
        encoding="utf-8-sig",
    )

    txt_path = main_module._write_overall_text_report(
        batch_results=[],
        report_dir=report_dir,
        overall_report_path=overall_path,
    )
    content = txt_path.read_text(encoding="utf-8")

    assert "No case comparison data." in content
    assert "No model summary data." in content
    assert "No overall winner data." in content


def test_overall_row_extractors_handle_missing_sections():
    rows = [{"section": "other", "case_id": "x"}]

    assert main_module._case_rows_from_overall_rows(rows) == []
    assert main_module._model_rows_from_overall_rows(rows) == []
    assert main_module._winner_rows_from_overall_rows(rows) == []
    assert main_module._case_rows_from_overall_rows(None) == []
    assert main_module._model_rows_from_overall_rows(None) == []
    assert main_module._winner_rows_from_overall_rows(None) == []


def test_overall_helpers_handle_empty_inputs(capsys, tmp_path):
    assert main_module._overall_winner_rows([]) == []
    assert main_module._average_float([]) == 0.0

    main_module._print_batch_summary(
        [],
        tmp_path / "overall.csv",
        tmp_path / "overall.txt",
    )
    captured = capsys.readouterr()

    assert "No supported benchmark cases were found." in captured.out
