import csv
import json
from pathlib import Path

import pytest

import src.category_accuracy_report as report_module


def _write_runtime_dataset(path: Path, expected_output: object) -> None:
    payload = {
        "rows": [
            {
                "prompt": "prompt",
                "expected_output": expected_output,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_report_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["row_index", "status", "score", "llm_output", "error_message"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_generate_category_accuracy_reports_writes_json_and_txt(tmp_path):
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    output_dir = tmp_path / "out"
    report_dir.mkdir()
    runtime_dir.mkdir()
    _write_report_csv(
        report_dir / "model-a__source-x__case01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"id":"A"}',
                "error_message": "",
            }
        ],
    )
    _write_runtime_dataset(
        runtime_dir / "model-a__source-x__case01.json",
        {
            "document_info": {"source_type": "Excel"},
            "id": "A",
        },
    )

    result = report_module.generate_category_accuracy_reports(
        report_dir=report_dir,
        runtime_dir=runtime_dir,
        output_dir=output_dir,
    )

    assert result["json_path"].is_file()
    assert result["txt_path"].is_file()
    assert result["total_evaluations"] == 1
    payload = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert "overall" in payload
    assert "by_model" in payload
    assert "by_source" in payload


def test_collect_evaluations_applies_source_and_model_filters(tmp_path):
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    report_dir.mkdir()
    runtime_dir.mkdir()
    _write_report_csv(
        report_dir / "model-a__source-x__case01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"id":"A"}',
                "error_message": "",
            }
        ],
    )
    _write_report_csv(
        report_dir / "model-b__source-y__case02.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"id":"B"}',
                "error_message": "",
            }
        ],
    )
    _write_runtime_dataset(
        runtime_dir / "model-a__source-x__case01.json",
        {"document_info": {"source_type": "Excel"}},
    )
    _write_runtime_dataset(
        runtime_dir / "model-b__source-y__case02.json",
        {"document_info": {"source_type": "PDF"}},
    )

    evaluations = report_module._collect_evaluations(
        report_dir=report_dir,
        runtime_dir=runtime_dir,
        source_filter=" source-x ",
        model_filter=" model-a ",
    )

    assert len(evaluations) == 1
    assert evaluations[0]["model"] == "model-a"
    assert evaluations[0]["source"] == "source-x"
    assert evaluations[0]["category"] == "Excel"


def test_collect_evaluations_uses_filename_extension_as_category(tmp_path):
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    report_dir.mkdir()
    runtime_dir.mkdir()
    _write_report_csv(
        report_dir / "model-a__source-x__case01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"id":"A"}',
                "error_message": "",
            }
        ],
    )
    _write_runtime_dataset(
        runtime_dir / "model-a__source-x__case01.json",
        {
            "document_info": {
                "source_type": "Excel",
                "filename": "sample_input.txt",
            },
            "id": "A",
        },
    )

    evaluations = report_module._collect_evaluations(
        report_dir=report_dir,
        runtime_dir=runtime_dir,
        source_filter="",
        model_filter="",
    )

    assert len(evaluations) == 1
    assert evaluations[0]["category"] == "txt"


def test_collect_evaluations_skips_rows_when_model_filter_does_not_match(tmp_path):
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    report_dir.mkdir()
    runtime_dir.mkdir()
    _write_report_csv(
        report_dir / "model-a__source-x__case01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"id":"A"}',
                "error_message": "",
            }
        ],
    )
    _write_runtime_dataset(
        runtime_dir / "model-a__source-x__case01.json",
        {"document_info": {"source_type": "Excel"}},
    )

    evaluations = report_module._collect_evaluations(
        report_dir=report_dir,
        runtime_dir=runtime_dir,
        source_filter="",
        model_filter="model-b",
    )

    assert evaluations == []


def test_collect_evaluations_skips_invalid_report_files_and_missing_runtime(tmp_path):
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    report_dir.mkdir()
    runtime_dir.mkdir()
    _write_report_csv(
        report_dir / "overall_something.csv",
        [],
    )
    _write_report_csv(
        report_dir / "malformed.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": "{}",
                "error_message": "",
            }
        ],
    )
    _write_report_csv(
        report_dir / "model-a__source-x__case01.csv",
        [
            {
                "row_index": "1",
                "status": "failed",
                "score": "0.0",
                "llm_output": "{}",
                "error_message": "x",
            }
        ],
    )

    evaluations = report_module._collect_evaluations(
        report_dir=report_dir,
        runtime_dir=runtime_dir,
        source_filter="",
        model_filter="",
    )

    assert evaluations == []


def test_collect_evaluations_skips_non_completed_rows_even_when_runtime_exists(tmp_path):
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    report_dir.mkdir()
    runtime_dir.mkdir()
    _write_report_csv(
        report_dir / "model-a__source-x__case01.csv",
        [
            {
                "row_index": "1",
                "status": "failed",
                "score": "0.0",
                "llm_output": '{"id":"A"}',
                "error_message": "x",
            }
        ],
    )
    _write_runtime_dataset(
        runtime_dir / "model-a__source-x__case01.json",
        {"document_info": {"source_type": "Excel"}},
    )

    evaluations = report_module._collect_evaluations(
        report_dir=report_dir,
        runtime_dir=runtime_dir,
        source_filter="",
        model_filter="",
    )

    assert evaluations == []


def test_parse_report_filename_requires_model_and_source():
    assert report_module._parse_report_filename("malformed") is None
    assert report_module._parse_report_filename("model__source") == (
        "model",
        "source",
    )


def test_resolve_report_context_applies_filename_and_filters():
    assert (
        report_module._resolve_report_context(
            report_path=Path("overall_report.csv"),
            source_filter="",
            model_filter="",
        )
        is None
    )
    assert (
        report_module._resolve_report_context(
            report_path=Path("bad.csv"),
            source_filter="",
            model_filter="",
        )
        is None
    )
    assert (
        report_module._resolve_report_context(
            report_path=Path("model-a__source-x__case.csv"),
            source_filter="source-y",
            model_filter="",
        )
        is None
    )
    assert (
        report_module._resolve_report_context(
            report_path=Path("model-a__source-x__case.csv"),
            source_filter="",
            model_filter="model-b",
        )
        is None
    )
    assert report_module._resolve_report_context(
        report_path=Path("model-a__source-x__case.csv"),
        source_filter="source-x",
        model_filter="model-a",
    ) == ("model-a", "source-x")


def test_completed_rows_to_evaluations_keeps_completed_only(tmp_path):
    report_path = tmp_path / "model-a__source-x__case.csv"
    _write_report_csv(
        report_path,
        [
            {
                "row_index": "1",
                "status": "failed",
                "score": "0.0",
                "llm_output": '{"id":"X"}',
                "error_message": "x",
            },
            {
                "row_index": "2",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"id":"A"}',
                "error_message": "",
            },
        ],
    )

    evaluations = report_module._completed_rows_to_evaluations(
        report_path=report_path,
        model_name="model-a",
        source_name="source-x",
        category="Excel",
        ground_truth={"id": "A"},
    )

    assert evaluations == [
        {
            "category": "Excel",
            "model": "model-a",
            "source": "source-x",
            "ground_truth": {"id": "A"},
            "llm_output": '{"id":"A"}',
        }
    ]


def test_read_ground_truth_handles_missing_rows_and_non_dict_payload(tmp_path):
    no_rows = tmp_path / "no_rows.json"
    no_rows.write_text(json.dumps({"rows": []}), encoding="utf-8")
    assert report_module._read_ground_truth(no_rows) == {}

    non_dict = tmp_path / "non_dict.json"
    non_dict.write_text(
        json.dumps({"rows": [{"expected_output": ["bad"]}]}),
        encoding="utf-8",
    )
    assert report_module._read_ground_truth(non_dict) == {}


def test_read_ground_truth_handles_malformed_json_and_non_object_root(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json}", encoding="utf-8")
    assert report_module._read_ground_truth(malformed) == {}

    non_object_root = tmp_path / "non_object_root.json"
    non_object_root.write_text(json.dumps(["bad"]), encoding="utf-8")
    assert report_module._read_ground_truth(non_object_root) == {}


def test_read_ground_truth_handles_non_list_rows_and_non_dict_first_row(tmp_path):
    rows_not_list = tmp_path / "rows_not_list.json"
    rows_not_list.write_text(json.dumps({"rows": "bad"}), encoding="utf-8")
    assert report_module._read_ground_truth(rows_not_list) == {}

    first_row_not_dict = tmp_path / "first_row_not_dict.json"
    first_row_not_dict.write_text(json.dumps({"rows": ["bad"]}), encoding="utf-8")
    assert report_module._read_ground_truth(first_row_not_dict) == {}


def test_extract_category_handles_missing_document_info():
    assert report_module._extract_category({}) == "unknown"
    assert report_module._extract_category({"document_info": []}) == "unknown"
    assert report_module._extract_category(
        {"document_info": {"source_type": "PDF"}}
    ) == "PDF"


def test_extract_category_prefers_filename_extension_when_available():
    assert report_module._extract_category(
        {
            "document_info": {
                "source_type": "Excel",
                "filename": "invoice_input.csv",
            }
        }
    ) == "csv"
    assert report_module._extract_category(
        {
            "document_info": {
                "source_type": "PDF",
                "filename": "scan_document.png",
            }
        }
    ) == "png"
    assert report_module._extract_category(
        {
            "document_info": {
                "source_type": "Excel",
                "filename": "raw_source.txt",
            }
        }
    ) == "txt"


def test_suffix_and_safe_label_cover_empty_and_sanitized_cases():
    assert report_module._suffix_from_filters("", "") == ""
    assert report_module._suffix_from_filters("source a", "") == "_source_a"
    assert report_module._suffix_from_filters("", "model/a") == "_model_a"
    assert report_module._suffix_from_filters("s", "m") == "_s_m"
    assert report_module._safe_label("!!!") == "all"


def test_group_summary_returns_empty_for_no_evaluations():
    assert report_module._group_summary([], "model") == {}


def test_format_text_report_handles_empty_and_non_empty_sections():
    empty_text = report_module._format_text_report(
        {"overall": {}, "by_category": {}, "by_model": {}, "by_source": {}},
        total_evaluations=0,
    )
    assert "No category data found." in empty_text
    assert "No model data found." in empty_text
    assert "No source data found." in empty_text

    text = report_module._format_text_report(
        {
            "overall": {
                "ground_truth_count": 1,
                "exact_match_count": 1,
                "error_count": 0,
                "exact_accuracy_percent": 100.0,
                "partial_accuracy_percent": 100.0,
            },
            "by_category": {"Excel": {"ground_truth_count": 1}},
            "by_model": {"model-a": {"ground_truth_count": 1}},
            "by_source": {"source-x": {"ground_truth_count": 1}},
        },
        total_evaluations=1,
    )
    assert "Excel" in text
    assert "model-a" in text
    assert "source-x" in text


def test_append_summary_block_handles_empty_and_data():
    lines = []
    report_module._append_summary_block(lines, None, title_field="model")
    assert lines == ["- No model data found."]

    lines = []
    report_module._append_summary_block(
        lines,
        {"m": {"ground_truth_count": 1}},
        title_field="model",
    )
    assert "m" in "\n".join(lines)


def test_parse_args_and_main_entrypoint(tmp_path, monkeypatch, capsys):
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    output_dir = tmp_path / "out"
    report_dir.mkdir()
    runtime_dir.mkdir()
    _write_report_csv(
        report_dir / "model-a__source-x__case01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"id":"A"}',
                "error_message": "",
            }
        ],
    )
    _write_runtime_dataset(
        runtime_dir / "model-a__source-x__case01.json",
        {"document_info": {"source_type": "Excel"}},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "category_accuracy_report",
            "--report-dir",
            str(report_dir),
            "--runtime-dir",
            str(runtime_dir),
            "--output-dir",
            str(output_dir),
            "--source",
            "source-x",
            "--model",
            "model-a",
        ],
    )

    exit_code = report_module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Saved JSON:" in captured.out
    assert "Total evaluations: 1" in captured.out


def test_generate_category_accuracy_reports_rejects_paths_outside_project(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(report_module, "PROJECT_ROOT", tmp_path / "repo")
    outside = Path("D:/outside")

    with pytest.raises(ValueError, match="project directory"):
        report_module.generate_category_accuracy_reports(
            report_dir=outside,
            runtime_dir=outside,
            output_dir=outside,
        )
