import csv
from pathlib import Path
from uuid import uuid4

import src.main as main_module


def _write_benchmark_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "row_index",
        "status",
        "score",
        "llm_output",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sandbox_dir(test_name: str) -> Path:
    root = Path.cwd() / "temp_pytest" / "red_benchmark_change_request"
    path = root / f"{test_name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parses_valid_json_in_llm_output_from_benchmark_csv():
    tmp_path = _sandbox_dir("parse_valid_json")
    report_path = tmp_path / "benchmark_reports" / "model-a__synthetic_examples__ex01.csv"
    report_path.parent.mkdir(parents=True)
    _write_benchmark_csv(
        report_path,
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"invoice_id":"INV-001","total":125000}',
                "error_message": "",
            }
        ],
    )

    rows = main_module._read_benchmark_report_rows(report_path)

    assert rows[0]["parsed_llm_output"] == {
        "invoice_id": "INV-001",
        "total": 125000,
    }


def test_handles_invalid_json_in_llm_output_without_crashing():
    tmp_path = _sandbox_dir("parse_invalid_json")
    report_path = tmp_path / "benchmark_reports" / "model-a__synthetic_examples__ex02.csv"
    report_path.parent.mkdir(parents=True)
    _write_benchmark_csv(
        report_path,
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "0.0",
                "llm_output": "{this is not json}",
                "error_message": "",
            }
        ],
    )

    rows = main_module._read_benchmark_report_rows(report_path)

    assert rows[0]["parsed_llm_output"] == {}


def test_default_source_filter_includes_only_synthetic_examples():
    sources = main_module._default_benchmark_report_sources()

    assert sources == ("synthetic_examples",)


def test_default_source_filter_excludes_lanjutan_and_like_real():
    sources = main_module._default_benchmark_report_sources()

    assert "synthetic_examples_lanjutan" not in sources
    assert "like-real_examples" not in sources


def test_default_collection_ignores_non_completed_and_empty_llm_output():
    tmp_path = _sandbox_dir("default_collection")
    report_dir = tmp_path / "benchmark_reports"
    report_dir.mkdir()
    _write_benchmark_csv(
        report_dir / "model-a__synthetic_examples__ex01.csv",
        [
            {
                "row_index": "1",
                "status": "failed",
                "score": "0.0",
                "llm_output": '{"invoice_id":"INV-FAILED"}',
                "error_message": "provider error",
            },
            {
                "row_index": "2",
                "status": "completed",
                "score": "0.0",
                "llm_output": "",
                "error_message": "",
            },
            {
                "row_index": "3",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"invoice_id":"INV-OK"}',
                "error_message": "",
            },
        ],
    )

    rows = main_module._collect_benchmark_report_rows(report_dir)

    assert [row["row_index"] for row in rows] == [3]
    assert rows[0]["parsed_llm_output"] == {"invoice_id": "INV-OK"}


def test_existing_txt_report_is_preserved_when_adding_new_source_column():
    tmp_path = _sandbox_dir("preserve_txt")
    report_dir = tmp_path / "benchmark_reports"
    report_dir.mkdir()
    existing_txt = report_dir / "overall_benchmark_report.txt"
    existing_txt.write_text(
        "Model | synthetic_examples\n"
        "model-a | 0.50\n",
        encoding="utf-8",
    )
    _write_benchmark_csv(
        report_dir / "model-a__synthetic_examples_lanjutan__dx01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"invoice_id":"INV-NEW"}',
                "error_message": "",
            }
        ],
    )

    output_path = main_module._append_source_column_to_text_report(
        existing_txt,
        report_dir=report_dir,
        source_name="synthetic_examples_lanjutan",
    )

    updated_content = output_path.read_text(encoding="utf-8")
    assert "Model | synthetic_examples" in updated_content
    assert "model-a | 0.50" in updated_content


def test_new_source_column_is_added_additively_to_existing_text_report():
    tmp_path = _sandbox_dir("additive_column")
    report_dir = tmp_path / "benchmark_reports"
    report_dir.mkdir()
    existing_txt = report_dir / "overall_benchmark_report.txt"
    existing_txt.write_text(
        "Model | synthetic_examples\n"
        "model-a | 0.50\n",
        encoding="utf-8",
    )
    _write_benchmark_csv(
        report_dir / "model-a__synthetic_examples_lanjutan__dx01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"invoice_id":"INV-NEW"}',
                "error_message": "",
            }
        ],
    )

    output_path = main_module._append_source_column_to_text_report(
        existing_txt,
        report_dir=report_dir,
        source_name="synthetic_examples_lanjutan",
    )

    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert "synthetic_examples" in header
    assert "synthetic_examples_lanjutan" in header
