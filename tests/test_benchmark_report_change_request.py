import csv
from pathlib import Path
from uuid import uuid4

import pytest

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


def test_project_path_returns_absolute_path_unchanged():
    absolute_path = Path.cwd().resolve()

    assert main_module._project_path(absolute_path) == absolute_path


def test_collect_benchmark_rows_skips_non_matching_and_malformed_filenames():
    tmp_path = _sandbox_dir("collect_filters")
    report_dir = tmp_path / "benchmark_reports"
    report_dir.mkdir()
    _write_benchmark_csv(
        report_dir / "malformed.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"ok": true}',
                "error_message": "",
            }
        ],
    )
    _write_benchmark_csv(
        report_dir / "model-a__like-real_examples__lr01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "1.0",
                "llm_output": '{"ok": true}',
                "error_message": "",
            }
        ],
    )

    rows = main_module._collect_benchmark_report_rows(report_dir)

    assert rows == []


def test_average_scores_by_model_ignores_rows_without_model(mocker):
    mocker.patch(
        "src.main._collect_benchmark_report_rows",
        return_value=[
            {"model": "", "score": "0.5", "status": "completed", "llm_output": "{}"},
            {
                "model": "model-a",
                "score": "1.0",
                "status": "completed",
                "llm_output": "{}",
            },
        ],
    )

    averages = main_module._average_scores_by_model_for_source(
        Path.cwd(),
        "synthetic_examples",
    )

    assert averages == {"model-a": pytest.approx(1.0)}


def test_append_source_column_raises_when_base_text_missing():
    with pytest.raises(FileNotFoundError):
        main_module._append_source_column_to_text_report(
            Path.cwd() / "missing_overall.txt",
            report_dir=Path.cwd(),
            source_name="synthetic_examples_lanjutan",
        )


def test_append_source_column_handles_empty_existing_text_file():
    tmp_path = _sandbox_dir("empty_text")
    report_dir = tmp_path / "benchmark_reports"
    report_dir.mkdir()
    existing_txt = report_dir / "overall_benchmark_report.txt"
    existing_txt.write_text("", encoding="utf-8")

    output_path = main_module._append_source_column_to_text_report(
        existing_txt,
        report_dir=report_dir,
        source_name="synthetic_examples_lanjutan",
    )

    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("Model | synthetic_examples_lanjutan")


def test_append_source_column_updates_full_report_sections():
    tmp_path = _sandbox_dir("full_sections")
    report_dir = tmp_path / "benchmark_reports"
    report_dir.mkdir()
    existing_txt = report_dir / "overall_benchmark_report.txt"
    existing_txt.write_text(
        "\n".join(
            [
                "LLM Benchmark Report",
                "====================",
                "",
                "Model Summary",
                "-------------",
                main_module._format_table(
                    ["Model", "Average Score", "Average Seconds", "Completed/Total"],
                    [["model-a", "0.50", "10.00", "1/1"]],
                ),
                "",
                "Overall Winners",
                "---------------",
                main_module._format_table(
                    ["Category", "Model", "Average Score", "Average Seconds"],
                    [
                        ["Best average score", "model-a", "0.50", "10.00"],
                        ["Fastest average runtime", "model-a", "0.50", "10.00"],
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_benchmark_csv(
        report_dir / "model-a__synthetic_examples_lanjutan__dx01.csv",
        [
            {
                "row_index": "1",
                "status": "completed",
                "score": "0.75",
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
    content = output_path.read_text(encoding="utf-8")

    assert "| synthetic_examples_lanjutan |" in content
    assert "| 0.7500" in content
    # Kept from overall winners baseline: column intentionally blank here.
    assert "Fastest average runtime" in content


@pytest.mark.parametrize(
    "source_name",
    [
        "../escape",
        "/tmp/escape",
        "..\\escape",
        "folder\\sub",
        "folder/sub",
    ],
)
def test_append_source_column_rejects_malicious_source_name(
    source_name,
):
    tmp_path = _sandbox_dir("malicious_source_name")
    report_dir = tmp_path / "benchmark_reports"
    report_dir.mkdir()
    existing_txt = report_dir / "overall_benchmark_report.txt"
    existing_txt.write_text("Model | synthetic_examples\nmodel-a | 0.5\n", encoding="utf-8")

    output_path = main_module._append_source_column_to_text_report(
        existing_txt,
        report_dir=report_dir,
        source_name=source_name,
    )
    # Path is derived from existing report only, never from source_name.
    assert output_path.name == "overall_benchmark_report_source_augmented.txt"


def test_derived_text_report_path_stays_inside_report_dir():
    tmp_path = _sandbox_dir("derived_path")
    report_dir = tmp_path / "benchmark_reports"
    report_dir.mkdir()

    derived_path = main_module._derived_text_report_path(report_dir=report_dir)

    assert derived_path.parent.resolve() == report_dir.resolve()
    assert derived_path.name == "overall_benchmark_report_source_augmented.txt"


def test_derived_text_report_path_rejects_base_dir_outside_project(monkeypatch):
    fake_project_root = Path.cwd().resolve() / "repo-root"
    monkeypatch.setattr(main_module, "PROJECT_ROOT", fake_project_root)

    outside_dir = Path("D:/outside-report-dir")
    with pytest.raises(ValueError, match="must stay inside the project directory"):
        main_module._derived_text_report_path(report_dir=outside_dir)


def test_model_summary_source_score_handles_missing_inputs():
    assert (
        main_module._model_summary_source_score(
            row_cells=[],
            source_model_scores={"model-a": 0.5},
        )
        == ""
    )
    assert (
        main_module._model_summary_source_score(
            row_cells=["model-a"],
            source_model_scores={},
        )
        == ""
    )


def test_overall_winner_source_score_handles_missing_inputs():
    assert (
        main_module._overall_winner_source_score(
            row_cells=[],
            source_model_scores={"model-a": 0.5},
        )
        == ""
    )
    assert (
        main_module._overall_winner_source_score(
            row_cells=["Best average score"],
            source_model_scores={"model-a": 0.5},
        )
        == ""
    )
    assert (
        main_module._overall_winner_source_score(
            row_cells=["Best average score", "unknown-model"],
            source_model_scores={},
        )
        == ""
    )


def test_add_column_to_section_table_handles_missing_section():
    lines = ["Header", "No table here"]

    assert (
        main_module._add_column_to_section_table(
            lines,
            section_title="Model Summary",
            column_name="synthetic_examples_lanjutan",
            value_by_row=lambda _cells: "",
        )
        == lines
    )


def test_add_column_to_section_table_handles_section_without_border():
    lines = ["Model Summary", "-------------", "| header |"]

    assert (
        main_module._add_column_to_section_table(
            lines,
            section_title="Model Summary",
            column_name="synthetic_examples_lanjutan",
            value_by_row=lambda _cells: "",
        )
        == lines
    )


def test_add_column_to_section_table_handles_non_tabular_header_line():
    lines = ["Model Summary", "-------------", "+---+", "not a row", "+---+"]

    assert (
        main_module._add_column_to_section_table(
            lines,
            section_title="Model Summary",
            column_name="synthetic_examples_lanjutan",
            value_by_row=lambda _cells: "",
        )
        == lines
    )


def test_add_column_to_section_table_handles_missing_closing_border():
    lines = [
        "Model Summary",
        "-------------",
        "+---+",
        "| Model |",
        "+---+",
        "| model-a |",
    ]

    assert (
        main_module._add_column_to_section_table(
            lines,
            section_title="Model Summary",
            column_name="synthetic_examples_lanjutan",
            value_by_row=lambda _cells: "",
        )
        == lines
    )


def test_add_column_to_section_table_does_not_duplicate_existing_column():
    rendered = main_module._format_table(
        ["Model", "synthetic_examples_lanjutan"],
        [["model-a", "0.75"]],
    )
    lines = ["Model Summary", "-------------", *rendered.splitlines()]

    updated = main_module._add_column_to_section_table(
        lines,
        section_title="Model Summary",
        column_name="synthetic_examples_lanjutan",
        value_by_row=lambda _cells: "0.80",
    )

    joined = "\n".join(updated)
    assert joined.count("synthetic_examples_lanjutan") == 1
    assert "0.80" in joined


def test_add_column_to_section_table_skips_non_row_lines_inside_table_body():
    lines = [
        "Model Summary",
        "-------------",
        "+-------+-------+",
        "| Model | Score |",
        "+-------+-------+",
        "not-a-row",
        "| model-a | 0.50 |",
        "+-------+-------+",
    ]

    updated = main_module._add_column_to_section_table(
        lines,
        section_title="Model Summary",
        column_name="synthetic_examples_lanjutan",
        value_by_row=lambda _cells: "0.75",
    )

    joined = "\n".join(updated)
    assert "model-a" in joined
    assert "0.75" in joined


def test_helpers_for_table_lookup_and_cells():
    lines = ["A", "Model Summary", "+---+", "| X |", "+---+"]

    assert main_module._find_line_index(lines, "Model Summary") == 1
    assert main_module._find_line_index(lines, "missing") is None
    assert main_module._find_table_border_line(lines, 0) == 2
    assert main_module._find_table_border_line(lines, 4) == 4
    assert main_module._find_table_border_line(lines, 5) is None
    assert main_module._table_cells("| A | B |") == ["A", "B"]
    assert main_module._table_cells("plain text") == []


def test_filename_source_extractor_handles_invalid_and_valid_names():
    assert (
        main_module._model_and_source_from_report_filename(Path("bad.csv"))
        is None
    )
    assert main_module._model_and_source_from_report_filename(
        Path("model-a__synthetic_examples__ex01.csv")
    ) == ("model-a", "synthetic_examples")
