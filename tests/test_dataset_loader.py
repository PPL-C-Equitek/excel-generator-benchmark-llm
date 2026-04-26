import json

import pytest
from docx import Document

from src.dataset_loader import (
    DatasetSchemaError,
    UnsupportedDatasetFormatError,
    load_dataset,
)


DATASET_SCHEMA = {
    "id": int,
    "prompt": str,
    "expected": str,
}


def _write_docx_table(path, headers, rows):
    document = Document()
    table = document.add_table(rows=1, cols=len(headers))
    for index, header in enumerate(headers):
        table.rows[0].cells[index].text = header

    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = value

    document.save(path)


@pytest.mark.parametrize("extension", [".csv", ".json", ".docx"])
def test_load_dataset_supports_csv_json_and_docx_with_schema_validation(
    tmp_path,
    extension,
):
    dataset_path = tmp_path / f"benchmark{extension}"

    if extension == ".csv":
        dataset_path.write_text(
            "id,prompt,expected\n"
            "1,Summarize invoice INV-001,Invoice INV-001 total is 1500000\n"
            "2,Extract customer name,Acme Corp\n",
            encoding="utf-8",
        )
    elif extension == ".json":
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "id": 1,
                        "prompt": "Summarize invoice INV-001",
                        "expected": "Invoice INV-001 total is 1500000",
                    },
                    {
                        "id": 2,
                        "prompt": "Extract customer name",
                        "expected": "Acme Corp",
                    },
                ]
            ),
            encoding="utf-8",
        )
    else:
        _write_docx_table(
            dataset_path,
            headers=["id", "prompt", "expected"],
            rows=[
                [
                    "1",
                    "Summarize invoice INV-001",
                    "Invoice INV-001 total is 1500000",
                ],
                ["2", "Extract customer name", "Acme Corp"],
            ],
        )

    rows = load_dataset(dataset_path, schema=DATASET_SCHEMA)

    assert rows == [
        {
            "id": 1,
            "prompt": "Summarize invoice INV-001",
            "expected": "Invoice INV-001 total is 1500000",
        },
        {
            "id": 2,
            "prompt": "Extract customer name",
            "expected": "Acme Corp",
        },
    ]
    assert all(isinstance(row["id"], int) for row in rows)


def test_load_dataset_rejects_unrecognized_file_format(tmp_path):
    dataset_path = tmp_path / "benchmark.txt"
    dataset_path.write_text("id,prompt,expected\n1,Prompt,Expected\n", encoding="utf-8")

    with pytest.raises(
        UnsupportedDatasetFormatError,
        match=r"Unsupported dataset format.*\.txt",
    ):
        load_dataset(dataset_path, schema=DATASET_SCHEMA)


def test_load_dataset_rejects_missing_required_columns(tmp_path):
    dataset_path = tmp_path / "benchmark.csv"
    dataset_path.write_text(
        "id,prompt\n"
        "1,Summarize invoice INV-001\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="Missing required columns.*expected"):
        load_dataset(dataset_path, schema=DATASET_SCHEMA)


def test_load_dataset_rejects_values_that_do_not_match_schema_type(tmp_path):
    dataset_path = tmp_path / "benchmark.csv"
    dataset_path.write_text(
        "id,prompt,expected\n"
        "not-an-integer,Summarize invoice INV-001,Invoice INV-001 total is 1500000\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="id.*int"):
        load_dataset(dataset_path, schema=DATASET_SCHEMA)


def test_load_dataset_supports_wrapped_json_rows_and_normalizes_keys(tmp_path):
    dataset_path = tmp_path / "benchmark.json"
    dataset_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id ": 1,
                        " prompt ": "Summarize invoice INV-001",
                        "expected ": "Invoice INV-001 total is 1500000",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = load_dataset(dataset_path, schema=DATASET_SCHEMA)

    assert rows == [
        {
            "id": 1,
            "prompt": "Summarize invoice INV-001",
            "expected": "Invoice INV-001 total is 1500000",
        }
    ]


def test_load_dataset_returns_empty_rows_for_empty_json_wrapper(tmp_path):
    dataset_path = tmp_path / "benchmark.json"
    dataset_path.write_text(json.dumps({"rows": []}), encoding="utf-8")

    rows = load_dataset(dataset_path, schema=DATASET_SCHEMA)

    assert rows == []


def test_load_dataset_skips_thousands_of_empty_rows_without_crashing(tmp_path):
    dataset_path = tmp_path / "benchmark.csv"
    empty_rows = "\n" * 5_000
    dataset_path.write_text(
        "id,prompt,expected\n"
        "1,First prompt,First expected\n"
        f"{empty_rows}"
        "2,Second prompt,Second expected\n",
        encoding="utf-8",
    )

    rows = load_dataset(dataset_path, schema=DATASET_SCHEMA)

    assert rows == [
        {"id": 1, "prompt": "First prompt", "expected": "First expected"},
        {"id": 2, "prompt": "Second prompt", "expected": "Second expected"},
    ]
