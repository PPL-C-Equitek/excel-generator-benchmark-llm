from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from docx import Document


class DatasetSchemaError(ValueError):
    """Raised when dataset columns or values do not match the schema."""


class UnsupportedDatasetFormatError(ValueError):
    """Raised when a dataset file extension is not supported."""


SUPPORTED_EXTENSIONS = {".csv", ".json", ".docx"}


def load_dataset(path: str | Path, schema: Mapping[str, type]) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    _validate_schema(schema)

    extension = dataset_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDatasetFormatError(
            f"Unsupported dataset format '{extension}' for {dataset_path}. "
            f"Supported formats: {supported}."
        )

    if extension == ".csv":
        rows = _iter_csv_rows(dataset_path, schema)
    elif extension == ".json":
        rows = _iter_json_rows(dataset_path, schema)
    else:
        rows = _iter_docx_rows(dataset_path, schema)

    return list(_validate_and_cast_rows(rows, schema))


def _validate_schema(schema: Mapping[str, type]) -> None:
    if not schema:
        raise DatasetSchemaError("Schema must define at least one required column.")

    invalid_columns = [
        column
        for column, expected_type in schema.items()
        if not isinstance(expected_type, type)
    ]
    if invalid_columns:
        joined_columns = ", ".join(invalid_columns)
        raise DatasetSchemaError(
            f"Schema columns must map to Python types: {joined_columns}."
        )


def _iter_csv_rows(
    path: Path,
    schema: Mapping[str, type],
) -> Iterator[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        headers = _next_non_empty_csv_row(reader)
        _validate_required_columns(headers, schema)

        for values in reader:
            if _is_empty_sequence(values):
                continue
            yield _row_from_values(headers, values)


def _iter_json_rows(
    path: Path,
    schema: Mapping[str, type],
) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        records = payload.get("rows") or payload.get("data")
        if records is None:
            records = [payload]
    else:
        records = payload

    if not isinstance(records, list):
        raise DatasetSchemaError(
            "JSON dataset must contain an object or list of objects."
        )

    first_non_empty_index: int | None = None

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetSchemaError("JSON dataset rows must be objects.")
        if _is_empty_mapping(record):
            continue
        first_non_empty_index = index
        break

    headers = (
        list(records[first_non_empty_index].keys())
        if first_non_empty_index is not None
        else []
    )
    _validate_required_columns(headers, schema)

    start_index = first_non_empty_index or 0
    for record in records[start_index:]:
        if not isinstance(record, dict):
            raise DatasetSchemaError("JSON dataset rows must be objects.")
        if _is_empty_mapping(record):
            continue
        yield record


def _iter_docx_rows(
    path: Path,
    schema: Mapping[str, type],
) -> Iterator[dict[str, Any]]:
    document = Document(path)
    if not document.tables:
        raise DatasetSchemaError("DOCX dataset must contain at least one table.")

    table = document.tables[0]
    if not table.rows:
        raise DatasetSchemaError("DOCX dataset table must contain a header row.")

    headers = [_cell_text(cell) for cell in table.rows[0].cells]
    _validate_required_columns(headers, schema)

    for table_row in table.rows[1:]:
        values = [_cell_text(cell) for cell in table_row.cells]
        if _is_empty_sequence(values):
            continue
        yield _row_from_values(headers, values)


def _validate_and_cast_rows(
    rows: Iterable[dict[str, Any]],
    schema: Mapping[str, type],
) -> Iterator[dict[str, Any]]:
    for row_number, row in enumerate(rows, start=1):
        if _is_empty_mapping(row):
            continue

        cast_row: dict[str, Any] = {}
        for column, expected_type in schema.items():
            value = row.get(column)
            if _is_empty_value(value):
                raise DatasetSchemaError(
                    f"Column '{column}' is required and cannot be empty "
                    f"at row {row_number}."
                )
            cast_row[column] = _cast_value(column, value, expected_type, row_number)

        yield cast_row


def _cast_value(
    column: str,
    value: Any,
    expected_type: type,
    row_number: int,
) -> Any:
    if isinstance(value, expected_type):
        return value

    if expected_type is str:
        return str(value)

    try:
        return expected_type(value)
    except (TypeError, ValueError) as exc:
        raise DatasetSchemaError(
            f"Column '{column}' must be {expected_type.__name__} at row {row_number}."
        ) from exc


def _next_non_empty_csv_row(reader: Iterator[list[str]]) -> list[str]:
    for row in reader:
        if _is_empty_sequence(row):
            continue
        return [_clean_header(value) for value in row]
    return []


def _validate_required_columns(
    headers: Iterable[str],
    schema: Mapping[str, type],
) -> None:
    normalized_headers = {_clean_header(header) for header in headers}
    missing_columns = [
        column for column in schema if _clean_header(column) not in normalized_headers
    ]
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise DatasetSchemaError(f"Missing required columns: {joined_columns}.")


def _row_from_values(headers: list[str], values: list[Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for index, header in enumerate(headers):
        if not header:
            continue
        row[header] = values[index] if index < len(values) else ""
    return row


def _cell_text(cell: Any) -> str:
    return cell.text.strip()


def _clean_header(value: Any) -> str:
    return str(value).strip()


def _is_empty_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _is_empty_sequence(values: Iterable[Any]) -> bool:
    return all(_is_empty_value(value) for value in values)


def _is_empty_mapping(row: Mapping[str, Any]) -> bool:
    return all(_is_empty_value(value) for value in row.values())
