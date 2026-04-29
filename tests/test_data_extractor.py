from pathlib import Path

import pytest
from docx import Document

from src.data_extractor import extract_text_from_file


def test_extract_text_from_file_reads_csv_and_txt(tmp_path):
    csv_path = tmp_path / "sample.csv"
    txt_path = tmp_path / "sample.txt"
    csv_path.write_text("unit,item,value\nIT,Laptop,15000000\n", encoding="utf-8")
    txt_path.write_text("plain extracted text", encoding="utf-8")

    assert extract_text_from_file(csv_path) == (
        "unit,item,value\nIT,Laptop,15000000\n"
    )
    assert extract_text_from_file(txt_path) == "plain extracted text"


def test_extract_text_from_file_extracts_docx_paragraphs_and_tables(tmp_path):
    docx_path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("Quarterly budget")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Item"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Laptop"
    table.cell(1, 1).text = "15000000"
    document.save(docx_path)

    extracted_text = extract_text_from_file(docx_path)

    assert extracted_text.splitlines() == [
        "Quarterly budget",
        "Item\tValue",
        "Laptop\t15000000",
    ]


def test_extract_text_from_file_extracts_pdf_with_pdfplumber(tmp_path, mocker):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF mocked bytes")
    pdf_reader = mocker.MagicMock()
    pdf_reader.__enter__.return_value.pages = [
        mocker.Mock(extract_text=mocker.Mock(return_value="First page")),
        mocker.Mock(extract_text=mocker.Mock(return_value=None)),
        mocker.Mock(extract_text=mocker.Mock(return_value="Second page")),
    ]
    pdf_open = mocker.patch(
        "src.data_extractor.pdfplumber.open",
        return_value=pdf_reader,
    )

    extracted_text = extract_text_from_file(pdf_path)

    pdf_open.assert_called_once_with(pdf_path)
    assert extracted_text == "First page\nSecond page"


def test_extract_text_from_file_extracts_png_with_pytesseract(tmp_path, mocker):
    png_path = tmp_path / "invoice.png"
    png_path.write_bytes(b"mock image bytes")
    image = mocker.MagicMock()
    image_open = mocker.patch(
        "src.data_extractor.Image.open",
        return_value=image,
    )
    image_to_string = mocker.patch(
        "src.data_extractor.pytesseract.image_to_string",
        return_value="Invoice OCR text",
    )

    extracted_text = extract_text_from_file(png_path)

    image_open.assert_called_once_with(png_path)
    image_to_string.assert_called_once_with(image)
    assert extracted_text == "Invoice OCR text"


def test_extract_text_from_file_raises_clear_error_when_ocr_is_unavailable(
    tmp_path,
    mocker,
):
    png_path = tmp_path / "invoice.png"
    png_path.write_bytes(b"mock image bytes")
    mocker.patch("src.data_extractor.Image.open", return_value=mocker.Mock())
    mocker.patch(
        "src.data_extractor.pytesseract.image_to_string",
        side_effect=RuntimeError("tesseract is not installed"),
    )

    with pytest.raises(ValueError, match="Tesseract OCR is not available"):
        extract_text_from_file(png_path)


def test_extract_text_from_file_rejects_unsupported_extensions(tmp_path):
    unsupported_path = tmp_path / "sample.xlsx"
    unsupported_path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_text_from_file(unsupported_path)
