from backend.document_parser import DocumentParser


def test_build_pdf_text_and_page_info_tracks_tables_in_spans() -> None:
    pages = [
        {"page": 1, "text": "第一页文本", "segments": ["\n[表格 1]\n| a |\n| --- |\n| 1 |\n"]},
        {"page": 2, "text": "", "segments": ["\n[表格 2]\n| b |\n| --- |\n| 2 |\n"]},
    ]

    text, page_info = DocumentParser._build_pdf_text_and_page_info(pages)

    assert len(page_info) == 2
    assert text[page_info[0]["char_start"] : page_info[0]["char_end"]].startswith("第一页文本")
    assert "[表格 1]" in text[page_info[0]["char_start"] : page_info[0]["char_end"]]
    assert "[表格 2]" in text[page_info[1]["char_start"] : page_info[1]["char_end"]]

    assert page_info[0]["char_end"] + 2 == page_info[1]["char_start"]


def test_build_pdf_text_and_page_info_includes_multiple_segments_same_page() -> None:
    pages = [
        {
            "page": 1,
            "text": "P1",
            "segments": ["\n[表格 1]\n| x |\n| --- |\n| y |\n", "\n[表格 2]\n| m |\n| --- |\n| n |\n"],
        }
    ]

    text, page_info = DocumentParser._build_pdf_text_and_page_info(pages)

    assert len(page_info) == 1
    page_text = text[page_info[0]["char_start"] : page_info[0]["char_end"]]
    assert "P1" in page_text
    assert "[表格 1]" in page_text
    assert "[表格 2]" in page_text
