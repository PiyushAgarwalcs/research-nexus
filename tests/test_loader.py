from src.loader import MultiLoader


def test_normalize_text_collapses_whitespace_but_keeps_line_breaks():
    loader = MultiLoader(".")

    assert loader._normalize_text("  Hello \n\n  world \t!  ") == "Hello\n\nworld !"
    assert loader._normalize_text("a\r\n\r\n\r\n\nb") == "a\n\nb"


def test_loaded_markdown_still_splits_on_headers(tmp_path):
    """Regression test: flattening all whitespace onto one line left the
    Markdown chunker with a single header line, i.e. one chunk per file."""
    from src.chunker import Chunker

    (tmp_path / "doc.md").write_text(
        "# Title\n\nIntro text.\n\n## Section A\n\nAlpha.\n\n## Section B\n\nBeta.\n",
        encoding="utf-8",
    )
    doc = MultiLoader(str(tmp_path)).load_documents()[0]

    chunks = Chunker(embedding_fn=None).chunk_documents(
        doc.page_content, doc.metadata, strategy="Markdown"
    )

    assert [c.metadata.get("Header 2") for c in chunks] == [None, "Section A", "Section B"]
    assert all(c.metadata["Header 1"] == "Title" for c in chunks)


def test_load_documents_returns_files_in_sorted_order(tmp_path):
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")

    documents = MultiLoader(str(tmp_path)).load_documents()

    assert [d.metadata["filename"] for d in documents] == ["a.txt", "b.txt", "c.txt"]


def test_load_documents_reads_supported_files_and_skips_others(tmp_path):
    (tmp_path / "note.txt").write_text("Hello   world", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")

    loader = MultiLoader(str(tmp_path))
    documents = loader.load_documents()

    assert len(documents) == 1
    assert documents[0].page_content == "Hello world"
    assert documents[0].metadata["filename"] == "note.txt"


def test_process_file_returns_none_for_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    loader = MultiLoader(str(tmp_path))
    assert loader._process_file(str(empty_file)) is None


def test_scientific_metadata_extraction(tmp_path):
    paper_content = """# Deep Learning in Genomics: A Comprehensive Survey

By John Doe and Jane Smith
Published in 2023. DOI: 10.1038/s41586-023-01234-5

Abstract:
This survey provides a comprehensive review of transformer architectures applied to genomic sequence analysis. We evaluate multiple benchmarks and analyze cross-domain generalization.

1. Introduction
Genomics has witnessed exponential data growth...
"""
    doc_path = tmp_path / "paper.md"
    doc_path.write_text(paper_content, encoding="utf-8")

    loader = MultiLoader(str(tmp_path))
    doc = loader.load_file(str(doc_path))

    assert doc is not None
    assert doc.metadata["doi"] == "10.1038/s41586-023-01234-5"
    assert doc.metadata["year"] == 2023
    assert "Deep Learning in Genomics" in doc.metadata["title"]
    assert doc.metadata["abstract"] is not None
    assert "comprehensive review of transformer architectures" in doc.metadata["abstract"]
    assert "content_hash" in doc.metadata
    assert doc.metadata["doc_id"] == doc.metadata["content_hash"]


def test_deduplication_filters_duplicate_content(tmp_path):
    content = "Identical scientific paper text for duplication testing."
    (tmp_path / "copy1.txt").write_text(content, encoding="utf-8")
    (tmp_path / "copy2.txt").write_text(content, encoding="utf-8")

    loader = MultiLoader(str(tmp_path))
    docs = loader.load_documents(deduplicate=True)

    # Should only keep 1 copy because the content hash matches
    assert len(docs) == 1
