import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Regular expressions for scientific metadata extraction
_DOI_REGEX = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_YEAR_REGEX = re.compile(r"\b(19\d{2}|20\d{2})\b")
_ABSTRACT_REGEX = re.compile(
    r"(?:abstract|summary)[:\s\n]+(.*?)(?:\n\s*\n|introduction|\b1[\.\s]|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def compute_content_hash(text: str) -> str:
    """Deterministic 12-char SHA-256 hash for document deduplication and ID."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def extract_scientific_metadata(
    text: str,
    filepath: str,
    pdf_meta: Optional[Dict[str, Any]] = None,
    html_soup: Optional[BeautifulSoup] = None,
) -> Dict[str, Any]:
    """Extract academic paper metadata (title, authors, year, DOI, abstract)."""
    meta: Dict[str, Any] = {
        "title": None,
        "authors": None,
        "year": None,
        "doi": None,
        "abstract": None,
    }

    # 1. DOI extraction
    doi_match = _DOI_REGEX.search(text)
    if doi_match:
        meta["doi"] = doi_match.group(1).rstrip(".")

    # 2. Metadata from PDF or HTML headers if available
    if pdf_meta:
        if pdf_meta.get("title") and len(pdf_meta["title"].strip()) > 3:
            meta["title"] = pdf_meta["title"].strip()
        if pdf_meta.get("author") and len(pdf_meta["author"].strip()) > 1:
            meta["authors"] = pdf_meta["author"].strip()
        if pdf_meta.get("creationDate"):
            # PDF date format e.g. D:20230501...
            date_str = str(pdf_meta["creationDate"])
            year_match = _YEAR_REGEX.search(date_str)
            if year_match:
                meta["year"] = int(year_match.group(1))

    elif html_soup:
        title_tag = html_soup.find("title")
        if title_tag and title_tag.string:
            meta["title"] = title_tag.string.strip()
        author_meta = html_soup.find("meta", attrs={"name": re.compile(r"author", re.I)})
        if author_meta and author_meta.get("content"):
            meta["authors"] = author_meta["content"].strip()

    # 3. Fallback title from text header or filename
    if not meta["title"]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            first_line = re.sub(r"^#+\s*", "", lines[0])
            if 5 < len(first_line) < 200:
                meta["title"] = first_line
            else:
                meta["title"] = os.path.splitext(os.path.basename(filepath))[0]
        else:
            meta["title"] = os.path.splitext(os.path.basename(filepath))[0]

    # 4. Fallback year from first 2000 chars of text
    if not meta["year"]:
        year_match = _YEAR_REGEX.search(text[:2000])
        if year_match:
            meta["year"] = int(year_match.group(1))

    # 5. Abstract extraction (from leading text)
    abstract_match = _ABSTRACT_REGEX.search(text[:4000])
    if abstract_match:
        clean_abs = re.sub(r"\s+", " ", abstract_match.group(1)).strip()
        if len(clean_abs) > 30:
            meta["abstract"] = clean_abs[:800]

    return meta


class multiloader:
    def __init__(self, path: str):
        self.path = path
        self.errors: List[Dict[str, str]] = []

    def _document_loader(self, deduplicate: bool = True) -> List[Document]:
        documents: List[Document] = []
        seen_hashes = set()

        for root, dirs, files in os.walk(self.path):
            dirs.sort()
            for file in sorted(files):
                filepath = os.path.join(root, file)
                doc = self._process_file(filepath)
                if doc:
                    if deduplicate:
                        chash = doc.metadata.get("content_hash")
                        if chash and chash in seen_hashes:
                            logger.info("Skipping duplicate document content: %s", filepath)
                            continue
                        if chash:
                            seen_hashes.add(chash)
                    documents.append(doc)
        return documents

    def load_file(self, filepath: str) -> Optional[Document]:
        """Public helper to load a single file directly."""
        return self._process_file(filepath)

    def _process_file(self, filepath: str) -> Optional[Document]:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in [".pdf", ".html", ".htm", ".md", ".txt"]:
            return None

        metadata: Dict[str, Any] = {
            "filepath": filepath,
            "extension": ext,
            "filename": os.path.basename(filepath),
            "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
        }

        try:
            pdf_meta = None
            html_soup = None
            page_count = 1

            if ext == ".pdf":
                content, pdf_meta, page_count = self._load_pdf_with_meta(filepath)
            elif ext in [".html", ".htm"]:
                content, html_soup = self._load_html_with_soup(filepath)
            elif ext in [".md", ".txt"]:
                content = self._load_text(filepath)
            else:
                return None

            if not content or not content.strip():
                return None

            # Enrich with scientific metadata
            sci_meta = extract_scientific_metadata(
                content, filepath, pdf_meta=pdf_meta, html_soup=html_soup
            )
            metadata.update(sci_meta)
            metadata["page_count"] = page_count
            metadata["content_hash"] = compute_content_hash(content)
            metadata["doc_id"] = metadata["content_hash"]

            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            err_msg = f"Failed to load {filepath}: {e}"
            logger.warning(err_msg)
            self.errors.append({"filepath": filepath, "error": str(e)})
            return None

    def _load_pdf_with_meta(self, filepath: str) -> Tuple[str, Dict[str, Any], int]:
        text = ""
        pdf_meta: Dict[str, Any] = {}
        page_count = 0
        with pymupdf.open(filepath) as doc:
            page_count = len(doc)
            try:
                pdf_meta = dict(doc.metadata or {})
            except Exception:
                pdf_meta = {}
            for page in doc:
                text += str(page.get_text("text")) + "\n"
        return self._normalize_text(text), pdf_meta, page_count

    def _load_pdf(self, filepath: str) -> str:
        content, _, _ = self._load_pdf_with_meta(filepath)
        return content

    def _load_html_with_soup(self, filepath: str) -> Tuple[str, BeautifulSoup]:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator=" ")
        return self._normalize_text(text), soup

    def _load_html(self, filepath: str) -> str:
        content, _ = self._load_html_with_soup(filepath)
        return content

    def _load_text(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self._normalize_text(text)

    def _normalize_text(self, text: str) -> str:
        """Collapse runs of spaces/tabs and excess blank lines, but keep line
        breaks: the Markdown chunker splits on lines starting with "#", and
        the recursive splitter prefers paragraph boundaries, so flattening
        everything onto one line would defeat both."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
