from __future__ import annotations

from io import BytesIO


def extract_document_text(file_name: str, content: bytes) -> str:
    suffix = file_name.lower().rsplit(".", 1)[-1]
    if suffix == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix == "docx":
        from docx import Document

        document = Document(BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    if suffix == "txt":
        return content.decode("utf-8", errors="replace").strip()
    raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT file.")