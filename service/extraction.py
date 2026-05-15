import fitz

MAX_EXTRACT_CHARS = 32_000  # slightly above the LLM truncation limit to allow reference stripping


def extract_pdf_text(file_bytes: bytes) -> tuple[str, int]:
    """Extracts plain text from PDF bytes directly (no temp file, no layout analysis).

    Stops reading pages once MAX_EXTRACT_CHARS is reached — no need to process
    the entire document when only the first ~24k chars will be sent to the model.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = len(doc)
    parts: list[str] = []
    total = 0

    for page in doc:
        text = page.get_text()
        parts.append(text)
        total += len(text)
        if total >= MAX_EXTRACT_CHARS:
            break

    doc.close()
    return "\n".join(parts), pages
