import re
from typing import List

def split_by_headings(md: str) -> List[str]:
    """
    Splits markdown into sections by headings (#, ##, ###).
    Keeps headings with the section body.
    """
    md = md.strip()
    if not md:
        return []

    # Ensure headings start at line beginnings
    parts = re.split(r"(?m)^(?=#)", md)
    sections = [p.strip() for p in parts if p.strip()]
    return sections

def chunk_text(text: str, max_chars: int = 1400, overlap: int = 180) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks

def smart_chunk_markdown(md: str, max_chars: int = 1400, overlap: int = 180) -> List[str]:
    """
    1) Split by headings to keep semantics
    2) For each section, apply window chunking if needed
    """
    sections = split_by_headings(md)
    out: List[str] = []
    for sec in sections:
        if len(sec) <= max_chars:
            out.append(sec)
        else:
            out.extend(chunk_text(sec, max_chars=max_chars, overlap=overlap))
    return out
