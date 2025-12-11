# detector/utils.py
import re
import unicodedata
from typing import List

# remove common zero-width & invisible characters
ZERO_WIDTH_RE = re.compile(
    "[" +
    "\u200B"  # zero width space
    "\u200C"  # zero width non-joiner
    "\u200D"  # zero width joiner
    "\uFEFF"  # zero width no-break
    "]"
)

def remove_zero_width(text: str) -> str:
    return ZERO_WIDTH_RE.sub("", text)

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if (ord(ch) >= 32 or ch in "\n\t"))
    return text

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = remove_zero_width(text)
    text = normalize_unicode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, max_words:int=300, overlap:int=50) -> List[str]:
    """
    Chunk by words. max_words: words per chunk. overlap: words overlapped between chunks.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks
