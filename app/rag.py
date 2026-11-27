import re
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

_model = None
_chunks: list[dict] = []
_vectors: np.ndarray | None = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model

def split_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    size = chunk_size or settings.chunk_size
    step_overlap = settings.chunk_overlap if overlap is None else overlap
    if size <= 0 or step_overlap < 0 or step_overlap >= size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(normalized):
            break
        start = max(end - step_overlap, start + 1)
    return chunks

def add_document(filename: str, text: str, page_number: int | None = None) -> int:
    global _chunks, _vectors
    pieces = split_text(text)
    if not pieces:
        return 0
    model = get_model()
    vectors = model.encode(pieces, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    start_id = len(_chunks)
    for offset, piece in enumerate(pieces):
        _chunks.append({
            "filename": filename,
            "chunk_id": start_id + offset,
            "page": page_number,
            "text": piece,
        })
    _vectors = vectors if _vectors is None else np.vstack([_vectors, vectors])
    return len(pieces)

def search(question: str, top_k: int = 4) -> list[dict]:
    if not _chunks or _vectors is None:
        return []
    query_vector = np.asarray(get_model().encode([question], normalize_embeddings=True)[0], dtype=np.float32)
    scores = _vectors @ query_vector
    indices = np.argsort(scores)[::-1][:top_k]
    return [{**_chunks[int(i)], "score": float(scores[int(i)])} for i in indices]


def generate_answer(question: str, results: list[dict]) -> tuple[str, str]:
    if not results:
        return (
            "No relevant information was found. Upload a document first.",
            "no_context"
        )

    # Build an extractive answer from retrieved document passages
    answer_parts = []

    for result in results[:3]:
        text = result["text"].strip()
        if text:
            answer_parts.append(
                f"Source: {result['filename']}\n{text}"
            )

    if not answer_parts:
        return "No usable passages were retrieved.", "no_context"

    answer = (
        "Answer based on the uploaded documents:\n\n"
        + "\n\n".join(answer_parts)
    )

    return answer, "extractive_rag"

def index_stats() -> dict:
    return {"indexed_chunks": len(_chunks), "indexed_files": len({c["filename"] for c in _chunks})}

def clear_index():
    global _chunks, _vectors
    _chunks = []
    _vectors = None
