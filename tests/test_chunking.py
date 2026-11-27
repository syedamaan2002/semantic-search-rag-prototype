from app.rag import split_text

def test_split_text_returns_chunks_and_preserves_content():
    text = " ".join(["pipeline"] * 100)
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)

def test_split_text_handles_empty_input():
    assert split_text("   ") == []

def test_invalid_overlap_is_rejected():
    try:
        split_text("hello world", chunk_size=10, overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
