from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from app.rag import add_document, search, generate_answer, index_stats
from app.schemas import AskRequest, AskResponse, SourceItem

app = FastAPI(
    title="Semantic Search & RAG Prototype",
    version="1.0.0",
    description="Upload documents, retrieve semantically relevant chunks and optionally generate answers with local Ollama."
)

@app.get("/")
def root():
    return {"message": "Semantic Search & RAG Prototype is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", **index_stats()}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded-document"
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in {"pdf", "txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB prototype limit.")
    try:
        if suffix == "pdf":
            reader = PdfReader(BytesIO(content))
            total_chunks = 0
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                total_chunks += add_document(filename, page_text, page_number)
        else:
            text = content.decode("utf-8-sig")
            total_chunks = add_document(filename, text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to process document: {exc}") from exc
    if total_chunks == 0:
        raise HTTPException(status_code=400, detail="No extractable text was found in this file.")
    return {"filename": filename, "chunks_indexed": total_chunks, **index_stats()}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    results = search(request.question, request.top_k)
    answer, generation = generate_answer(request.question, results)
    sources = [
        SourceItem(
            filename=r["filename"],
            chunk_id=r["chunk_id"],
            score=round(r["score"], 4),
            excerpt=r["text"][:350],
        ) for r in results
    ]
    return AskResponse(answer=answer, generation=generation, sources=sources)

@app.get("/stats")
def stats():
    return index_stats()
