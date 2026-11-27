# Semantic Search & RAG Prototype (GenAI)

A local Retrieval-Augmented Generation (RAG) prototype. Upload PDF or text documents, split them into chunks, create sentence-transformer embeddings, retrieve relevant passages with cosine similarity, and optionally generate an answer using a local Ollama model.

## Architecture

```text
PDF / TXT upload
      |
      v
Text extraction (pypdf)
      |
      v
Chunking with overlap
      |
      v
Sentence Transformers embeddings
      |
      v
Vector index (NumPy cosine similarity)
      |
      v
Top-k relevant passages
      |
      +----> Local Ollama (optional answer generation)
      |
      v
Answer + source references
```

## Features
- Upload PDF and TXT files
- Text extraction and overlapping chunking
- Local embeddings using `all-MiniLM-L6-v2`
- Semantic retrieval using cosine similarity
- Source file and page/chunk references
- Optional local answer generation through Ollama
- Works without a paid API key; without Ollama it returns a grounded context-based response
- FastAPI Swagger UI

## Requirements
- Python 3.10+
- 4 GB+ RAM recommended
- First embedding-model run requires internet to download the model
- Optional: Ollama installed locally for generative answers

## Run locally

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

## Quick start
1. Call `POST /documents/upload` with a PDF or TXT file.
2. Call `POST /ask` with a question.
3. Inspect `answer`, `sources`, and retrieved chunk scores.

Example request:
```json
{
  "question": "What are the main responsibilities described in the document?",
  "top_k": 4
}
```

## Optional local LLM with Ollama
Install Ollama, then pull a model:
```bash
ollama pull llama3.2:3b
```
Keep Ollama running and set `OLLAMA_MODEL=llama3.2:3b` in `.env`. The app will attempt local generation. If Ollama is unavailable, it safely falls back to returning retrieved context excerpts rather than inventing an answer.

## Run tests
```bash
pytest -q
```

## Project structure
```text
semantic-search-rag-prototype/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── rag.py
│   └── schemas.py
├── tests/
│   └── test_chunking.py
├── storage/.gitkeep
├── .env.example
├── .gitignore
└── requirements.txt
```

## Limitations and next steps
This is a prototype, not a production multi-user document service. The in-memory vector index resets when the server restarts. Add persistent vector storage, authentication, background ingestion, document-level access control and monitoring before production use. Ollama generation is optional and runs on your own machine.
