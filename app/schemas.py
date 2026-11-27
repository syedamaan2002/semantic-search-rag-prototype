from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)

class SourceItem(BaseModel):
    filename: str
    chunk_id: int
    score: float
    excerpt: str

class AskResponse(BaseModel):
    answer: str
    generation: str
    sources: list[SourceItem]
