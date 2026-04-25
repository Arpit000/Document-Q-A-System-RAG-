from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask_question

app = FastAPI(
    title="Local RAG Document QA",
    version="1.0"
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "RAG API is working"
    }

@app.post("/ask")
def ask(request: QuestionRequest):
    answer = ask_question(request.question)

    return {
        "question": request.question,
        "answer": answer
    }