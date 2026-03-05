from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import run_agent, suggest_questions  # ✅ single import at top

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    query: str

class SuggestRequest(BaseModel):
    topic: str
    summary: str

@app.post("/research")
async def research(request: QueryRequest):
    result = run_agent(request.query)
    return result

@app.post("/suggest")
async def suggest(request: SuggestRequest):
    questions = suggest_questions(request.topic, request.summary)
    return {"questions": questions}

@app.get("/")
async def root():
    return {"message": "Research chatbot API is running!"}