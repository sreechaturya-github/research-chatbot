import os
print("Files in directory:", os.listdir("."))  # ✅ debug line
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from main import run_agent, suggest_questions

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
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))