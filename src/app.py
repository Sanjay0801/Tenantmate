from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from src.rag import search_chunks, build_prompt, call_claude
 
#Load environment variables
load_dotenv()
 
#DB Config 
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}
 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
 
#Create FastAPI app 
app = FastAPI(
    title="TenantMate",
    description="A chatbot that answers NSW residential tenancy law questions.",
    version="1.0.0"
)
 
#Load model once at startup (not on every request)
#This avoids reloading the model for every question — much faster
model = SentenceTransformer(EMBEDDING_MODEL)
 
 
#Request and Response models 
class QuestionRequest(BaseModel):
    """What the user sends to the API."""
    question: str
 
 
class AnswerResponse(BaseModel):
    """What the API sends back."""
    question: str
    answer: str
 
 

@app.get("/")
def root():
    """
    Health check — confirms the API is running.
    Open http://127.0.0.1:8000/ in your browser to check.
    """
    return {"status": "ok", "message": "TenantMate is running!"}
 
 
#chat endpoint
@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    """
    The main endpoint. Receives a question and returns an answer.
 
    Example request:
        POST /chat
        {"question": "Can my landlord enter without notice?"}
 
    Example response:
        {
            "question": "Can my landlord enter without notice?",
            "answer": "No, per Section 55..."
        }
    """
 
    #Validate the question is not empty
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
 
    try:
        #Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
 
        #Search for relevant sections
        chunks = search_chunks(request.question, conn, model)
 
        #Build prompt
        prompt = build_prompt(request.question, chunks)
 
        #Call Claude
        answer = call_claude(prompt)
 
        conn.close()
 
        return AnswerResponse(
            question=request.question,
            answer=answer
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 