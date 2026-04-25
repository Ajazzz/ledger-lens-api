import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from upstash_redis import Redis

# Import your previous logic
from query_engine import LedgerLensQuery

# 1. Setup & Environment
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LedgerLens_Backend")

app = FastAPI(title="LedgerLens API")

# Enable CORS for your Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # In production, replace with your specific Replit URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Connect to Upstash Redis for Memory
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"), 
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

# Initialize the Query Engine
engine = LedgerLensQuery()

# 3. Data Models
class ChatRequest(BaseModel):
    user_id: str = "default_user"
    message: str

# 4. API Endpoints
@app.get("/")
async def root():
    return {"status": "LedgerLens API is live"}

@app.post("/query")
async def process_query(request: ChatRequest):
    try:
        # A. Fetch previous context from Upstash (Last 3 messages)
        history_key = f"chat_history:{request.user_id}"
        previous_context = redis.lrange(history_key, 0, 2)
        
        # B. Construct Enhanced Query with Memory
        full_query = f"Context from previous conversation: {previous_context}\nCurrent Question: {request.message}"
        
        # C. Execute RAG
        response = engine.ask(full_query)
        
        # D. Save to Memory (Push to Redis)
        redis.lpush(history_key, f"User: {request.message}", f"AI: {str(response)}")
        redis.ltrim(history_key, 0, 10) # Keep only last 10 messages to save costs
        
        return {
            "answer": str(response),
            "sources": [n.node.get_content()[:200] + "..." for n in response.source_nodes]
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Financial Analysis Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
