import os
import logging
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Standard CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows connections from any domain
    allow_credentials=False, # Must be False when using wildcard "*"
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

# FIX 1: Hardcoded OPTIONS handler to force browser preflight success
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, DELETE, PUT"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
    return response

@app.get("/")
async def root():
    return {"status": "LedgerLens API is live"}

@app.post("/query")
async def process_query(request: ChatRequest):
    try:
        # EXECUTE DIRECT FAST RAG
        response = engine.ask(request.message)
        
        result_data = {
            "answer": str(response),
            "sources": [n.node.get_content()[:200] + "..." for n in response.source_nodes]
        }
        
        api_response = JSONResponse(content=result_data)
        api_response.headers["Access-Control-Allow-Origin"] = "*"
        return api_response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_response = JSONResponse(
            content={"answer": f"Internal Error: {str(e)}", "sources": []},
            status_code=500
        )
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        return error_response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
