from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import asyncio
import json
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    conversation_id: str = None
    history: List[Dict[str, Any]] = []

@app.get("/health")
def health():
    return {"status": "ok", "bot_initialized": True}

@app.post("/chat")
async def chat(request: ChatRequest):
    # Regular JSON response for compatibility
    await asyncio.sleep(0.5)
    return {
        "answer": f"This is a mock response for testing purposes. Your query was: {request.query}",
        "conversation_id": request.conversation_id or "test-conversation-id",
        "citations": [{"id": "1", "title": "Mock Medical Document", "content": "Test citation"}],
        "latency_s": 0.5
    }

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    # Streaming response for frontend compatibility
    async def generate():
        # Simulate thinking time
        await asyncio.sleep(0.5)
        
        # Mock response text
        response_text = f"This is a mock response for testing purposes. Your query was: {request.query}"
        
        # Stream tokens one by one
        words = response_text.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            event_data = json.dumps({"token": token}) + "\n\n"
            yield f"data: {event_data}"
            await asyncio.sleep(0.05)  # Small delay between tokens
        
        # Send done event with metadata
        done_data = json.dumps({
            "done": True,
            "answer": response_text,
            "conversation_id": request.conversation_id or "test-conversation-id",
            "citations": [{"id": "1", "title": "Mock Medical Document", "content": "Test citation"}],
            "latency": 0.5
        }) + "\n\n"
        yield f"data: {done_data}"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
