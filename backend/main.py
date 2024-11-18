from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os
from services.chat_service import BuildingMaterialsChatService
import time
# Load environment variables
load_dotenv()

# Initialize BuildingMaterialsChatService with API key
chat_service = BuildingMaterialsChatService(api_key=os.getenv('OPENAI_API_KEY'))

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str
    id: str = ""
    createTime: int = 0

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/welcome")
async def welcome():
    """Return the welcome message"""
    return {
        "id": str(time.time()),
        "role": "assistant",
        "content": "Hi! I'm BuildMate, your expert building materials assistant, ready to help you make informed decisions about construction materials, specifications, and best practices while keeping safety and sustainability in mind.",
        "createTime": int(time.time() * 1000),
        "status": "success"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        messages = [msg.dict() for msg in request.messages]
        return chat_service.get_chat_response(messages)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)