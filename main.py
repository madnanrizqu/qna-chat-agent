from fastapi import FastAPI, HTTPException

from agent import agent
from models import ChatRequest, ChatResponse

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "QnA Chat Agent API", "status": "running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with the AI assistant.

    Accepts a message string and optional conversation history,
    returns the full conversation including the AI's response.
    """
    try:
        messages, escalate = agent.process_chat(request.message, request.history)
        return ChatResponse(messages=messages, escalate=escalate)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )
