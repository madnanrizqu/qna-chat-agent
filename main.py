from fastapi import FastAPI, HTTPException
import uvicorn

from agent import agent
from logger import RequestIDMiddleware, logger
from models import ChatRequest, ChatResponse

app = FastAPI()
app.add_middleware(RequestIDMiddleware)


@app.get("/")
def read_root():
    logger.info("GET / - Request started")
    response = {"message": "QnA Chat Agent API", "status": "running"}
    logger.info("GET / - Request completed")
    return response


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with the AI assistant.

    Accepts a message string and optional conversation history,
    returns the full conversation including the AI's response.
    """
    logger.info(f"POST /chat - Request started | message='{request.message[:50]}...'")
    try:
        messages, escalate, _ = agent.process_chat(request.message, request.history)
        logger.info(f"POST /chat - Request completed")
        return ChatResponse(messages=messages, escalate=escalate)

    except Exception as e:
        logger.error(f"POST /chat - Request failed")
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )


# Running the main file directly is for debugger
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
