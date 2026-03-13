from fastapi import FastAPI, HTTPException

from ai import ai_client
from config import settings
from models import Message, ChatRequest, ChatResponse

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
        # Build conversation from history (if provided) + current message
        conversation = []
        if request.history:
            conversation = [msg.model_dump() for msg in request.history]

        # Add current user message
        conversation.append({"role": "user", "content": request.message})

        client = ai_client.get_client()
        completion = client.chat.completions.create(
            model=settings.default_model,
            messages=conversation,
        )

        ai_response_content = completion.choices[0].message.content

        assistant_message = Message(role="assistant", content=ai_response_content)

        # Build full conversation for response
        full_messages = (request.history or []) + [
            Message(role="user", content=request.message),
            assistant_message,
        ]

        return ChatResponse(messages=full_messages, escalate=False)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )
