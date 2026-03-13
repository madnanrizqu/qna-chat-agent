from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai import ai_client
from config import settings

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "QnA Chat Agent API", "status": "running"}


class QuestionRequest(BaseModel):
    question: str
    model: str = settings.default_model


@app.post("/chat")
def ask_question(request: QuestionRequest):
    """Ask a question to the AI model."""
    try:
        client = ai_client.get_client()
        completion = client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.question}],
        )
        response = completion.choices[0].message.content
        return {"question": request.question, "answer": response}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
