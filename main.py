from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from config import settings

app = FastAPI()


def get_openai_client():
    """Get configured OpenAI client."""
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


class QuestionRequest(BaseModel):
    question: str
    model: str = settings.default_model


@app.get("/")
def read_root():
    return {"message": "QnA Chat Agent API", "status": "running"}


@app.post("/chat")
def ask_question(request: QuestionRequest):
    """Ask a question to the AI model."""
    try:
        client = get_openai_client()
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
