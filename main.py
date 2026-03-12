import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


def get_openai_client():
    """Get configured OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    print(api_key)
    print(base_url)

    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    if not base_url:
        raise HTTPException(status_code=500, detail="OPENAI_BASE_URL not configured")

    return OpenAI(api_key=api_key, base_url=base_url)


class QuestionRequest(BaseModel):
    question: str
    model: str = "gemini-3.1-flash-lite-preview"


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
