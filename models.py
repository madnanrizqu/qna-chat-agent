from pydantic import BaseModel, Field
from typing import Literal


class Message(BaseModel):
    """Single message in a conversation."""

    role: Literal["user", "system", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(min_length=1)
    history: list[Message] | None = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    messages: list[Message]
    escalate: bool
