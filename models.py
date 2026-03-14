from pydantic import BaseModel, Field
from typing import Literal, Any
from uuid import UUID


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

    messages: list[Message] = Field(
        description=(
            "Complete conversation history including the current user message "
            "and your assistant response."
        )
    )

    escalate: bool = Field(
        description=("Whether this conversation should be escalated to a human agent.")
    )


class SearchResult(BaseModel):
    """Single search result with similarity score."""

    id: UUID
    content: str
    category: str | None = None
    similarity: float = Field(ge=0.0, le=1.0)
