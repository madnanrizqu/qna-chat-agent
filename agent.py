from ai import ai_client
from config import settings
from models import Message


def process_chat(message: str, history: list[Message] | None = None) -> list[Message]:
    """
    Process a chat message with the AI assistant.

    Args:
        message: The user's message
        history: Optional conversation history

    Returns:
        Full conversation including the new user message and AI response

    Raises:
        Exception: If there's an error communicating with the AI
    """
    # Build conversation from history (if provided) + current message
    conversation = []
    if history:
        conversation = [msg.model_dump() for msg in history]

    # Add current user message
    conversation.append({"role": "user", "content": message})

    # Call AI client
    client = ai_client.get_client()
    completion = client.chat.completions.create(
        model=settings.default_model,
        messages=conversation,
    )

    ai_response_content = completion.choices[0].message.content

    # Build assistant message
    assistant_message = Message(role="assistant", content=ai_response_content)

    # Build full conversation for response
    full_messages = (history or []) + [
        Message(role="user", content=message),
        assistant_message,
    ]

    return full_messages
