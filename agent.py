from typing import Any, Type

from ai import AIClient, ai_client
from config import settings
from models import Message


class Agent:
    """Chat agent with configurable prompts, tools, and structured output.

    Handles conversation processing with the AI model, including:
    - System prompts for agent behavior
    - Tool/function calling capabilities
    - Conversation history management
    """

    def __init__(
        self,
        client: AIClient | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ):
        """Initialize the agent with optional configuration.

        Args:
            client: AI client instance (defaults to global ai_client)
            system_prompt: Optional system prompt to configure agent behavior
            tools: Optional list of tool definitions for function calling
            response_format: Optional Pydantic model for structured output
        """
        self._client = client or ai_client
        self._system_prompt = system_prompt
        self._tools = tools or []

    def process_chat(
        self, message: str, history: list[Message] | None = None
    ) -> list[Message]:
        """Process a chat message with the AI assistant.

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

        if self._system_prompt:
            conversation.append({"role": "system", "content": self._system_prompt})

        if history:
            conversation.extend([msg.model_dump() for msg in history])

        conversation.append({"role": "user", "content": message})

        api_params = {
            "model": settings.default_model,
            "messages": conversation,
        }

        if self._tools:
            api_params["tools"] = self._tools

        client = self._client.get_client()

        completion = client.chat.completions.create(**api_params)
        ai_response_content = completion.choices[0].message.content

        assistant_message = Message(role="assistant", content=str(ai_response_content))

        # Build full conversation for response (without system prompt)
        full_messages = (history or []) + [
            Message(role="user", content=message),
            assistant_message,
        ]

        return full_messages


agent = Agent()
