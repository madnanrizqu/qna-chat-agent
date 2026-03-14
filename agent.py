from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from config import settings
from embeddings import vector_store
from models import Message, SearchResult


@dataclass
class ToolDef:
    """Framework-agnostic tool definition."""

    name: str
    description: str
    fn: Callable[..., str]
    parameters: dict[str, Any]


@dataclass
class AgentResult:
    """Framework-agnostic result from an agent run."""

    output: str
    escalated: bool = False


class AgentRunner(ABC):
    """Abstract agent execution engine.

    Implementations wrap a specific framework (LangChain, LlamaIndex, etc.)
    to run a ReAct-style tool-calling loop.
    """

    @abstractmethod
    def run(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        tools: list[ToolDef],
    ) -> AgentResult:
        """Execute the agent loop and return a result.

        Args:
            messages: Conversation history as list of {"role": ..., "content": ...} dicts.
            system_prompt: The system prompt string.
            tools: List of framework-agnostic tool definitions.

        Returns:
            AgentResult with the assistant's final output and escalation flag.
        """
        ...


class LangChainAgentRunner(AgentRunner):
    """ReAct agent runner using LangChain."""

    def __init__(self, model: str, google_api_key: str):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key,
        )

    def _convert_tool(self, tool_def: ToolDef) -> Any:
        """Convert a ToolDef into a LangChain StructuredTool."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=tool_def.fn,
            name=tool_def.name,
            description=tool_def.description,
        )

    def _convert_messages(
        self, messages: list[dict[str, str]], system_prompt: str
    ) -> list:
        """Convert raw message dicts to LangChain message objects."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        return lc_messages

    def run(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        tools: list[ToolDef],
    ) -> AgentResult:
        from langchain.agents import create_agent

        lc_tools = [self._convert_tool(t) for t in tools]

        # Create the ReAct agent (modern LangChain API)
        agent = create_agent(self._llm, lc_tools)

        # Separate the last user message from history
        if not messages:
            return AgentResult(output="", escalated=False)

        # Build full conversation including system prompt
        all_messages = self._convert_messages(messages, system_prompt)

        print("\n" + "=" * 60)
        print("🤖 ReAct Loop Started")
        print("=" * 60)
        print(f"📝 User Input: {messages[-1]['content']}")
        print(f"📚 Total Messages: {len(all_messages)}")
        print("-" * 60)

        # Invoke the agent with messages
        result = agent.invoke({"messages": all_messages})

        # Extract the final message from the agent
        output_messages = result.get("messages", [])
        if not output_messages:
            return AgentResult(output="", escalated=False)

        # The last message should be the agent's response
        final_message = output_messages[-1]

        # Extract text from the message content
        # Google's API returns content as a list of dicts with 'type' and 'text' fields
        if hasattr(final_message, "content"):
            content = final_message.content
            if isinstance(content, list) and len(content) > 0:
                # Extract text from the first content part
                if isinstance(content[0], dict) and "text" in content[0]:
                    output_text = content[0]["text"]
                else:
                    output_text = str(content)
            elif isinstance(content, str):
                output_text = content
            else:
                output_text = str(content)
        else:
            output_text = str(final_message)

        # Print intermediate steps (the ReAct loop trace)
        print("\n🔄 ReAct Loop Trace:")
        tool_calls_made = []
        escalated = False

        # LangGraph stores tool calls in the message history
        for msg in output_messages:
            # Check for tool calls (AIMessage with tool_calls)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_input = tool_call.get("args", {})
                    tool_calls_made.append((tool_name, tool_input))

                    print(f"\n  🧠 Reasoning → Action: Call tool '{tool_name}'")
                    print(f"    📥 Tool Input: {tool_input}")

                    if tool_name == "escalate_to_human":
                        escalated = True

            # Check for tool messages (responses from tools)
            if hasattr(msg, "type") and msg.type == "tool":
                tool_output = msg.content
                print(
                    f"    📤 Tool Output: {tool_output[:100]}{'...' if len(tool_output) > 100 else ''}"
                )

        if not tool_calls_made:
            print("  ℹ️  No tools were called (direct answer)")

        print(
            f"\n✅ Final Answer: {output_text[:100]}{'...' if len(output_text) > 100 else ''}"
        )
        print(f"🚨 Escalated: {escalated}")
        print("=" * 60 + "\n")

        return AgentResult(output=output_text, escalated=escalated)


# Tool functions (plain Python, framework-agnostic)


def knowledge_base_search(query: str) -> str:
    """Search the telecommunications company knowledge base for information about
    service plans, billing, and troubleshooting.

    Args:
        query: The search query describing what information to look up.

    Returns:
        Relevant knowledge base articles, or a message if nothing was found.
    """
    print(f"\n    🔍 [TOOL EXECUTING] knowledgeBaseSearch")
    print(f"       Query: '{query}'")

    results: list[SearchResult] = vector_store.search_similar(query)

    if not results:
        print(f"       Results: ❌ No documents found")
        return "No relevant documents found in the knowledge base."

    print(f"       Results: ✅ Found {len(results)} documents")
    for r in results:
        print(f"         - [{r.category or 'Unknown'}] similarity={r.similarity:.2f}")

    return "\n\n".join(
        f"[{r.category or 'Unknown'}] (similarity: {r.similarity:.2f}): {r.content}"
        for r in results
    )


def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent.

    Use this tool when you cannot answer the customer's question confidently,
    or when the knowledge base search returned no relevant results.

    Args:
        reason: Brief explanation of why escalation is needed.

    Returns:
        Confirmation message.
    """
    print(f"\n    🚨 [TOOL EXECUTING] escalate_to_human")
    print(f"       Reason: '{reason}'")
    return f"Escalation requested: {reason}"


# Tool definitions
TOOLS = [
    ToolDef(
        name="knowledgeBaseSearch",
        description=(
            "Search the telecommunications company knowledge base for information "
            "about service plans, billing policies, and troubleshooting guides. "
            "Do NOT use for general knowledge questions."
        ),
        fn=knowledge_base_search,
        parameters={"query": {"type": "string", "description": "Search query"}},
    ),
    ToolDef(
        name="escalate_to_human",
        description=(
            "Escalate to a human agent when you cannot answer confidently or "
            "when knowledge base search returns no results."
        ),
        fn=escalate_to_human,
        parameters={
            "reason": {"type": "string", "description": "Reason for escalation"}
        },
    ),
]


class Agent:
    """Chat agent with configurable prompts, tools, and agentic execution.

    Handles conversation processing with the AI model, including:
    - System prompts for agent behavior
    - Tool/function calling capabilities via ReAct loop
    - Conversation history management
    """

    def __init__(
        self,
        runner: AgentRunner,
        system_prompt: str | None = None,
        tools: list[ToolDef] | None = None,
    ):
        """Initialize the agent with optional configuration.

        Args:
            runner: AgentRunner instance for executing the agentic loop
            system_prompt: Optional system prompt to configure agent behavior
            tools: Optional list of tool definitions for function calling
        """
        self._runner = runner
        self._system_prompt = system_prompt or ""
        self._tools = tools or []

    def process_chat(
        self, message: str, history: list[Message] | None = None
    ) -> tuple[list[Message], bool]:
        """Process a chat message through the agentic loop.

        Args:
            message: The user's message
            history: Optional conversation history

        Returns:
            Tuple of (full conversation messages, escalate flag)

        Raises:
            Exception: If there's an error communicating with the AI
        """
        # Build conversation as raw dicts (runner handles framework conversion)
        conversation = []
        if history:
            conversation.extend([msg.model_dump() for msg in history])
        conversation.append({"role": "user", "content": message})

        # Run the agent
        result: AgentResult = self._runner.run(
            messages=conversation,
            system_prompt=self._system_prompt,
            tools=self._tools,
        )

        # Build return value matching the original signature
        output_messages = []
        if history:
            output_messages.extend(history)
        output_messages.append(Message(role="user", content=message))
        output_messages.append(Message(role="assistant", content=result.output))

        return output_messages, result.escalated


SYSTEM_PROMPT = (
    "You are a helpful assistant that handles inbound customer questions via chat — "
    "answering questions about service plans, billing, and basic troubleshooting.\n"
    "Your primary goal is to answer the user's question accurately.\n\n"
    "TOOL USAGE GUIDE:\n"
    "1. Use 'knowledgeBaseSearch' ONLY IF the question is about the Telecommunication "
    "company, its billing policy, service plans, or troubleshooting guide. DO NOT use "
    "this tool for general knowledge, current events, or unrelated topics.\n"
    "2. Answer directly IF you already know the answer or the question is conversational.\n"
    "3. Use 'escalate_to_human' when you cannot answer confidently, or when "
    "knowledgeBaseSearch returns no relevant results for a company-specific question.\n\n"
    "After using a tool, integrate its findings into a concise and helpful response."
)


def build_system_prompt() -> str:
    """Return the system prompt for the agent."""
    return SYSTEM_PROMPT


# Module-level wiring
runner = LangChainAgentRunner(
    model=settings.default_model,
    google_api_key=settings.google_api_key,
)

agent = Agent(runner=runner, system_prompt=build_system_prompt(), tools=TOOLS)
