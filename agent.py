from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from config import settings
from models import Message
from prompts import build_system_prompt
from tools import TOOLS, ToolDef


@dataclass
class AgentResult:
    """Framework-agnostic result from an agent run."""

    output: str
    escalated: bool = False


class AgentRunner(ABC):
    """Abstract agent execution engine.

    Implementations wrap a specific framework (LangChain, LlamaIndex, etc.)
    to run a tool-calling agent loop.
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


class LangChainGeminiAgentRunner(AgentRunner):
    """ReAct agent runner using LangChain with Google Gemini."""

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

        # Bail early before constructing anything if there is nothing to process.
        if not messages:
            return AgentResult(output="", escalated=False)

        lc_tools = [self._convert_tool(t) for t in tools]
        agent = create_agent(self._llm, lc_tools)
        all_messages = self._convert_messages(messages, system_prompt)

        print("\n" + "=" * 60)
        print("🤖 ReAct Loop Started")
        print("=" * 60)
        print(f"📝 User Input: {messages[-1]['content']}")
        print(f"📚 Total Messages: {len(all_messages)}")
        print("-" * 60)

        result = agent.invoke({"messages": all_messages})

        # Extract the final text from the last message in the result.
        output_messages = result.get("messages", [])
        if not output_messages:
            return AgentResult(output="", escalated=False)
        final_message = output_messages[-1]

        if hasattr(final_message, "content"):
            content = final_message.content
            if isinstance(content, list) and len(content) > 0:
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

        # Trace each tool call and detect whether escalation was triggered.
        print("\n🔄 ReAct Loop Trace:")
        tool_calls_made = []
        escalated = False

        for msg in output_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_input = tool_call.get("args", {})
                    tool_calls_made.append((tool_name, tool_input))

                    print(f"\n  🧠 Reasoning → Action: Call tool '{tool_name}'")
                    print(f"    📥 Tool Input: {tool_input}")

                    if tool_name == "escalate_to_human":
                        escalated = True

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


class Agent:
    """Chat agent with configurable prompts, tools, and agentic execution."""

    def __init__(
        self,
        runner: AgentRunner,
        system_prompt: str | None = None,
        tools: list[ToolDef] | None = None,
    ):
        """
        Args:
            runner: AgentRunner instance for executing the agentic loop.
            system_prompt: Optional system prompt to configure agent behavior.
            tools: Optional list of tool definitions for function calling.
        """
        self._runner = runner
        self._system_prompt = system_prompt or ""
        self._tools = tools or []

    def process_chat(
        self, message: str, history: list[Message] | None = None
    ) -> tuple[list[Message], bool]:
        """Process a chat message through the agentic loop.

        Args:
            message: The user's message.
            history: Optional conversation history.

        Returns:
            Tuple of (full conversation messages, escalate flag).
        """
        # Build conversation as raw dicts (runner handles framework conversion).
        conversation = []
        if history:
            conversation.extend([msg.model_dump() for msg in history])
        conversation.append({"role": "user", "content": message})

        result: AgentResult = self._runner.run(
            messages=conversation,
            system_prompt=self._system_prompt,
            tools=self._tools,
        )

        # Reconstruct full conversation history with the assistant's response.
        output_messages = []
        if history:
            output_messages.extend(history)
        output_messages.append(Message(role="user", content=message))
        output_messages.append(Message(role="assistant", content=result.output))

        return output_messages, result.escalated


runner = LangChainGeminiAgentRunner(
    model=settings.default_model,
    google_api_key=settings.google_api_key,
)

agent = Agent(runner=runner, system_prompt=build_system_prompt(), tools=TOOLS)
