from dataclasses import dataclass
from typing import Any, Callable

from embeddings import vector_store
from logger import logger
from models import SearchResult


@dataclass
class ToolDef:
    """Framework-agnostic tool definition."""

    name: str
    description: str
    fn: Callable[..., str]
    parameters: dict[str, Any]


def knowledge_base_search(query: str) -> str:
    """Search the telecommunications company knowledge base for information about
    service plans, billing, and troubleshooting.

    Args:
        query: The search query describing what information to look up.

    Returns:
        Relevant knowledge base articles, or a message if nothing was found.
    """
    logger.debug(f"🔍 [TOOL EXECUTING] knowledgeBaseSearch")
    logger.debug(f"   Query: '{query}'")

    results: list[SearchResult] = vector_store.search_similar(query)

    if not results:
        logger.debug(f"   Results: ❌ No documents found")
        return "No relevant documents found in the knowledge base."

    logger.debug(f"   Results: ✅ Found {len(results)} documents")
    for r in results:
        logger.debug(f"     - [{r.category or 'Unknown'}] similarity={r.similarity:.2f}")

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
    logger.debug(f"🚨 [TOOL EXECUTING] escalate_to_human")
    logger.debug(f"   Reason: '{reason}'")
    return f"Escalation requested: {reason}"


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
