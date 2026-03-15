"""Evaluate chat agent quality against predefined test cases."""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent, LangChainGeminiAgentRunner  # noqa: E402  (after sys.path mutation)
from models import Message  # noqa: E402
from config import settings  # noqa: E402
from prompts import build_system_prompt  # noqa: E402
from tools import TOOLS  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test_cases(test_file: Path) -> list[dict]:
    """Load and validate test cases from JSON file.

    Args:
        test_file: Path to chat_test_cases.json

    Returns:
        List of valid test case dicts.
    """
    try:
        with open(test_file) as f:
            raw = json.load(f)
    except FileNotFoundError:
        print(f"Error: test file not found: {test_file}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: could not parse {test_file}: {exc}")
        sys.exit(1)

    valid: list[dict] = []
    required_top = {"id", "category", "message", "expected"}
    required_expected = {
        "tools_called",
        "escalate",
        "response_should_contain",
        "response_should_not_contain",
    }

    for i, case in enumerate(raw):
        missing_top = required_top - case.keys()
        if missing_top:
            print(f"Warning: skipping case #{i} — missing fields: {missing_top}")
            continue

        expected = case.get("expected", {})
        missing_exp = required_expected - expected.keys()
        if missing_exp:
            print(
                f"Warning: skipping case '{case.get('id', i)}' — missing expected fields: {missing_exp}"
            )
            continue

        valid.append(case)

    return valid


def parse_history(history_data: list[dict] | None) -> list[Message] | None:
    """Convert raw JSON history to Message objects.

    Args:
        history_data: List of ``{"role": ..., "content": ...}`` dicts or None.

    Returns:
        List of Message objects, or None if history is absent / malformed.
    """
    if not history_data:
        return None

    messages: list[Message] = []
    for item in history_data:
        try:
            messages.append(Message(role=item["role"], content=item["content"]))
        except Exception:
            return None

    return messages or None


def evaluate_tools_match(
    expected: list[str], actual: list[str]
) -> tuple[bool, str | None]:
    """Compare expected vs. actual tools using set equality (order-independent).

    Args:
        expected: Tool names the test case expects to be called.
        actual: Tool names actually called by the agent.

    Returns:
        (is_match, failure_reason)
    """
    expected_set = set(expected)
    actual_set = set(actual)

    if expected_set == actual_set:
        return True, None

    reasons: list[str] = []
    missing = expected_set - actual_set
    unexpected = actual_set - expected_set
    if missing:
        reasons.append(f"Missing tools: {sorted(missing)}")
    if unexpected:
        reasons.append(f"Unexpected tools: {sorted(unexpected)}")

    return False, "; ".join(reasons)


def evaluate_escalation_match(expected: bool, actual: bool) -> tuple[bool, str | None]:
    """Compare expected vs. actual escalation flag.

    Args:
        expected: Whether the test case expects escalation.
        actual: Whether the agent actually escalated.

    Returns:
        (is_match, failure_reason)
    """
    if expected == actual:
        return True, None

    if expected and not actual:
        reason = "Expected escalation but agent did not escalate"
    else:
        reason = "Agent escalated unexpectedly"

    return False, reason


def evaluate_content_match(
    response: str,
    should_contain: list[str],
    should_not_contain: list[str],
) -> tuple[bool, list[str]]:
    """Check response for required and forbidden substrings (case-insensitive).

    Args:
        response: The agent's response text.
        should_contain: Substrings that must appear in the response.
        should_not_contain: Substrings that must not appear in the response.

    Returns:
        (all_passed, list_of_failures)
    """
    lower_response = response.lower()
    failures: list[str] = []

    for phrase in should_contain:
        if phrase.lower() not in lower_response:
            failures.append(f"Missing required phrase: '{phrase}'")

    for phrase in should_not_contain:
        if phrase.lower() in lower_response:
            failures.append(f"Found forbidden phrase: '{phrase}'")

    return len(failures) == 0, failures


def evaluate_single_case(test_case: dict, agent_instance: Agent) -> dict:
    """Run a single test case through the agent and evaluate the result.

    Args:
        test_case: Validated test case dict.
        agent_instance: Instantiated Agent to call.

    Returns:
        Detailed result dict with pass/fail per check.
    """
    case_id = test_case["id"]
    category = test_case["category"]
    message = test_case["message"]
    expected = test_case["expected"]

    history = parse_history(test_case.get("history"))

    # --- Call the agent ---
    error: str | None = None
    response_text = ""
    actual_tools: list[str] = []
    actual_escalated = False

    try:
        messages, actual_escalated, actual_tools = agent_instance.process_chat(
            message, history
        )
        # The last message is the assistant reply
        last = messages[-1]
        response_text = last.content if hasattr(last, "content") else str(last)
    except Exception as exc:
        error = str(exc)

    # --- Run checks ---
    tools_passed, tools_reason = evaluate_tools_match(
        expected["tools_called"], actual_tools
    )
    esc_passed, esc_reason = evaluate_escalation_match(
        expected["escalate"], actual_escalated
    )
    content_passed, content_failures = evaluate_content_match(
        response_text,
        expected["response_should_contain"],
        expected["response_should_not_contain"],
    )

    overall_passed = tools_passed and esc_passed and content_passed and error is None

    return {
        "id": case_id,
        "category": category,
        "message": message,
        "passed": overall_passed,
        "checks": {
            "tools": {
                "passed": tools_passed,
                "expected": expected["tools_called"],
                "actual": actual_tools,
                "reason": tools_reason,
            },
            "escalation": {
                "passed": esc_passed,
                "expected": expected["escalate"],
                "actual": actual_escalated,
                "reason": esc_reason,
            },
            "content": {
                "passed": content_passed,
                "failures": content_failures,
            },
        },
        "response": response_text,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_chat(
    test_cases_file: Path, output_file: Path, agent_instance: Agent
) -> None:
    """Run full chat agent evaluation and write results.

    Args:
        test_cases_file: Path to chat_test_cases.json.
        output_file: Path to write eval_results_chat.json.
        agent_instance: Agent to evaluate.
    """
    print("=" * 70)
    print("CHAT AGENT EVALUATION")
    print("=" * 70)
    print(f"Test Cases: {test_cases_file}")

    test_cases = load_test_cases(test_cases_file)
    total = len(test_cases)
    print(f"Total Test Cases: {total}\n")

    # Per-category accumulators
    cat_stats: dict[str, dict] = defaultdict(
        lambda: {
            "total_cases": 0,
            "passed_cases": 0,
            "tools_correct": 0,
            "escalation_correct": 0,
            "content_correct": 0,
        }
    )

    detailed_results: list[dict] = []

    for idx, case in enumerate(test_cases, start=1):
        case_id = case["id"]
        category = case["category"]

        print(f"Testing [{idx}/{total}] {case_id}...", end=" ", flush=True)
        result = evaluate_single_case(case, agent_instance)
        detailed_results.append(result)

        # Accumulate stats
        stats = cat_stats[category]
        stats["total_cases"] += 1
        if result["passed"]:
            stats["passed_cases"] += 1
        if result["checks"]["tools"]["passed"]:
            stats["tools_correct"] += 1
        if result["checks"]["escalation"]["passed"]:
            stats["escalation_correct"] += 1
        if result["checks"]["content"]["passed"]:
            stats["content_correct"] += 1

        print("pass" if result["passed"] else "FAIL")

    # --- Build metrics ---
    metrics_by_category: dict[str, dict] = {}
    for cat, stats in cat_stats.items():
        n = stats["total_cases"]
        metrics_by_category[cat] = {
            "total_cases": n,
            "passed_cases": stats["passed_cases"],
            "accuracy": stats["passed_cases"] / n if n else 0.0,
            "tool_accuracy": stats["tools_correct"] / n if n else 0.0,
            "escalation_accuracy": stats["escalation_correct"] / n if n else 0.0,
            "content_accuracy": stats["content_correct"] / n if n else 0.0,
        }

    # Overall (aggregate)
    overall_total = sum(s["total_cases"] for s in cat_stats.values())
    overall_passed = sum(s["passed_cases"] for s in cat_stats.values())
    overall_tools = sum(s["tools_correct"] for s in cat_stats.values())
    overall_esc = sum(s["escalation_correct"] for s in cat_stats.values())
    overall_content = sum(s["content_correct"] for s in cat_stats.values())

    overall_metrics = {
        "total_cases": overall_total,
        "passed_cases": overall_passed,
        "accuracy": overall_passed / overall_total if overall_total else 0.0,
        "tool_accuracy": overall_tools / overall_total if overall_total else 0.0,
        "escalation_accuracy": overall_esc / overall_total if overall_total else 0.0,
        "content_accuracy": overall_content / overall_total if overall_total else 0.0,
    }

    # --- Print summary table ---
    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print()

    header = f"{'Category':<30} {'Accuracy':<11} {'Tool Acc':<11} {'Esc Acc':<11} {'Content':<11} {'Cases'}"
    print(header)
    print("-" * 78)

    for cat in sorted(metrics_by_category.keys()):
        m = metrics_by_category[cat]
        row = (
            f"{cat:<30} "
            f"{m['accuracy']:.1%}    "
            f"   {m['tool_accuracy']:.1%}    "
            f"   {m['escalation_accuracy']:.1%}    "
            f" {m['content_accuracy']:.1%}    "
            f" {m['passed_cases']}/{m['total_cases']}"
        )
        print(row)

    print("-" * 78)
    m = overall_metrics
    row = (
        f"{'OVERALL':<30} "
        f"{m['accuracy']:.1%}    "
        f"   {m['tool_accuracy']:.1%}    "
        f"   {m['escalation_accuracy']:.1%}    "
        f" {m['content_accuracy']:.1%}    "
        f" {m['passed_cases']}/{m['total_cases']}"
    )
    print(row)
    print("=" * 70)
    print()

    # --- Build output JSON ---
    output_data = {
        "metrics": {
            "overall": overall_metrics,
            "by_category": metrics_by_category,
        },
        "detailed_results": detailed_results,
        "summary": {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "test_file": str(test_cases_file),
            "total_categories": len(metrics_by_category),
        },
        "metadata": {
            "model": settings.default_model,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "similarity_threshold": settings.similarity_threshold,
            "max_search_results": settings.max_search_results,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "use_chunked_storage": settings.use_chunked_storage,
            "environment": settings.environment,
        },
    }

    try:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Detailed results saved to {output_file}\n")
    except OSError as exc:
        print(
            f"Warning: could not write output file ({exc}). Printing JSON to stdout:\n"
        )
        print(json.dumps(output_data, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    eval_dir = Path(__file__).parent
    test_cases_file = eval_dir / "chat_test_cases.json"
    output_file = eval_dir / "eval_results_chat.json"

    # Create agent with caching disabled for evaluation
    eval_runner = LangChainGeminiAgentRunner(
        model=settings.default_model,
        google_api_key=settings.google_api_key,
        disable_cache=True,
    )
    eval_agent = Agent(
        runner=eval_runner,
        system_prompt=build_system_prompt(),
        tools=TOOLS,
    )

    evaluate_chat(test_cases_file, output_file, eval_agent)
