# Chat Agent Test Scenario Categories

## Purpose

This document explains the rationale behind the test scenario categories used to evaluate the QnA chat agent, and provides guidelines for creating new test cases.

## Agent Scope

The chat agent is a **telecommunications customer service assistant** designed to:

- Answer questions about service plans, billing, and basic troubleshooting
- Search the company knowledge base when needed
- Escalate to human agents when it cannot answer confidently

The agent has **two tools**:

1. `knowledge_base_search` - Searches vector store for company-specific information
2. `escalate_to_human` - Transfers conversation to a human agent

## Test Categories

### 1. Direct Conversational (`direct_conversational`)

**Purpose:** Verify the agent handles routine social interactions without unnecessary tool calls.

**Characteristics:**

- Simple greetings, thanks, goodbyes, casual check-ins
- No company-specific information needed
- Should answer directly using conversational knowledge

**Expected Behavior:**

- ✅ No tools called
- ✅ No escalation
- ✅ Natural, friendly response

**Example:**

```json
{
  "message": "Hello!",
  "expected": {
    "tools_called": [],
    "escalate": false
  }
}
```

**Why This Category:**

- Tests basic conversational ability
- Ensures agent doesn't over-use tools for simple interactions
- Critical for user experience (users expect friendly greetings)

---

### 2. Off-Topic (`off_topic`)

**Purpose:** Verify the agent correctly identifies non-company questions and avoids searching the KB.

**Characteristics:**

- General knowledge questions ("What is the capital of France?")
- Device recommendations ("Best phone for gaming?")
- Current events, weather, unrelated topics

**Expected Behavior:**

- ✅ No KB search (critical - searching wastes resources and degrades relevance)
- ✅ No escalation needed
- ✅ Polite deflection or brief answer without using tools

**Example:**

```json
{
  "message": "What's the best phone for gaming?",
  "expected": {
    "tools_called": [],
    "escalate": false,
    "response_should_not_contain": ["specific phone model"]
  }
}
```

**Why This Category:**

- Prevents wasted vector store searches
- Tests the agent's ability to distinguish company vs. general knowledge
- Ensures prompt instruction compliance ("DO NOT use this tool for general knowledge")

---

### 3. KB Retrieval - Billing (`kb_retrieval_billing`)

**Purpose:** Verify the agent correctly searches for and returns billing policy information.

**Characteristics:**

- Late payment fees, payment methods, auto-pay setup
- Billing disputes, charge disputes
- Company-specific billing policies

**Expected Behavior:**

- ✅ Calls `knowledge_base_search`
- ✅ No escalation (info is in KB)
- ✅ Response contains specific information from KB (e.g., "50,000" for late fee, "30 days" for disputes)

**Example:**

```json
{
  "message": "What is the late payment fee?",
  "expected": {
    "tools_called": ["knowledge_base_search"],
    "escalate": false,
    "response_should_contain": ["50,000"]
  }
}
```

**Why This Category:**

- Billing questions are common in telecom support
- Tests accurate KB retrieval and factual grounding
- Ensures agent returns precise policy information

---

### 4. KB Retrieval - Plans (`kb_retrieval_plans`)

**Purpose:** Verify the agent correctly retrieves service plan information.

**Characteristics:**

- Plan pricing (Basic, Standard, Pro)
- Plan features (data limits, call minutes)
- Plan comparison questions

**Expected Behavior:**

- ✅ Calls `knowledge_base_search`
- ✅ No escalation (info is in KB)
- ✅ Response contains specific plan details (e.g., "199,000" for Pro Plan)

**Example:**

```json
{
  "message": "How much does the Pro Plan cost?",
  "expected": {
    "tools_called": ["knowledge_base_search"],
    "escalate": false,
    "response_should_contain": ["199,000"]
  }
}
```

**Why This Category:**

- Plan inquiries are core to telecom customer service
- Tests retrieval of structured data (pricing, features)
- Verifies agent can differentiate between multiple plan types

---

### 5. KB Retrieval - Troubleshooting (`kb_retrieval_troubleshooting`)

**Purpose:** Verify the agent retrieves and provides basic technical support.

**Characteristics:**

- No signal issues, APN settings, SIM card problems
- Network connectivity troubleshooting
- Device configuration steps

**Expected Behavior:**

- ✅ Calls `knowledge_base_search`
- ✅ No escalation (basic troubleshooting is in KB)
- ✅ Response contains specific troubleshooting steps (e.g., "airplane mode", "APN")

**Example:**

```json
{
  "message": "I have no signal, what should I do?",
  "expected": {
    "tools_called": ["knowledge_base_search"],
    "escalate": false,
    "response_should_contain": ["airplane mode"]
  }
}
```

**Why This Category:**

- Troubleshooting is a primary use case for support agents
- Tests multi-step instruction retrieval
- Ensures agent provides actionable guidance

---

### 6. KB Retrieval - Not In KB (`kb_retrieval_not_in_kb`)

**Purpose:** Verify the agent searches first, then escalates when information is missing.

**Characteristics:**

- Company-related questions NOT in the knowledge base
- Examples: international roaming, number porting, family plans, eSIM, 5G coverage

**Expected Behavior:**

- ✅ Calls `knowledge_base_search` (attempts to find answer)
- ✅ Then calls `escalate_to_human` (when KB returns no results)
- ✅ Escalation flag set to true

**Example:**

```json
{
  "message": "Do you offer international roaming packages?",
  "expected": {
    "tools_called": ["knowledge_base_search", "escalate_to_human"],
    "escalate": true
  }
}
```

**Why This Category:**

- Tests the agent's fallback behavior
- Ensures agent doesn't hallucinate answers when info is missing
- Critical for maintaining accuracy and preventing misinformation
- Validates adherence to escalation policy: "when KB search returns no results"

---

### 7. Ambiguous/Vague (`ambiguous_vague`)

**Purpose:** Verify the agent handles unclear requests by asking for clarification.

**Characteristics:**

- Vague help requests ("Help me", "I have a problem")
- No specific question ("I need information", "Something is wrong")
- Requests lacking context

**Expected Behavior:**

- ✅ No tools called (nothing specific to search for)
- ✅ No escalation (just needs clarification)
- ✅ Response asks for more details

**Example:**

```json
{
  "message": "Help me",
  "expected": {
    "tools_called": [],
    "escalate": false
  }
}
```

**Why This Category:**

- Common real-world scenario (users often start vague)
- Tests agent's ability to guide conversation
- Prevents premature KB searches or escalations
- Ensures good UX by helping users articulate their needs

---

### 8. Escalation Explicit (`escalation_explicit`)

**Purpose:** Verify the agent respects direct requests for human agents.

**Characteristics:**

- Explicit requests: "I want to speak to a human", "Transfer me to a real person"
- Supervisor/manager requests
- Frustrated escalation: "This isn't helping, connect me to a live agent"

**Expected Behavior:**

- ✅ Calls `escalate_to_human` immediately
- ✅ No KB search needed
- ✅ Escalation flag set to true

**Example:**

```json
{
  "message": "I want to speak to a human agent",
  "expected": {
    "tools_called": ["escalate_to_human"],
    "escalate": true
  }
}
```

**Why This Category:**

- User autonomy - customers have the right to request human support
- Tests intent recognition for escalation keywords
- Ensures agent respects user preferences
- Critical for customer satisfaction (don't force chatbot on frustrated users)

---

## Categories NOT Included (and Why)

### Multi-Turn Conversations

**Why excluded:** The current test suite focuses on single-turn evaluation. Multi-turn testing requires conversation state management and is more complex to evaluate programmatically.

**Future consideration:** Add if context tracking becomes a priority.

### Emotional/Sentiment-Based Categories

**Why excluded:** Sentiment analysis is not in the agent's core requirements. The agent should handle all questions professionally regardless of user emotion.

**Note:** Some emotion is captured in `escalation_explicit` (frustrated users).

### Edge Cases (Profanity, Abuse, Spam)

**Why excluded:** These are safety/moderation concerns better handled by upstream filters or the LLM provider's safety features, not by the application layer.

### Complex Technical Issues (Account Deletion, Porting, Cancellation)

**Why excluded:** These are partially covered in `kb_retrieval_not_in_kb`. Complex account operations intentionally escalate to humans for security and compliance reasons.

### Transaction/Action-Based Categories

**Why excluded:** The agent is read-only (information retrieval only). It cannot perform actions like "change my plan" or "update payment method."

**Why this design:** Keeps agent simple and safe. Actions require authentication and are handled by human agents.

---

## Guidelines for Creating New Test Cases

### 1. Identify the Correct Category

Ask yourself:

- **Is it conversational/social?** → `direct_conversational`
- **Is it unrelated to the company?** → `off_topic`
- **Is it about billing?** → `kb_retrieval_billing`
- **Is it about service plans?** → `kb_retrieval_plans`
- **Is it about troubleshooting?** → `kb_retrieval_troubleshooting`
- **Is it company-related but not in KB?** → `kb_retrieval_not_in_kb`
- **Is it too vague to answer?** → `ambiguous_vague`
- **Is it an explicit request for human help?** → `escalation_explicit`

### 2. Define Expected Behavior

For each test case, specify:

**Tools Called:**

- `[]` - No tools (conversational, off-topic, ambiguous)
- `["knowledge_base_search"]` - KB retrieval only
- `["escalate_to_human"]` - Direct escalation
- `["knowledge_base_search", "escalate_to_human"]` - Search then escalate

**Escalation:**

- `false` - Agent can handle it
- `true` - Must escalate to human

**Content Assertions:**

- `response_should_contain` - Specific facts that MUST appear (e.g., `["50,000"]`, `["30 days"]`)
- `response_should_not_contain` - Things that should NOT appear (e.g., hallucinated facts, off-topic info)

### 3. Write Clear Descriptions

Use descriptive IDs and descriptions:

```json
{
  "id": "kb_billing_07", // Category prefix + sequence number
  "category": "kb_retrieval_billing",
  "description": "Payment plan installment question", // What aspect is being tested
  "message": "Can I pay my bill in installments?"
}
```

### 4. Test Variations

For each category, include:

- **Different phrasings** - Same intent, different wording
- **Short vs. detailed** - "Late fee?" vs. "What is the late payment fee?"
- **Direct vs. indirect** - "Pro Plan cost?" vs. "How much would I pay for Pro?"

### 5. Verify Against Knowledge Base

For KB retrieval categories:

- **Check the KB first** - Ensure the information exists (or doesn't exist for `kb_retrieval_not_in_kb`)
- **Use exact values** - If KB says "50,000", test for "50,000" not "50000"
- **Check categories** - Ensure the query maps to the right KB category (billing, plans, troubleshooting)

### 6. Balance the Test Suite

Maintain approximately equal representation:

- Each category should have **at least 6 test cases**
- Avoid over-representing edge cases
- Focus on **common real-world scenarios**

### 7. Avoid Ambiguous Expectations

❌ **Bad:**

```json
{
  "message": "What's your refund policy?",
  "expected": {
    "tools_called": ["knowledge_base_search"],
    "response_should_contain": ["refund"]
  }
}
```

_Problem: If the KB has no refund policy, this should escalate._

✅ **Good:**

```json
{
  "message": "What's your refund policy?",
  "category": "kb_retrieval_not_in_kb", // If not in KB
  "expected": {
    "tools_called": ["knowledge_base_search", "escalate_to_human"],
    "escalate": true
  }
}
```

### 8. Test One Behavior Per Case

Keep test cases focused:

- ❌ Don't combine multiple questions in one message
- ✅ Test one clear scenario per case
- If testing multi-part questions, create a separate category

---

## Test Case Template

```json
{
  "id": "{category_prefix}_{sequence}",
  "category": "{category_name}",
  "description": "{what_aspect_is_tested}",
  "message": "{user_message}",
  "history": null,
  "expected": {
    "tools_called": [],
    "escalate": false,
    "response_should_contain": [],
    "response_should_not_contain": []
  }
}
```

---

## Evaluation Metrics

Each test case is evaluated on three dimensions:

1. **Tools Match** - Did the agent call the expected tools (order-independent)?
2. **Escalation Match** - Did the agent set the escalation flag correctly?
3. **Content Match** - Does the response contain expected strings and avoid excluded strings?

A test case **passes** only if all three checks pass.

---

## Extending the Test Suite

When adding new categories:

1. **Document the rationale** - Why is this category needed?
2. **Define clear boundaries** - How is it different from existing categories?
3. **Update this document** - Add the new category with examples and guidelines
4. **Update eval script** - Ensure metrics are calculated for the new category
5. **Maintain balance** - Add at least 6 test cases per new category

---
