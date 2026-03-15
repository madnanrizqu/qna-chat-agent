# Technical Debt & Known Issues

Issues identified during codebase review. Issue #1 (live credentials in `.env`) is omitted — treat it as an immediate operational action, not tracked here.

---

## #1 — No Authentication or Authorization (HIGH)

**Location:** `main.py`

All endpoints (`GET /`, `POST /chat`) are fully open. Any caller with network access can query the LLM agent with no identity verification. There is no API key check, JWT, OAuth, rate limiting, or CORS policy.

**Remediation:** Add an API key middleware or integrate an auth provider (e.g., Supabase Auth, Auth0). At minimum, require a static bearer token for `POST /chat` before exposing to the internet.

---

## #2 — Internal Errors Leaked to Callers (HIGH)

**Location:** `main.py` — `/chat` handler

The bare `except Exception as e` block returns `str(e)` directly as the HTTP 500 `detail`. This can expose internal implementation details — Python tracebacks, internal service URLs, database error messages — to API consumers.

**Remediation:** Return a generic error message to the caller (e.g., `"An internal error occurred."`), and log the full exception server-side with the request ID for tracing.

---

## #3 — No Input Size Limits (MEDIUM)

**Location:** `models.py` — `ChatRequest`, `Message`

`message` has `min_length=1` but no `max_length`. The `history` list is unbounded. A single request can carry an arbitrarily large payload, consuming a very large LLM context window and driving up Google API costs proportionally.

**Remediation:** Add `max_length` to `ChatRequest.message` (e.g., 4000 chars) and cap `history` list length (e.g., 20 messages) via a Pydantic validator.

---

## #4 — No Rate Limiting (MEDIUM)

**Location:** `main.py`

There is no throttling at any layer. A single client can send unlimited parallel requests, linearly scaling Google API and Supabase costs with no circuit breaker.

**Remediation:** Add request-rate middleware (e.g., `slowapi` for FastAPI) keyed by IP or API key. Define per-client and global request-per-minute limits.

---

## #5 — Prompt Injection Not Mitigated (MEDIUM)

**Location:** `agent.py`, `prompts.py`

User-supplied `message` and `history` content is passed directly to the LLM with no sanitization. A crafted input could subvert the system prompt, override tool-calling behavior, or manipulate the escalation flag.

**Remediation:** Add an input sanitization step before passing user content to the agent. Consider a separate LLM-based moderation call (e.g., Google's safety filters or a guard model) to detect and reject adversarial inputs.

---

## #6 — `escalate_to_human` Is a Stub (LOW)

**Location:** `tools.py` — `escalate_to_human`

The tool does nothing except return a string. No ticket is created, no webhook is fired, no queue is written. The escalation signal only exists as a flag in the API response — if the caller ignores it, nothing happens.

**Remediation:** Implement a real escalation side-effect: write to a queue (e.g., SQS, Supabase table), call a CRM webhook, or emit a structured event. The current behavior silently drops escalations if the client doesn't act on the flag.

---

## #7 — No Retry or Fallback on External API Failures (LOW)

**Location:** `ai.py`, `embeddings.py`, `agent.py`

Calls to the Gemini API and Supabase have zero retry logic. A transient network blip or upstream rate limit propagates immediately as an HTTP 500 to the caller.

**Remediation:** Wrap external calls with exponential-backoff retry logic (e.g., `tenacity`). Define a fallback response for embedding failures (e.g., degrade gracefully without KB search rather than failing the entire request).

---

## #8 — Non-Standard LangChain Agent API (LOW)

**Location:** `agent.py` — `LangChainGeminiAgentRunner`

The agent is constructed using internal or non-public LangChain APIs. The standard public surface is `create_react_agent`. Minor LangChain version bumps can silently break construction or tool-binding behavior.

**Remediation:** Migrate to `create_react_agent` from `langchain.agents` and pin LangChain to an exact version in `pyproject.toml` until the migration is verified.

---

## #9 — Service-Role DB Key Used for All Operations (LOW)

**Location:** `embeddings.py` — `SupabaseVectorStore.search_similar`

The Supabase service-role key (which bypasses all Row Level Security) is used for every similarity search at runtime. This is excessive privilege — read-only search queries only need the anon key.

**Remediation:** Use the anon (public) key for `search_similar` calls. Reserve the service-role key exclusively for the ingestion script (`scripts/load_documents.py`). Enable and enforce RLS on `documents` and `document_chunks` tables.
