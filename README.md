# QNA Chat Agent

## Design Documentation

### System Prompt

The system prompt of the ./agent.py lives in ./prompts.py. We also try to make a simple eval as an initial check of whether the complete end-to-end agentic system is working as expected in ./eval

Here are the rationale of each parts of the prompt:

> You are a helpful assistant that handles inbound customer questions via chat

This tries to make the LLM reasons and converses within the appropriate scope and uses the tone and language typically expected from helpful assistants.

> answering questions about service plans, billing, and basic troubleshooting.

This tries to make the LLM only answer questions that are directly related to the Telecommunication company

> Your primary goal is to answer the user's question accurately.

This tries to ensure the LLM only answer questions factually

Then below this line is all instructions so that the LLM knows what tools are available and when/how to use each one of them

> Use 'knowledge_base_search' ONLY IF the question is about the Telecommunication company, its billing policy, service plans, or troubleshooting guide. DO NOT use this tool for general knowledge, current events, or unrelated topics.

This tries to ensure the knowledge_base_search that triggers a vector similarity search against the documents in ./data is only invoked when there are relevant queries

> Answer directly IF you already know the answer or the question is conversational.

This tries to ensure that conversational chats from the user are responded properly

> Use escalate_to_human when you cannot answer confidently, or when knowledge_base_search returns no relevant results for a company-specific question.

This tries to ensure the LLM knows the exact thing todo after escalation is needed. The ./agent.py file then checks whether this tool is invoked to determine whether to return `escalate: True`

We have determined 8 categories of scenarios which likely will happen in production, detailed in ./docs/chat_scenarios.md. Out of those 8 categories, 2 are determined to be scenarios which the LLM needs to escalate for a human agent. The first one `kb_retrieval_not_in_kb` is a scenario when the LLM determines the query needs search to the knowledge base and the LLM couldn't find any matching documents. The second scenario `escalation_explicit` is scenarios where the customer specifically asks to chat with a human agent.

The fallback and escalation is handled with a combination of: 1. System prompt 2. `escalate_to_human` tool and 3. Code in ./agent.py. The system prompt instructs to when to call `escalate_to_human`. The tool provides LLM a way to end conversation and flag that escalation is needed. The code detects when the `escalate_to_human` is invoked so that appropriate response can be given.

> After using a tool, integrate its findings into a concise and helpful response.

This again tries to make sure the style and tone and language are appropriate for a helpful assistant

### Chucking & Embedding Strategy

To determine what strategy to use we first try to introduce a baseline approach of embedding each document as one chuck. After evaluating the precision and recall of this baseline, we use introduce targeted chunking and document title injection post processing. Our approach managed to delivers a **11% recall improvement** while having **86% precision**, making it a arguable win over the baseline. The details of the strategy can be found in ./docs/chunking-strategy.md. This finding is strengthened by the fact the chunked method results in overall **6% accuracy improvement** from the non chunked chat, which can be seen in ./eval/eval_results_chat.json and eval/eval_results_chat_not_chunked.json

On document injection side, we specifically use LangChain's RecursiveTextSplitter to split the text with 200 `chuck_size` and 0 `chuck_overlap`. We also did some post processing inform of injecting the documents title to the chunk if not exists yet, this is relevant for bullet points that are splited alone at the end which makes the chunk lose important context. After embedding the chucks, the vector values are stored in Supabase's postgres using pgvector extension. The code responsible for document injection are in ./embeddings.py and ./scripts/load_documents.py

Regarding the similarity search, we use a `similarity_threshold` of 0.6 and `max_search_results` of 5 which empirically we found suitable for the current set of document we have in ./data.

Here are the rationale for every hyperparamaters of the whole strategy:

| Parameter              | Value | Rationale                                                                                                                                                     |
| ---------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chunk_size`           | 200   | Matches natural bullet-point boundaries (54-88 chars each). Allows 2-3 bullets per chunk for context while keeping chunks focused.                            |
| `chunk_overlap`        | 0     | No overlap needed — each bullet is self-contained, and title post-processing provides all necessary context.                                                  |
| `max_search_results`   | 5     | With 6 total chunks, returning up to 5 ensures good coverage without returning everything. Balances recall and precision.                                     |
| `similarity_threshold` | 0.6   | Permissive threshold allows semantically related (but not identical) queries to match. Can be tuned up to 0.65-0.70 to reduce cross-category noise if needed. |

### Embedding model choice

We use Google Gemini's `gemini-embedding-001` as the embedding model because this model is latest the non preview stable model which focuses on text embeddings. The `embedding_dimension` of `3072` is chosen because this is the recommended default from Google. Reference: <https://ai.google.dev/gemini-api/docs/embeddings#control-embedding-size>

As why Google Gemini's is chosen specifically, we chose the provider because they have very generous free tier which can help development efforts more efficient. To facilitate changing providers, we have used the Facade design pattern in ./embeddings.py, ./ai.py and ./agent.py so that making provider switch is easier

### Featured limitation and mitigation steps

With the assumption the knowledge base provided in ./data is complete, the next pressing limitation is that actual production queries are not clearly known. We designed the evaluation test cases of the agent in ./eval based queries that are predicted to be used by the consumer. While the predictions are reasonable, actual production queries might differ. In the future, there needs to be a feedback loop to monitor production queries and eventually update the evaluation system so that it more closely match actual customer queries.

Besides this limitation there are operational limitations that we couldn't complete for first version of this project, which are documented in ./docs/tech_debt.md

---

## Quick Start

**Prerequisites:** Python 3.13+, [`uv`](https://docs.astral.sh/uv/), a Google Gemini API key, and a Supabase project with the pgvector extension enabled.

**1. Install dependencies**

```bash
uv sync
```

**2. Configure environment**

```bash
cp .env.example .env
# Fill in GOOGLE_API_KEY, VECTOR_DATABASE_URL, and VECTOR_DATABASE_API_KEY
```

**3. Set up the database**

Run the SQL in `./sql/vector_database.sql` against your Supabase project to create the tables and RPC functions.

**4. Load documents into the knowledge base**

```bash
uv run python scripts/load_documents.py
```

**5. Start the server**

```bash
uv run fastapi dev main.py
```

The API is now available at `http://localhost:8000`. Send a chat message:

uv run fastapi dev main.py

````

The API is now available at `http://localhost:8000`. Send a chat message:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What service plans do you offer?"}'
`  -d '{"message": "What service plans do you offer?"}'
```
````
