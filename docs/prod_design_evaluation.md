# Telco Customer Service AI Agent — Design and Evaluation

## Design Document

![Architecture Diagram](prod_design_architecture_diagram.png)

### Component Responsibilities

**FE (Web/App)** — The front-end layer serving both customers and human agents. It handles text-based chat over HTTP and renders voice interactions routed through Telephony/SIP. It also delivers escalation notices to human agents when the AI cannot resolve a query.

**Telephony/SIP** — Manages inbound voice calls and establishes WebSocket connections to the Chat Service for real-time, bidirectional audio streaming.

**Chat Service** — The central orchestrator. It receives all inbound messages (text and voice), coordinates the RAG pipeline (embedding queries, performing semantic search, calling the LLM), persists conversation state, stores logs, and decides whether to respond directly or escalate to a human agent.

**AI Service** — The LLM inference layer. It receives enriched prompts (user query + retrieved context + session history) from the Chat Service and returns generated responses.

**Embedding Model** — Converts text into vector representations. Used in two contexts: transforming user queries into embeddings for semantic search, and processing new documents during knowledge base ingestion.

**Embeddings Store** — A vector database holding the indexed knowledge base. Supports similarity search at query time and accepts new vectors when documents are ingested.

**Chat Persistence DB** — Stores conversation history and application state. The Chat Service both writes to and reads from this store, enabling session continuity.

**Object Storage** — Holds raw conversation logs. The Admin Fullstack retrieves and analyzes these for quality monitoring and operational insight.

**Admin Fullstack** — The internal operations interface. It allows the internal team to update the knowledge base (triggering document ingestion), review conversation logs, and retrieve application data for analysis.

### Voice vs. Chat Input

Chat and voice follow different entry paths that converge at the Chat Service. A text chat request arrives from the FE via `/chat (HTTP)` as a standard request-response cycle. Voice, by contrast, flows from the FE through Telephony/SIP and into the Chat Service over `/chat-ws (WebSocket)`, which maintains a persistent connection for real-time audio streaming. Once messages reach the Chat Service, the downstream processing: RAG retrieval, LLM inference, session persistence; is identical regardless of input channel.

### Knowledge Base Update Flow

When the internal team adds a new document through the Admin Fullstack, the system follows a three-step ingestion pipeline. First, the Admin Fullstack sends the document to the Embedding Model ("Ingest new Document"). The Embedding Model chunks and vectorizes the content. Finally, the resulting embeddings are written to the Embeddings Store ("Store/Retrieve"). No redeployment is required, the next semantic search query from the Chat Service will immediately pick up the new content.

### Conversation Memory

Session context lives in the Chat Persistence DB. When a user sends a message, the Chat Service reads prior turns from the database ("Retrieve/Stores App Data"), appends the current message, and includes the full history in the prompt sent to the AI Service. After the AI responds, the new exchange is written back to the same store. This gives the agent within-session memory without requiring the LLM itself to maintain state.

### Scalability Concern: Chat Service as a Single Point of Convergence

Every request: text, voice, RAG queries, LLM calls, persistence writes, and log storage; routes through the Chat Service. Under load, this becomes a bottleneck and a single point of failure. If the Chat Service goes down, the entire system is unavailable.

To address this, the Chat Service should be deployed as a horizontally scalable, stateless service behind a load balancer. Because session state already lives externally in the Chat Persistence DB, any instance can handle any request. Adding an autoscaling policy based on concurrent WebSocket connections and HTTP request latency would allow the system to absorb traffic spikes; particularly important for a telco, where call volume can surge unpredictably during outages or billing cycles.

## Evaluation & Observability

### Evaluation Metrics

To evaluate the Customer Service AI Agent holistically, we should follow these following metrics:

1. Retrieval recall and precision: Since RAG is using, retrieving representative chunks are paramount. There are two sides of how good the retrieval could be which is precision and recall. Recall means for a given query does the LLM retrieve at least one relevant chunks, whereas precision means out of the retrieved chunks how many are actually relevant compared to total chunks

2. Tone and style language accuracy: As the customer service agent serves as a primary touchpoint representing the telecommunications company, it is important to measure how well the LLM communicates in the style and tone defined by the brand. Agents are often depicted with a specific persona, such as Telkomsel's AVA, Veronika, Savia, or Apple's Siri. Evaluating tone and language ensures consistency with the company's desired image and customer experience.

3. Generation accuracy: This is the end-to-end evaluation of the whole agentic RAG flow. Accuracy here means how much of the queries are classified as accurate compared to the ground truth labels.

### Test cases and test methodology

Here are the test cases that we should create these test cases:

- Retrieval test case: This dataset should consist with the predicted queries that the customer would happen do. Furthermore the queries should be categorized into meaningful parts. An initial form of this test case can be seen in ./eval/retrieval_queries.json. The queries are categorized into scenarios of which the semantic search could result to:

  a. `billing_policy`: Queries related to billing policy
  b. `service_plans` : Queries related to service plans
  c. `troubleshooting_guide`: Queries related to troubleshooting guide
  d. `not_matching`: Queries that does not match any of above categories

The test script then would evaluate the recall and precision of the semantic search using the queries with the vector database. A semantic search has passing recall when one of the retrieved chunk's category matches with the query's category. While the search's precision is the amount of correctly retrieved chunk compared to the total amount of retrieved chunks.

Since the test cases has the above categories, we can see the recall and precision of the semantic search on each categories. while the aggregated scores are also available

- Tone and style language accuracy; This dataset should consist with records that collects sample queries, category of the queries and reference answer which describe the tone and language the Telecommunication company would approve various scenarios. There would be effectively 3 columns:

a. `query`: The question/query that the LLM would be asked against
b. `category`: The scenario of query that is prevalently asked by customers
c. `reference_answer`: The source of truth answer handcrafted by the company

The test script then would call the agent with each record and get the `generated_answer`. Afterwords, another LLM would act as a judge to match the `generated_answer` with the `reference_answer`, an answer is said to be have appropriate tone and style if the judge deems both answers are in the same tone and style.

Since the test cases has the above categories, we can see the accuracy of the agent on each categories. while the aggregated score is also available

- Chat generation test case: This dataset should consist of conversations that shows how the LLM agent is expected to answer on various scenarios. An initial form of this test case can be seen in ./eval/chat_test_cases.json. The queries are categorized into scenarios of conversation end results:

  a. `direct_conversational`: To verify the agent handles routine social interactions without unnecessary tool calls.
  b. `off_topic` : To verify the agent correctly identifies non-company questions and avoids searching the KB.
  c. `kb_retrieval_billing`: To verify the agent correctly searches for and returns billing policy information.
  d. `kb_retrieval_plans`: To verify the agent correctly retrieves service plan information.
  e. `kb_retrieval_troubleshooting`: To verify the agent retrieves and provides basic technical support.
  f. `kb_retrieval_not_in_kb`: To verify the agent searches first, then escalates when information is missing.
  g. `ambiguous_vague`: To verify the agent handles unclear requests by asking for clarification.
  h. `escalation_explicit`: To verify the agent respects direct requests for human agents.

Furthermore each test cases details the expected outcome in 4 dimensions: 1. `tools_called`; 2. `escalate`; 3. `response_should_contain`; 4. `response_should_not_contain`

The test script then would call the agent with each record and get the message response. Furthermore the list of tool calls and escalation flag is retrieved. The agent is correct if all provided expectation dimension matches the retrieved message, tool calls and escalation flag.

Since the test cases has the above categories, we can see the accuracy of the agent on each categories. while the aggregated score is also available

### Minimum passing threshold

The exact threshold is for sure dependent on the priorities of the telecommunication company, with that being said the bare minimum for a customer service agent that is accompanied by a human agent would look like this:

1. Recall & Precision: Recall should be above 80% for all categories while precision should be above 90% for all categories. Precision should be higher because the ability to retrieved correct information over false information is more important than the ability to actually retrieve at least one correct information. In cases where agent fails in recall, the customer service agent has a fallback of escalating to a human agent. Whilst if the agent fails in precision theres no reverting the damage of giving false information

2. Tone and style of language accuracy: Above 70% for all categories should be good. The matchness of tone and style the LLM generates with the reference answer for most cases won't be detectable by many customers. Furthermore, on initial release the functional correctness would be more important over non functional metrics like this.

3. Generation accuracy: The agent should be produce an correct answer above 80% for all categories with the exception for `kb_retrieval_billing`, `kb_retrieval_plans`, `kb_retrieval_troubleshooting`, `kb_retrieval_not_in_kb` and `escalation_explicit`. These categories should be high in accuracy since wrong answer would lead to either: a. false information given or b. wrongly escalated to human agent
