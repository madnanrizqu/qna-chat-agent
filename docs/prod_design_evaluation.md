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
