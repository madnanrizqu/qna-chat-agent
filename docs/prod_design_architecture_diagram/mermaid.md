```mermaid
architecture-beta

    group users[Users]
    group voice[Voice]
    group core[Core]
    group ai[AI]
    group data[Data]
    group admin[Admin]

    service customer(internet)[Customer] in users
    service human_agent(internet)[Human Agent] in users

    service sip(server)[SIP] in voice
    service speech_service(server)[Speech Service] in voice

    service fe(server)[FE Web App] in core
    service chat_service(server)[Chat Service] in core
    service escalation_router(server)[Escalation Router] in core

    service session_store(database)[Session Store] in data
    service embeddings_store(database)[Embeddings Store] in data
    service monitoring(server)[Monitoring] in data
    service logs_store(database)[Logs Store] in data

    service ai_service(cloud)[AI Service] in ai
    service embedding_model(server)[Embedding Model] in ai

    service admin_fullstack(server)[Admin Fullstack] in admin
    service internal_team(internet)[Internal Team] in admin

    customer:R <--> L:fe
    customer:T <--> B:sip
    human_agent:R <--> T:escalation_router
    fe:R <--> L:chat_service
    sip:R <--> L:speech_service
    speech_service:B <--> T:chat_service
    chat_service:R <--> L:ai_service
    chat_service:B <--> T:session_store
    chat_service:B <--> T:escalation_router
    escalation_router:B <--> R:session_store
    ai_service:T <--> B:embedding_model
    embedding_model:B <--> T:embeddings_store
    embeddings_store:L <--> T:monitoring
    monitoring:L --> L: logs_store
    chat_service:B --> T:monitoring
    admin_fullstack:L <--> R:monitoring
    admin_fullstack:T <--> B:embeddings_store
    internal_team:L <--> R:admin_fullstack
```
