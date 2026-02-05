# Architecture Diagram

## System Architecture Diagram (Mermaid)

Copy and paste this into any Mermaid-compatible viewer (GitHub, VS Code with Mermaid extension, mermaid.live)

```mermaid
graph TB
    subgraph "Trigger Layer"
        A[Azure Databricks Job Failed] --> B[Event Grid Trigger]
        B --> C[Azure Function: Log Retrieval Agent]
    end
    
    subgraph "Data Ingestion & Vectorization"
        C --> D[Retrieve Job Logs + Screenshots]
        D --> E[Azure AI Search: Vector Indexing]
        E --> F[RAG Knowledge Base: Historical Fixes]
    end
    
    subgraph "Azure AI Foundry Agent Service"
        F --> G[Agent Orchestrator]
        G --> H[Context Retrieval Agent]
        H --> I{Log Type?}
        I -->|Text Only| J[GPT-4o: Error Analysis]
        I -->|Contains Visuals| K[IDP Visual Underwriting Agent]
        K --> L[Azure Document Intelligence]
        L --> J
    end
    
    subgraph "Evaluation & Classification"
        J --> M[LangSmith Tracing & Evaluation]
        M --> N[RAG Triad Evaluation]
        N --> O[TensorFlow Reliability Classifier]
        O --> P[Error Category + Fix Suggestion]
    end
    
    subgraph "Human-in-the-Loop"
        P --> Q[Azure Logic App: Email Notification]
        Q --> R[Engineering Team Review]
        R --> S{Approval Signal}
        S -->|Approve| T[Webhook: Restart Job]
        S -->|Reject| U[Webhook: Terminate Job]
    end
    
    subgraph "Action Execution"
        T --> V[Databricks API: Restart Job]
        U --> W[Databricks API: Cancel Job]
        V --> X[Azure Monitor: Log Action]
        W --> X
    end
    
    subgraph "Observability Layer"
        X --> Y[Application Insights: E2E Trace]
        Y --> Z[Databricks Lakeflow: Data Lineage]
        Z --> AA[Dashboard: Pipeline Health Metrics]
        M -.Feedback Loop.-> G
    end
    
    style G fill:#0078d4,color:#fff
    style J fill:#10a37f,color:#fff
    style L fill:#ff6b6b,color:#fff
    style O fill:#4ecdc4,color:#000
```

## Component Interaction Flow

```mermaid
sequenceDiagram
    participant DB as Databricks
    participant EG as Event Grid
    participant WF as Workflow (Container App)
    participant AI as Azure AI Foundry
    participant RAG as Azure AI Search
    participant IDP as Document Intelligence
    participant TF as TensorFlow Model
    participant Email as Communication Services
    participant Human as Engineering Team
    participant Webhook as Approval Function
    
    DB->>EG: Job Failed Event
    EG->>WF: Trigger Workflow
    WF->>DB: Retrieve Logs & Screenshots
    DB-->>WF: Return Error Data
    
    WF->>RAG: Vectorize & Search Similar Errors
    RAG-->>WF: Historical Fixes
    
    alt Has Screenshots
        WF->>IDP: Analyze Visual Telemetry
        IDP-->>WF: Extracted Text & Insights
    end
    
    WF->>AI: Create Agent Thread with Context
    AI->>AI: Analyze with GPT-4o
    AI-->>WF: Error Analysis & Suggested Fix
    
    WF->>TF: Predict Reliability
    TF-->>WF: Risk Classification
    
    WF->>Email: Send Approval Request
    Email->>Human: Email with Analysis
    
    Human->>Webhook: Click Approve/Reject
    Webhook->>Webhook: Update Approval Status
    
    WF->>Webhook: Poll for Approval
    Webhook-->>WF: Return Decision
    
    alt Approved
        WF->>DB: Restart Job
    else Rejected
        WF->>DB: Terminate Job
    end
    
    WF->>WF: Log to Azure Monitor
```

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input
        A[Error Logs]
        B[Screenshots]
        C[Job Metadata]
    end
    
    subgraph Processing
        D[Text Vectorization]
        E[Visual OCR]
        F[Feature Engineering]
    end
    
    subgraph AI Layer
        G[Vector Search]
        H[GPT-4o Analysis]
        I[TensorFlow Classifier]
    end
    
    subgraph Output
        J[Error Category]
        K[Root Cause]
        L[Suggested Fix]
        M[Reliability Score]
    end
    
    A --> D
    B --> E
    C --> F
    
    D --> G
    E --> H
    F --> I
    
    G --> H
    H --> J
    H --> K
    H --> L
    I --> M
    
    style G fill:#e1f5ff
    style H fill:#ffe1f5
    style I fill:#f5ffe1
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Azure Cloud"
        subgraph "Compute"
            CA[Container Apps<br/>Main Workflow]
            FA[Function App<br/>Webhook]
        end
        
        subgraph "AI Services"
            AOAI[Azure OpenAI<br/>GPT-4o]
            AIS[Azure AI Search<br/>Vector Store]
            DI[Document Intelligence<br/>IDP]
            AIF[AI Foundry<br/>Agent Service]
        end
        
        subgraph "Data & Storage"
            TS[Table Storage<br/>Approval State]
            ACR[Container Registry<br/>Images]
        end
        
        subgraph "Communication"
            CS[Communication Services<br/>Email]
            EG[Event Grid<br/>Events]
        end
        
        subgraph "Observability"
            AI[Application Insights<br/>Tracing]
            LS[LangSmith<br/>LLM Eval]
        end
    end
    
    subgraph "External"
        DB[(Databricks<br/>Data Platform)]
        User[Engineering Team]
    end
    
    DB -->|Job Failed| EG
    EG --> CA
    CA --> AOAI
    CA --> AIS
    CA --> DI
    CA --> AIF
    CA --> CS
    CA --> AI
    CA --> LS
    CA --> DB
    CS --> User
    User --> FA
    FA --> TS
    CA --> TS
    
    style CA fill:#0078d4,color:#fff
    style FA fill:#0078d4,color:#fff
    style AOAI fill:#10a37f,color:#fff
    style AIF fill:#0078d4,color:#fff
```

## RAG Triad Evaluation Flow

```mermaid
graph TD
    A[Query: Error Description] --> B[Retrieve Documents]
    B --> C{Retrieved Context}
    
    C --> D[Faithfulness Check]
    C --> E[Context Precision]
    
    D --> F[Generate Answer]
    F --> G[Answer Relevance]
    
    D --> H{Score > 0.75?}
    E --> H
    G --> H
    
    H -->|Yes| I[High Quality Response]
    H -->|No| J[Flag for Review]
    
    I --> K[Send to Human Approval]
    J --> L[Require Manual Review]
    
    style D fill:#ffd700
    style E fill:#ffd700
    style G fill:#ffd700
    style H fill:#90ee90
```

---

## View These Diagrams

1. **GitHub**: Diagrams render automatically in README files
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Copy to https://mermaid.live
4. **Documentation**: Use in Confluence, Notion (with Mermaid support)
