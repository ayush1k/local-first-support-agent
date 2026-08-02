# Local-First Support Agent Network - Architecture Specification

## 1. Overview & Objectives
The **Local-First Support Agent Network** is an autonomous, privacy-focused support intelligence system designed for OrbitDesk. Operating entirely locally using Hugging Face models and open-source components, the system retrieves documentation, analyzes past resolved cases, resolves conflicts (ensuring current documentation supersedes historical resolved cases), and generates strictly validated answers using Pydantic and LangGraph.

---

## 2. Component Mapping

```
+-----------------------------------------------------------------------------------+
|                                  LangGraph Orchestrator                            |
|                                                                                   |
|  +--------------------+     +------------------------+     +-------------------+  |
|  |  Query Router /    | --> | Knowledge Retrieval &  | --> |  Resolved Case    |  |
|  |  Sanitizer Node    |     | Vector Search Node     |     |  Evaluator Node   |  |
|  +--------------------+     +------------------------+     +-------------------+  |
|                                                                      |            |
|                                                                      v            |
|  +--------------------+     +------------------------+     +-------------------+  |
|  | Output Validator & | <-- | Local HF Generation    | <-- | Conflict & Doc    |  |
|  | Guardrail Node     |     | (LLM Inference) Node   |     | Priority Resolver |  |
|  +--------------------+     +------------------------+     +-------------------+  |
+-----------------------------------------------------------------------------------+
```

### Core Components
1. **LangGraph Workflow Orchestrator**: Manages state transitions, branch logic, and node execution flow.
2. **Query Router & Sanitizer Node**: Normalizes user prompts, classifies intent (e.g., product query vs. account action), and strips sensitive data.
3. **Knowledge Base Retrieval Engine (ChromaDB / FAISS)**: Indexes `/knowledge_base/` markdown files using local Hugging Face embeddings (e.g., `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`).
4. **Resolved Cases Evaluator**: Searches `resolved_cases.json` for similar historical issues while flagging `superseded` entries to ensure they do not override current docs.
5. **Conflict & Priority Resolver**: Applies business logic rules:
   - Official product documentation takes highest precedence.
   - Active resolved cases provide supplementary guidance.
   - `superseded` resolved cases are excluded or explicitly filtered out.
   - Non-supported operations (e.g., refund processing, account modifications) trigger refusal fallback responses.
6. **Local HF Generation Engine**: Uses local Hugging Face model runners (e.g., `transformers` / `vLLM` / `llama-cpp-python` with quantized GGUF or HF Transformers) to construct answers.
7. **Pydantic Validation & Output Guard**: Validates the output against the defined `output_schema.json` Pydantic model before returning to the user.

---

## 3. Data Flow Architecture

1. **User Query Input**: Natural language question enters the pipeline.
2. **State Initialization**: LangGraph initializes `AgentState` with user query, message history, and default execution metadata.
3. **Intent Classification & Routing**:
   - Checks if query asks for non-executable operations (account edits, refunds).
   - Routes to query expansion / embedding generation.
4. **Parallel Retrieval**:
   - Query Vector Store for relevant `knowledge_base` passages.
   - Search `resolved_cases.json` for matching case histories.
5. **Context Aggregation & Conflict Filtering**:
   - Filters out cases marked `"superseded": true`.
   - Ranks passages ensuring doc source precedence.
6. **Prompt Assembly & Local Inference**:
   - Constructs grounded prompt containing retrieved docs, active cases, and strict constraints.
   - Passes prompt to local Hugging Face model.
7. **Pydantic State Validation**:
   - Formats and parses response into `SupportResponse` Pydantic model.
   - Validates required fields: `answer`, `sources`, `confidence`, `suggested_followups`.
8. **Final Output Delivery**: Validated response payload delivered to user/client.

---

## 4. Pydantic State & Output Models

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class DocumentSource(BaseModel):
    source_id: str = Field(description="Document ID, file path, or resolved case ID")
    source_type: str = Field(description="'documentation' or 'resolved_case'")
    relevance_score: float

class SupportResponse(BaseModel):
    query: str
    answer: str = Field(description="Direct, grounded answer based strictly on knowledge base")
    sources: List[DocumentSource] = Field(description="List of cited documents and cases")
    requires_human_escalation: bool = False
    refusal_reason: Optional[str] = None

class AgentState(BaseModel):
    user_query: str
    query_intent: Optional[str] = None
    retrieved_docs: List[dict] = []
    retrieved_cases: List[dict] = []
    conflict_notes: List[str] = []
    generation: Optional[str] = None
    final_output: Optional[SupportResponse] = None
    error: Optional[str] = None
```

---

## 5. File Structure

```
local-first-support-agent/
│
├── architecture.md             # Complete architecture specification & data flow
├── memory.md                   # Project progress & phase tracking memory
├── README.md                   # Project description and package contents
├── requirements.txt            # Dependencies (langgraph, transformers, pydantic, chromadb, torch)
├── output_schema.json          # Starter JSON Schema for responses
├── sample_questions.json       # Benchmark evaluation questions
├── resolved_cases.json         # Historical resolved support cases dataset
│
├── knowledge_base/             # Product documentation files (Primary Source of Truth)
│   └── ... (markdown files)
│
├── src/                        # Core Application Source Code
│   ├── __init__.py
│   ├── main.py                 # CLI / Entry point for processing support queries
│   ├── config.py               # Global settings, paths, and model configurations
│   │
│   ├── state/                  # LangGraph state & Pydantic schemas
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models for agent state and outputs
│   │
│   ├── graph/                  # LangGraph workflow definition
│   │   ├── __init__.py
│   │   ├── builder.py          # StateGraph compilation & edge routing
│   │   └── nodes/              # Individual execution nodes
│   │       ├── __init__.py
│   │       ├── router.py       # Intent classification & query sanitization node
│   │       ├── retrieval.py    # Vector retrieval & document search node
│   │       ├── conflict.py     # Priority resolution & conflict detection node
│   │       ├── generation.py   # Local HF LLM response generation node
│   │       └── validation.py   # Output guardrail & Pydantic validation node
│   │
│   ├── retrieval/              # Indexing and Vector Database Management
│   │   ├── __init__.py
│   │   ├── embeddings.py       # Local Hugging Face embeddings wrapper
│   │   └── vector_store.py     # ChromaDB / FAISS indexer for docs & cases
│   │
│   └── models/                 # Local LLM Inference Pipeline
│       ├── __init__.py
│       └── hf_runner.py        # Local HF model loader (Transformers/vLLM/GGUF)
│
└── tests/                      # Suite of tests
    ├── test_retrieval.py       # Test vector store indexing and query retrieval
    ├── test_conflict.py        # Test doc precedence over superseded resolved cases
    └── test_pipeline.py       # End-to-end evaluation using sample_questions.json
```
