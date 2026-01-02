# Bid Evaluation Agent

A production-grade bid evaluation system for commercial construction projects. The system ingests bid submissions, enriches contractor data via external search APIs, applies weighted scoring algorithms, and generates AI-powered recommendations.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐  │
│  │   Upload    │    │   Results   │    │        Observability            │  │
│  │   Page      │    │   Page      │    │   (OpenTelemetry Traces)        │  │
│  └──────┬──────┘    └──────▲──────┘    └─────────────▲───────────────────┘  │
└─────────┼──────────────────┼─────────────────────────┼──────────────────────┘
          │                  │                         │
          ▼                  │                         │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CONTROLLER                                       │
│                     (Workflow Orchestration)                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. Parse Bids → 2. Search → 3. Score → 4. Rank → 5. Explain        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────┬─────────────────┬─────────────────┬─────────────────┬───────────────┘
        │                 │                 │                 │
        ▼                 ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   SEARCHER    │ │    SCORER     │ │   EXPLAINER   │ │   OBSERVER    │
│               │ │               │ │               │ │               │
│ ┌───────────┐ │ │ Cost Score    │ │ Groq LLM      │ │ OpenTelemetry │
│ │  Tavily   │ │ │ Timeline      │ │ GPT-OSS-120B  │ │ Traces/Spans  │
│ │   API     │ │ │ Scope Fit     │ │               │ │ Event Logging │
│ └───────────┘ │ │ Risk Score    │ │ Rationale     │ │ RAG Trace     │
│ ┌───────────┐ │ │ Reputation    │ │ Generation    │ │               │
│ │  SerpAPI  │ │ │               │ │               │ │               │
│ └───────────┘ │ │ Flag Engine   │ │               │ │               │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │                 │
        ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SERVICES                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐  │
│  │   Tavily    │    │   SerpAPI   │    │           Groq API              │  │
│  │  (Search)   │    │  (Search)   │    │    (LLM Inference)              │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. Controller (`controller.py`)

The central orchestrator that coordinates the evaluation pipeline.

**Responsibilities:**
- Receives parsed bid data from the UI
- Orchestrates the evaluation workflow
- Manages OpenTelemetry tracing
- Returns structured evaluation results

**Pipeline Stages:**
```
Input Bids → Gather Insights → Score Bids → Rank → Generate Explanation → Output
```

### 2. Searcher (`services/searcher.py`)

External data enrichment service using dual search APIs.

**Features:**
- Parallel execution with `ThreadPoolExecutor`
- Batch processing for multiple companies
- In-memory caching to avoid duplicate API calls
- Signal extraction (US experience, scale alignment, negative news)

**Data Flow:**
```
Company Name → [Tavily + SerpAPI] (parallel) → Merge → Analyze → CompanyInsight
```

### 3. Scorer (`services/scorer.py`)

Deterministic scoring engine with configurable weights.

**Score Dimensions:**

| Dimension | Weight | Calculation |
|-----------|--------|-------------|
| Cost | 20% | Normalized against all bids (lower = better) |
| Timeline | 20% | Duration × confidence level |
| Scope Fit | 25% | Mandatory coverage with subcontracting penalty |
| Risk | 25% | Delivery history + compliance + critical path |
| Reputation | 10% | US experience + scale alignment + news sentiment |

**Flag Engine:**

| Flag | Trigger Condition |
|------|-------------------|
| `f1` | Critical work subcontracted |
| `f2` | Timeline exceeds target (6 months) |
| `f3` | Confidence level < 70% |
| `f4` | Missing mandatory scope items |
| `x1` | No verified US commercial experience |

### 4. Explainer (`services/explainer.py`)

LLM-powered explanation generator using Groq API.

**Model:** `openai/gpt-oss-120b` (120B parameters, reasoning-optimized)

**Outputs:**
- Natural language explanation for stakeholders
- Decision rationale points (up to 4)
- Comparative analysis against runner-up

### 5. Observer (`services/observer.py`)

Observability service with OpenTelemetry integration.

**Capabilities:**
- Distributed tracing with trace/span IDs
- Event logging with timestamps
- Duration tracking for performance analysis
- RAG trace metadata
- Error tracking and analysis

### 6. Memory Store (`services/memory.py`)

SQLite-based persistent storage for evaluation history and caching.

**Tables:**
| Table | Purpose |
|-------|---------|
| `evaluations` | Stores all evaluation results with input/output |
| `company_insights` | Cached external search results |
| `traces` | OpenTelemetry trace data |
| `errors` | Error logs with stack traces |
| `feedback` | User feedback on recommendations |

**Features:**
- Automatic deduplication (same input → cached result)
- Company insight caching (24h TTL)
- Feedback collection for model improvement
- Data export/cleanup utilities

### 7. Historical RAG (`services/historical_rag.py`)

ChromaDB-based vector store for learning from past evaluations.

**Collections:**
| Collection | Purpose |
|------------|---------|
| `evaluations` | Searchable evaluation summaries |
| `company_profiles` | Aggregated company performance |
| `bid_patterns` | Pricing and timeline benchmarks |

**Features:**
- Semantic search across past evaluations
- Company win rate and score tracking
- Risk pattern identification
- Pricing benchmarks from historical data
- Timeline benchmarks for scope comparison

**How it improves accuracy:**
1. **Historical Win Rate** — Companies with proven track records score higher
2. **Pricing Validation** — Flags bids significantly above historical averages
3. **Risk Patterns** — Identifies companies with recurring issues
4. **Timeline Learning** — Compares against similar past projects

---

## Data Models

### Input Schema (`Bid`)

```json
{
  "bid_id": "string",
  "company_name": "string",
  "cost": {
    "total_usd": "number",
    "currency": "USD",
    "cost_breakdown": {
      "labor": "number",
      "materials": "number",
      "contingency": "number"
    }
  },
  "timeline": {
    "estimated_months": "number",
    "confidence_level": "float (0-1)",
    "critical_path_risk": "Low | Medium | High"
  },
  "scope_coverage": {
    "included": ["string"],
    "excluded": ["string"],
    "subcontracted": ["string"]
  },
  "assumptions": ["string"],
  "dependencies": ["string"],
  "prior_similar_projects_count": "number",
  "delivery_history": {
    "on_time_percentage": "float (0-1)",
    "on_budget_percentage": "float (0-1)",
    "known_delays": ["string"]
  },
  "legal_and_compliance": {
    "open_litigation": "boolean",
    "safety_violations_last_5_years": "number"
  },
  "bid_metadata": {
    "submission_channel": "string",
    "submission_timestamp": "ISO 8601",
    "bid_revision": "number"
  }
}
```

### Output Schema (`EvaluationResult`)

```json
{
  "external_company_insights": {
    "Company Name": {
      "sources": ["url"],
      "key_signals": {
        "us_commercial_experience": "boolean | 'Limited'",
        "project_scale_alignment": "High | Medium | Low",
        "recent_negative_news": "boolean"
      },
      "confidence_score": "float (0-1)"
    }
  },
  "retrieved_context_used": "boolean",
  "rag_trace": {
    "vector_store": "string",
    "embedding_model": "string",
    "documents_retrieved_per_company": "number",
    "retrieval_confidence_threshold": "float"
  },
  "bid_scores": [
    {
      "bid_id": "string",
      "company_name": "string",
      "score_breakdown": {
        "cost_score": "float",
        "timeline_score": "float",
        "scope_fit_score": "float",
        "risk_score": "float",
        "external_reputation_score": "float"
      },
      "final_weighted_score": "float",
      "flags": ["string"]
    }
  ],
  "ranked_recommendations": ["bid_id"],
  "final_recommendation": {
    "bid_id": "string",
    "company_name": "string",
    "confidence": "float (0-1)",
    "decision_rationale": ["string"]
  },
  "explanation": "string"
}
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| UI | Streamlit | Web interface |
| LLM | Groq (GPT-OSS-120B) | Explanation generation |
| Search | Tavily + SerpAPI | External data enrichment |
| Observability | OpenTelemetry | Distributed tracing |
| Runtime | Python 3.9+ | Application runtime |

---

## Project Structure

```
bid-eval-agent/
├── app.py                  # Streamlit UI (4 pages)
├── config.py               # Configuration and weights
├── controller.py           # Workflow orchestration
├── models.py               # Data models (dataclasses)
├── services/
│   ├── __init__.py
│   ├── searcher.py         # Tavily + SerpAPI integration
│   ├── scorer.py           # Scoring engine with historical learning
│   ├── explainer.py        # Groq LLM integration
│   ├── observer.py         # OpenTelemetry observability
│   ├── memory.py           # SQLite persistence layer
│   └── historical_rag.py   # ChromaDB historical RAG
├── requirements.txt        # Python dependencies
├── input_bids.json         # Sample input data
├── env.example.txt         # Environment template
├── bid_eval_memory.db      # SQLite database (auto-created)
├── chroma_db/              # ChromaDB vector store (auto-created)
└── README.md               # This file
```

---

## Configuration

### Environment Variables

```bash
# Required
GROQ_API_KEY=gsk_...           # Groq API key for LLM
TAVILY_API_KEY=tvly-...        # Tavily search API key

# Optional
SERPAPI_API_KEY=...            # SerpAPI key (enhances search)
```

### Scoring Weights (`config.py`)

```python
WEIGHTS = {
    "cost": 0.20,       # Lower priority (risk > cost)
    "timeline": 0.20,   # Medium priority
    "scope": 0.25,      # High priority
    "risk": 0.25,       # High priority
    "reputation": 0.10  # External validation
}
```

### Project Parameters

```python
MANDATORY_SCOPE = ["Electrical", "HVAC", "Interior"]
TARGET_MONTHS = 6
TARGET_BUDGET = 1_200_000
CONFIDENCE_THRESHOLD = 0.7
```

---

## Setup & Installation

```bash
# Clone/navigate to project
cd bid-eval-agent

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate      # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy env.example.txt .env    # Windows
cp env.example.txt .env      # Linux/Mac
# Edit .env with your API keys

# Run application
streamlit run app.py
```

---

## Evaluation Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PIPELINE                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐    ┌─────────────┐    ┌─────────┐    ┌──────────────┐  │
│  │  INPUT  │───▶│   SEARCH    │───▶│  SCORE  │───▶│    RANK      │  │
│  │  BIDS   │    │  (Parallel) │    │         │    │              │  │
│  └─────────┘    └─────────────┘    └─────────┘    └──────────────┘  │
│       │                │                │                │          │
│       │                ▼                ▼                ▼          │
│       │         ┌───────────┐    ┌───────────┐    ┌───────────┐    │
│       │         │ Company   │    │ Score     │    │ Ranked    │    │
│       │         │ Insights  │    │ Breakdown │    │ List      │    │
│       │         └───────────┘    └───────────┘    └───────────┘    │
│       │                                                  │          │
│       │                                                  ▼          │
│       │                                          ┌──────────────┐   │
│       │                                          │   EXPLAIN    │   │
│       │                                          │   (LLM)      │   │
│       │                                          └──────────────┘   │
│       │                                                  │          │
│       ▼                                                  ▼          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      OUTPUT RESULT                           │   │
│  │  • External Insights    • Bid Scores    • Recommendation     │   │
│  │  • RAG Trace            • Rankings      • Explanation        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Observability

### OpenTelemetry Tracing

The system generates distributed traces for each evaluation:

```
bid_evaluation (root span)
├── gather_insights
│   └── search_batch
├── score_bids
│   ├── score_b1
│   ├── score_b2
│   └── score_b3
└── generate_explanation
```

### Trace Output

```json
{
  "trace_id": "0af7651916cd43dd8448eb211c80319c",
  "span_id": "b7ad6b7169203331",
  "timestamp": "2026-01-02T16:45:00.000Z",
  "event_type": "search",
  "node": "batch_complete",
  "duration_ms": 1250,
  "data": {
    "companies_searched": 3,
    "total_sources": 12
  }
}
```

### Production Export

To export traces to a backend (Jaeger, Zipkin, etc.), modify `observer.py`:

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
```

---

## API Rate Limits

| Service | Rate Limit | Notes |
|---------|------------|-------|
| Groq | 250K TPM / 1K RPM | Production tier |
| Tavily | Varies by plan | Search API |
| SerpAPI | 100/month (free) | Optional |

---

## Design Decisions

This section documents the rationale behind every significant technical and architectural decision made in this project.

---

### 1. LLM Selection: OpenAI GPT-OSS-120B via Groq

**Decision:** Use `openai/gpt-oss-120b` through Groq's API instead of other models.

**Rationale:**
- **Reasoning capability**: 120B parameters with enhanced reasoning for complex bid analysis
- **Speed**: 500 tokens/second on Groq's LPU infrastructure (faster than cloud GPT-4)
- **Cost efficiency**: $0.15/1M input tokens vs $30/1M for GPT-4
- **Formal output**: Lower temperature (0.1) produces consistent, professional language
- **Context window**: 131K tokens handles large bid documents

**Alternatives Considered:**
| Model | Why Not Chosen |
|-------|----------------|
| `llama-3.3-70b-versatile` | Fewer parameters, less sophisticated reasoning |
| `llama-3.1-8b-instant` | Too small for nuanced bid analysis |
| `openai/gpt-oss-20b` | Faster but lower quality for complex tasks |
| OpenAI GPT-4 | 10x more expensive, slower inference |

---

### 2. Dual Search Strategy: Tavily + SerpAPI

**Decision:** Use two search APIs in parallel rather than a single provider.

**Rationale:**
- **Redundancy**: If one API fails, results still available from the other
- **Coverage**: Different search indexes catch different sources
- **Confidence scoring**: Dual success = higher confidence (0.80 vs 0.65)
- **Source diversity**: Tavily excels at content extraction; SerpAPI at SERP features

**Implementation:**
```python
# Parallel execution - both APIs called simultaneously
tavily_future = executor.submit(self._tavily_search, company)
serp_future = executor.submit(self._serp_search, company)
```

**Trade-off:** Additional API cost (~$0.01/search) justified by improved data quality.

---

### 3. Observability: OpenTelemetry over Custom Logging

**Decision:** Implement OpenTelemetry for observability instead of custom logging.

**Rationale:**
- **Industry standard**: Compatible with Jaeger, Zipkin, Datadog, Honeycomb
- **Distributed tracing**: Trace IDs correlate events across services
- **Future-proof**: Easy to export to any backend without code changes
- **Structured data**: Spans and attributes vs unstructured log strings
- **Performance insights**: Built-in duration tracking per operation

**What we get:**
- Trace ID per evaluation run
- Span hierarchy (parent-child relationships)
- Duration bars for latency analysis
- Event attributes for debugging

**Alternative Considered:** Python `logging` module — rejected because it lacks correlation IDs and requires custom parsing for analysis.

---

### 4. UI Framework: Streamlit

**Decision:** Use Streamlit instead of Flask, FastAPI, or React.

**Rationale:**
- **Rapid prototyping**: Full UI in single Python file
- **No frontend expertise required**: Python-only development
- **Built-in state management**: `st.session_state` for cross-page data
- **Interactive widgets**: File upload, expanders, metrics out of the box
- **Hot reload**: Instant updates during development

**Trade-offs:**
| Streamlit | Traditional Stack |
|-----------|-------------------|
| Limited customization | Full CSS/JS control |
| Single-user focus | Multi-user scalability |
| Slower for heavy loads | Production-grade performance |

**Justification:** POC phase prioritizes development speed over production scaling.

---

### 5. Scoring Weights: Risk-Prioritized

**Decision:** Weight risk (25%) and scope (25%) higher than cost (20%).

**Rationale from Project Context:**
> "While cost matters, delivery risk and operational disruption are higher-priority concerns."

**Weight Distribution:**
```python
WEIGHTS = {
    "cost": 0.20,       # Important but not primary
    "timeline": 0.20,   # Must meet 6-month target
    "scope": 0.25,      # Mandatory items non-negotiable
    "risk": 0.25,       # Delivery history critical
    "reputation": 0.10  # External validation supplementary
}
```

**Design Principle:** The client explicitly stated willingness to accept "slightly higher cost... if risk and disruption are demonstrably lower."

---

### 6. Flag System Design

**Decision:** Use coded flags (f1, f2, f3, f4, x1) instead of free-text warnings.

**Rationale:**
- **Machine-readable**: Enables programmatic filtering and sorting
- **Consistent**: Same flag always means the same thing
- **Compact**: UI-friendly for display in tables
- **Extensible**: Easy to add new flags without breaking existing code

**Flag Definitions:**
| Flag | Trigger | Why It Matters |
|------|---------|----------------|
| `f1` | Subcontracted work | Adds coordination risk, less control |
| `f2` | Timeline > 6 months | Violates project constraint |
| `f3` | Confidence < 70% | Contractor uncertainty = execution risk |
| `f4` | Missing scope | May require change orders |
| `x1` | No US experience | LA permitting/codes unfamiliarity |

**The `x1` flag** was added specifically because the project requires "Proven US commercial renovation experience" — this is a hard requirement, not just nice-to-have.

---

### 7. Parallel Processing Architecture

**Decision:** Use `ThreadPoolExecutor` for concurrent API calls.

**Rationale:**
- **I/O bound operations**: Search APIs are network-limited, not CPU-limited
- **3x speed improvement**: 3 companies searched in ~1s instead of ~3s
- **Simple implementation**: No async/await complexity
- **Built-in timeout handling**: Prevent hanging on slow APIs

**Implementation Pattern:**
```python
# Batch all companies in parallel
futures = {executor.submit(search, company): company for company in companies}
for future in as_completed(futures, timeout=30):
    results[futures[future]] = future.result()
```

**Why not `asyncio`?** ThreadPoolExecutor is simpler for this use case and Streamlit doesn't require async.

---

### 8. No Fallback Design

**Decision:** Remove all fallback mechanisms; fail fast on errors.

**Rationale:**
- **Data integrity**: Partial/mock data could lead to wrong recommendations
- **Clear errors**: Users know immediately when something is wrong
- **Accountability**: No silent failures that go unnoticed
- **Production readiness**: Forces proper API key configuration

**Example:**
```python
# Before (with fallback)
if not GROQ_API_KEY:
    return self._fallback_explanation(...)  # Bad: hides the problem

# After (fail fast)
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is required")  # Good: explicit failure
```

---

### 9. LangSmith-Style Observability UI

**Decision:** Model the Observability tab after LangSmith's trace visualization.

**Rationale:**
- **Industry familiarity**: Developers recognize the pattern from LangSmith/LangFuse
- **Information density**: Shows more data in less space
- **Visual hierarchy**: Duration bars immediately show bottlenecks
- **Debugging efficiency**: Input/output panels per span

**UI Components:**
| Component | Purpose |
|-----------|---------|
| Trace header | Run status, ID, total latency |
| Span rows | Hierarchical view with icons |
| Duration bars | Visual latency comparison |
| I/O panels | Debug data per operation |
| Metrics tab | Aggregate performance stats |

---

### 10. Dataclasses over Pydantic

**Decision:** Use Python `dataclasses` instead of Pydantic models.

**Rationale:**
- **Simplicity**: No additional dependency for POC
- **Performance**: Dataclasses are faster for simple structures
- **Readability**: Clear, minimal syntax
- **Sufficient validation**: Type hints + runtime checks adequate for POC

**When to switch to Pydantic:**
- If adding API endpoints (automatic validation)
- If needing JSON schema generation
- If complex nested validation required

---

### 11. Persistent Memory Layer (SQLite)

**Decision:** Use SQLite for persistent storage instead of in-memory only.

**Rationale:**
- **History tracking**: View and reload past evaluations
- **Caching**: Avoid redundant API calls for same companies
- **Learning**: Collect feedback for future model improvements
- **Zero infrastructure**: SQLite requires no external database
- **Portable**: Single file database, easy backup/transfer

**Schema Design:**
```sql
-- Core tables
evaluations     -- Full evaluation results
company_insights -- Cached search results
traces          -- OpenTelemetry events
errors          -- Error tracking
feedback        -- User feedback
```

**Caching Strategy:**
- Search results cached with 24h TTL
- Same input bids → offer cached result
- Cache hits logged for observability

**Data retained:**
- Evaluation inputs and outputs
- Winner decisions and scores
- All trace data for debugging
- User feedback for learning

---

### 12. Score Normalization Strategy

**Decision:** Normalize cost scores relative to all bids (0.5-1.0 range).

**Rationale:**
- **Relative comparison**: Cheapest bid = 1.0, most expensive = 0.5
- **No zero scores**: Even the worst bid gets 0.5 (not penalized to zero)
- **Fair comparison**: Handles any price range automatically

**Formula:**
```python
normalized = 1 - (cost - min_cost) / (max_cost - min_cost)
score = 0.5 + (normalized * 0.5)  # Scale to 0.5-1.0
```

**Why not 0-1 range?** A bid being most expensive shouldn't mean "zero value" — it still might be the best choice for other reasons.

---

### 13. Temperature and Token Settings

**Decision:** Use temperature=0.1 and max_tokens=800 for LLM calls.

**Rationale:**
- **Low temperature (0.1)**: Produces consistent, formal, deterministic output
- **Higher tokens (800)**: Allows thorough comparative analysis
- **No creativity needed**: Bid evaluation requires accuracy, not creativity

**Comparison:**
| Temperature | Output Style |
|-------------|--------------|
| 0.0 | Completely deterministic (may repeat) |
| 0.1 | Slightly varied but consistent (chosen) |
| 0.7 | Creative, varied responses |
| 1.0 | Highly random, unpredictable |

---

### 14. Service Initialization with Validation

**Decision:** Validate API keys at service initialization, not at call time.

**Rationale:**
- **Fail fast**: Errors surface when app starts, not mid-evaluation
- **Clear diagnostics**: User knows exactly which key is missing
- **No wasted compute**: Don't process bids only to fail at LLM step

**Implementation:**
```python
class Explainer:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=GROQ_API_KEY)
```

---

### 15. IBM Plex Sans Typography

**Decision:** Use IBM Plex Sans as the primary font family.

**Rationale:**
- **Professional appearance**: Designed for enterprise software
- **Excellent readability**: Optimized for screens and data displays
- **Monospace variant**: IBM Plex Mono for trace IDs and code
- **Open source**: Free for commercial use
- **Distinctive**: Avoids generic "AI slop" aesthetics (Inter, Roboto)

---

## Summary of Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| GPT-OSS-120B | Better reasoning | Slightly slower than 70B |
| Dual search | Higher accuracy | 2x API costs |
| OpenTelemetry | Industry standard | Learning curve |
| Streamlit | Fast development | Limited scaling |
| No fallbacks | Data integrity | Less forgiving |
| Parallel processing | 3x faster | Thread complexity |
| In-memory cache | Simple & fast | No persistence |

---

## License

Internal use only. Not for distribution.

---

## Support

For issues or questions, contact the development team.

