# Bid Evaluation Agent

Production-grade bid evaluation system for commercial construction projects using LangGraph orchestration, multi-criteria scoring, and LLM-powered analysis.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT UI                                  │
│     Upload  │  Results  │  History  │  Observability                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH WORKFLOW                               │
│  parse_bids → check_cache → gather_insights → gather_historical →      │
│  score_bids → rank_bids → generate_explanation → build_result →        │
│  persist_result → END                                                   │
└────────┬────────────┬────────────┬────────────┬─────────────────────────┘
         │            │            │            │
         ▼            ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
    │Searcher │  │ Scorer  │  │Explainer│  │Observer │
    │Tavily   │  │Weighted │  │Groq LLM │  │OTel     │
    │SerpAPI  │  │Scoring  │  │GPT-120B │  │Tracing  │
    └─────────┘  └─────────┘  └─────────┘  └─────────┘
         │            │            │            │
         ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│   SQLite (Memory)  │  ChromaDB (RAG)  │  External APIs                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI | Streamlit | Web interface |
| Orchestration | LangGraph | Workflow state machine |
| LLM | Groq GPT-OSS-120B | Explanation generation |
| Search | Tavily + SerpAPI | External data enrichment |
| Vector DB | ChromaDB | Historical RAG |
| Storage | SQLite | Persistent memory |
| Observability | OpenTelemetry | Distributed tracing |

## Project Structure

```
bid-eval-agent/
├── app.py              # Streamlit UI
├── controller.py       # LangGraph invoker
├── models.py           # Data models
├── config.py           # Configuration
├── services/
│   ├── graph.py        # LangGraph workflow
│   ├── searcher.py     # Tavily + SerpAPI
│   ├── scorer.py       # Scoring engine
│   ├── explainer.py    # Groq LLM
│   ├── observer.py     # OpenTelemetry
│   ├── memory.py       # SQLite persistence
│   └── historical_rag.py # ChromaDB RAG
├── requirements.txt
├── input_bids.json     # Sample data
└── env.example.txt     # Environment template
```

## Installation

```bash
cd bid-eval-agent
python -m venv venv
.\venv\Scripts\Activate  # Windows
pip install -r requirements.txt
copy env.example.txt .env
# Edit .env with API keys
streamlit run app.py
```

**Requirements:** Python 3.10-3.13

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `SERPAPI_API_KEY` | No | SerpAPI key (enhances search) |

### Scoring Weights

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Cost | 20% | Normalized bid cost |
| Timeline | 20% | Duration × confidence |
| Scope | 25% | Mandatory coverage |
| Risk | 25% | Delivery history + compliance |
| Reputation | 10% | External validation |

### Flags

| Flag | Condition |
|------|-----------|
| `f1` | Critical work subcontracted |
| `f2` | Timeline > 6 months |
| `f3` | Confidence < 70% |
| `f4` | Missing mandatory scope |
| `x1` | No US experience |
| `h1` | Historical risk patterns |
| `h2` | Poor track record |

## Input Schema

**Required fields:**
```json
{
  "bid_id": "string",
  "company_name": "string",
  "cost": { "total_usd": number },
  "timeline": { "estimated_months": number, "confidence_level": float },
  "scope_coverage": { "included": ["string"] }
}
```

**Optional fields:** `cost_breakdown`, `critical_path_risk`, `excluded`, `subcontracted`, `assumptions`, `dependencies`, `prior_similar_projects_count`, `delivery_history`, `legal_and_compliance`, `bid_metadata`

## Output Schema

```json
{
  "external_company_insights": { "Company": { "sources": [], "key_signals": {}, "confidence_score": float } },
  "bid_scores": [{ "bid_id": "", "company_name": "", "score_breakdown": {}, "final_weighted_score": float, "flags": [] }],
  "final_recommendation": { "bid_id": "", "company_name": "", "confidence": float, "decision_rationale": [] },
  "explanation": "string"
}
```

## LangGraph Workflow

| Node | Function |
|------|----------|
| `parse_bids` | Parse JSON → Bid objects |
| `check_cache` | Return cached result if available |
| `gather_insights` | Tavily + SerpAPI enrichment |
| `gather_historical` | Historical RAG lookup |
| `score_bids` | Weighted multi-criteria scoring |
| `rank_bids` | Sort by score, select winner |
| `generate_explanation` | Groq LLM explanation |
| `build_result` | Assemble output |
| `persist_result` | Save to SQLite + ChromaDB |

## API Rate Limits

| Service | Limit |
|---------|-------|
| Groq | 250K TPM / 1K RPM |
| Tavily | Plan dependent |
| SerpAPI | 100/month (free) |

---

Internal use only.
