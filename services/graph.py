# services/graph.py
"""LangGraph-based workflow for bid evaluation."""

from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END

from models import (
    Bid, BidScore, EvaluationResult,
    FinalRecommendation, CompanyInsight, KeySignals,
    ScoreBreakdown, RAGTrace
)
from services.searcher import Searcher
from services.scorer import Scorer
from services.explainer import Explainer
from services.observer import Observer
from services.memory import get_memory_store
from services.historical_rag import get_historical_rag


# ==================== STATE DEFINITION ====================

class BidEvaluationState(TypedDict):
    """State that flows through the bid evaluation graph."""
    # Input
    bids: List[Dict]  # Raw bid data
    use_cached_result: bool
    
    # Intermediate state
    parsed_bids: List[Bid]
    insights: Dict[str, CompanyInsight]
    scores: List[BidScore]
    ranked_scores: List[BidScore]
    winner: Optional[BidScore]
    historical_insights: Dict[str, dict]
    
    # Output
    explanation: str
    result: Optional[EvaluationResult]
    
    # Metadata
    error: Optional[str]
    cache_hit: bool
    evaluation_id: Optional[str]


# ==================== NODE FUNCTIONS ====================

def parse_bids_node(state: BidEvaluationState) -> BidEvaluationState:
    """Parse raw bid data into Bid objects."""
    try:
        parsed = [Bid.from_dict(b) for b in state["bids"]]
        return {"parsed_bids": parsed, "error": None}
    except Exception as e:
        return {"parsed_bids": [], "error": f"Failed to parse bids: {str(e)}"}


def check_cache_node(state: BidEvaluationState) -> BidEvaluationState:
    """Check for cached evaluation result."""
    if not state.get("use_cached_result", False):
        return {"cache_hit": False}
    
    memory = get_memory_store()
    cached = memory.find_similar_evaluation(state["bids"])
    
    if cached:
        # Reconstruct result from cache
        return {
            "cache_hit": True,
            "result": _dict_to_result(cached["result"]),
            "evaluation_id": cached.get("id")
        }
    
    return {"cache_hit": False}


def gather_insights_node(state: BidEvaluationState) -> BidEvaluationState:
    """Gather external company insights."""
    searcher = Searcher()
    memory = get_memory_store()
    
    insights = {}
    companies_to_search = []
    
    # Check cache first
    for bid in state["parsed_bids"]:
        cached = memory.get_cached_company_insight(bid.company_name)
        if cached:
            data = cached["data"]
            insights[bid.company_name] = CompanyInsight(
                sources=data.get("sources", []),
                key_signals=KeySignals(
                    us_commercial_experience=data.get("key_signals", {}).get("us_commercial_experience", False),
                    project_scale_alignment=data.get("key_signals", {}).get("project_scale_alignment", "Medium"),
                    recent_negative_news=data.get("key_signals", {}).get("recent_negative_news", False)
                ),
                confidence_score=data.get("confidence_score", 0.5)
            )
        else:
            companies_to_search.append(bid.company_name)
    
    # Search for uncached companies
    if companies_to_search:
        search_results = searcher.search_batch(companies_to_search)
        insights.update(search_results)
    
    return {"insights": insights}


def gather_historical_node(state: BidEvaluationState) -> BidEvaluationState:
    """Gather historical insights from RAG."""
    historical_rag = get_historical_rag()
    
    if not historical_rag:
        return {"historical_insights": {}}
    
    historical_insights = {}
    for bid in state["parsed_bids"]:
        scope_items = bid.scope_coverage.included + bid.scope_coverage.subcontracted
        hist = historical_rag.get_historical_insight(bid.company_name, scope_items)
        historical_insights[bid.company_name] = {
            "win_rate": hist.historical_win_rate,
            "avg_score": hist.avg_score,
            "risk_patterns": hist.risk_patterns,
            "confidence": hist.confidence,
            "pricing_benchmark": hist.pricing_benchmark
        }
    
    return {"historical_insights": historical_insights}


def score_bids_node(state: BidEvaluationState) -> BidEvaluationState:
    """Score all bids."""
    scorer = Scorer(use_historical=True)
    historical_rag = get_historical_rag()
    
    scores = []
    for bid in state["parsed_bids"]:
        insight = state["insights"].get(bid.company_name)
        
        # Get historical insight if available
        historical_insight = None
        if historical_rag and bid.company_name in state.get("historical_insights", {}):
            scope_items = bid.scope_coverage.included + bid.scope_coverage.subcontracted
            historical_insight = historical_rag.get_historical_insight(
                bid.company_name, scope_items
            )
        
        score = scorer.score(bid, state["parsed_bids"], insight, historical_insight)
        scores.append(score)
    
    return {"scores": scores}


def rank_bids_node(state: BidEvaluationState) -> BidEvaluationState:
    """Rank bids by score and select winner."""
    ranked = sorted(state["scores"], key=lambda s: s.final_weighted_score, reverse=True)
    winner = ranked[0] if ranked else None
    
    return {"ranked_scores": ranked, "winner": winner}


def generate_explanation_node(state: BidEvaluationState) -> BidEvaluationState:
    """Generate LLM explanation for the recommendation."""
    if not state.get("winner"):
        return {"explanation": "No winner could be determined."}
    
    explainer = Explainer()
    explanation = explainer.explain(
        state["winner"],
        state["scores"],
        state["insights"]
    )
    
    return {"explanation": explanation}


def build_result_node(state: BidEvaluationState) -> BidEvaluationState:
    """Build the final evaluation result."""
    winner = state["winner"]
    ranked = state["ranked_scores"]
    insights = state["insights"]
    
    if not winner:
        return {"result": None, "error": "No winner determined"}
    
    explainer = Explainer()
    winner_insight = insights[winner.company_name]
    rationale = explainer.generate_rationale(winner, winner_insight)
    
    # Calculate recommendation confidence
    runner_up = ranked[1] if len(ranked) > 1 else None
    score_gap = (winner.final_weighted_score - runner_up.final_weighted_score) if runner_up else 0.1
    
    rec_confidence = min(
        0.95,
        winner.final_weighted_score * 0.5 +
        min(score_gap * 2, 0.3) +
        winner_insight.confidence_score * 0.2
    )
    
    observer = Observer()
    
    result = EvaluationResult(
        external_company_insights=insights,
        retrieved_context_used=True,
        rag_trace=observer.get_rag_trace(),
        bid_scores=state["scores"],
        ranked_recommendations=[s.bid_id for s in ranked],
        final_recommendation=FinalRecommendation(
            bid_id=winner.bid_id,
            company_name=winner.company_name,
            confidence=round(rec_confidence, 2),
            decision_rationale=rationale
        ),
        explanation=state["explanation"]
    )
    
    return {"result": result}


def persist_result_node(state: BidEvaluationState) -> BidEvaluationState:
    """Persist the evaluation result."""
    if not state.get("result"):
        return state
    
    memory = get_memory_store()
    historical_rag = get_historical_rag()
    
    result_dict = state["result"].to_dict()
    
    # Save evaluation
    eval_id = memory.save_evaluation(
        input_data=state["bids"],
        result=result_dict,
        traces=[]
    )
    
    # Cache company insights
    for company, insight in state["insights"].items():
        memory.cache_company_insight(company, insight.to_dict())
    
    # Index into Historical RAG
    if historical_rag:
        historical_rag.index_evaluation(result_dict, state["bids"])
    
    return {"evaluation_id": eval_id}


# ==================== CONDITIONAL EDGES ====================

def should_skip_evaluation(state: BidEvaluationState) -> str:
    """Determine if we should skip evaluation due to cache hit or error."""
    if state.get("error"):
        return "error"
    if state.get("cache_hit") and state.get("result"):
        return "cached"
    return "continue"


# ==================== GRAPH CONSTRUCTION ====================

def create_bid_evaluation_graph() -> StateGraph:
    """Create the LangGraph workflow for bid evaluation."""
    
    # Initialize the graph with our state type
    workflow = StateGraph(BidEvaluationState)
    
    # Add nodes
    workflow.add_node("parse_bids", parse_bids_node)
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("gather_insights", gather_insights_node)
    workflow.add_node("gather_historical", gather_historical_node)
    workflow.add_node("score_bids", score_bids_node)
    workflow.add_node("rank_bids", rank_bids_node)
    workflow.add_node("generate_explanation", generate_explanation_node)
    workflow.add_node("build_result", build_result_node)
    workflow.add_node("persist_result", persist_result_node)
    
    # Set entry point
    workflow.set_entry_point("parse_bids")
    
    # Add edges
    workflow.add_edge("parse_bids", "check_cache")
    
    # Conditional: skip if cached or error
    workflow.add_conditional_edges(
        "check_cache",
        should_skip_evaluation,
        {
            "cached": END,
            "error": END,
            "continue": "gather_insights"
        }
    )
    
    # Parallel insight gathering (sequential for now, can be parallelized)
    workflow.add_edge("gather_insights", "gather_historical")
    workflow.add_edge("gather_historical", "score_bids")
    
    # Scoring and ranking
    workflow.add_edge("score_bids", "rank_bids")
    workflow.add_edge("rank_bids", "generate_explanation")
    workflow.add_edge("generate_explanation", "build_result")
    workflow.add_edge("build_result", "persist_result")
    workflow.add_edge("persist_result", END)
    
    return workflow


# ==================== HELPER FUNCTIONS ====================

def _dict_to_result(data: Dict) -> EvaluationResult:
    """Convert dictionary back to EvaluationResult."""
    insights = {}
    for company, insight_data in data.get("external_company_insights", {}).items():
        insights[company] = CompanyInsight(
            sources=insight_data.get("sources", []),
            key_signals=KeySignals(
                us_commercial_experience=insight_data.get("key_signals", {}).get("us_commercial_experience", False),
                project_scale_alignment=insight_data.get("key_signals", {}).get("project_scale_alignment", "Medium"),
                recent_negative_news=insight_data.get("key_signals", {}).get("recent_negative_news", False)
            ),
            confidence_score=insight_data.get("confidence_score", 0.5)
        )
    
    scores = []
    for s in data.get("bid_scores", []):
        sb = s.get("score_breakdown", {})
        scores.append(BidScore(
            bid_id=s["bid_id"],
            company_name=s["company_name"],
            score_breakdown=ScoreBreakdown(
                cost_score=sb.get("cost_score", 0),
                timeline_score=sb.get("timeline_score", 0),
                scope_fit_score=sb.get("scope_fit_score", 0),
                risk_score=sb.get("risk_score", 0),
                external_reputation_score=sb.get("external_reputation_score", 0)
            ),
            final_weighted_score=s.get("final_weighted_score", 0),
            flags=s.get("flags", [])
        ))
    
    rag_data = data.get("rag_trace", {})
    rag_trace = RAGTrace(
        vector_store=rag_data.get("vector_store", ""),
        embedding_model=rag_data.get("embedding_model", ""),
        documents_retrieved_per_company=rag_data.get("documents_retrieved_per_company", 0),
        retrieval_confidence_threshold=rag_data.get("retrieval_confidence_threshold", 0.7)
    )
    
    rec = data.get("final_recommendation", {})
    
    return EvaluationResult(
        external_company_insights=insights,
        retrieved_context_used=data.get("retrieved_context_used", True),
        rag_trace=rag_trace,
        bid_scores=scores,
        ranked_recommendations=data.get("ranked_recommendations", []),
        final_recommendation=FinalRecommendation(
            bid_id=rec.get("bid_id", ""),
            company_name=rec.get("company_name", ""),
            confidence=rec.get("confidence", 0),
            decision_rationale=rec.get("decision_rationale", [])
        ),
        explanation=data.get("explanation", "")
    )


# ==================== COMPILED GRAPH ====================

# Create and compile the graph
_graph = create_bid_evaluation_graph()
bid_evaluation_app = _graph.compile()

