# services/graph.py
"""LangGraph-based workflow for bid evaluation with LangSmith-style tracing."""

import time
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
from services.observer import Observer, RunType
from services.memory import get_memory_store
from services.historical_rag import get_historical_rag


# Global observer for tracing
_observer = Observer()


def get_observer() -> Observer:
    """Get the global observer instance."""
    return _observer


# ==================== STATE DEFINITION ====================

class BidEvaluationState(TypedDict):
    """State that flows through the bid evaluation graph."""
    # Input
    bids: List[Dict]
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
    
    # Tracing
    trace_id: Optional[str]


# ==================== NODE FUNCTIONS WITH TRACING ====================

def parse_bids_node(state: BidEvaluationState) -> BidEvaluationState:
    """Parse raw bid data into Bid objects."""
    observer = get_observer()
    
    with observer.trace_run(
        name="parse_bids",
        run_type=RunType.PARSER,
        inputs={"bid_count": len(state["bids"]), "bids": state["bids"]},
        tags=["parser", "input"]
    ) as run:
        start = time.perf_counter()
        
        try:
            parsed = [Bid.from_dict(b) for b in state["bids"]]
            
            # Log parsed output
            run.outputs = {
                "parsed_count": len(parsed),
                "companies": [b.company_name for b in parsed],
                "total_cost_range": {
                    "min": min(b.total_cost for b in parsed),
                    "max": max(b.total_cost for b in parsed)
                }
            }
            run.latency_ms = round((time.perf_counter() - start) * 1000, 2)
            
            return {"parsed_bids": parsed, "error": None}
            
        except Exception as e:
            run.outputs = {"error": str(e)}
            raise


def check_cache_node(state: BidEvaluationState) -> BidEvaluationState:
    """Check for cached evaluation result."""
    observer = get_observer()
    
    with observer.trace_run(
        name="check_cache",
        run_type=RunType.RETRIEVER,
        inputs={"use_cached": state.get("use_cached_result", False)},
        tags=["cache", "retriever"]
    ) as run:
        if not state.get("use_cached_result", False):
            run.outputs = {"cache_hit": False, "reason": "caching_disabled"}
            return {"cache_hit": False}
        
        memory = get_memory_store()
        cached = memory.find_similar_evaluation(state["bids"])
        
        if cached:
            run.outputs = {
                "cache_hit": True,
                "cached_id": cached.get("id"),
                "cached_timestamp": cached.get("timestamp")
            }
            return {
                "cache_hit": True,
                "result": _dict_to_result(cached["result"]),
                "evaluation_id": cached.get("id")
            }
        
        run.outputs = {"cache_hit": False, "reason": "no_match_found"}
        return {"cache_hit": False}


def gather_insights_node(state: BidEvaluationState) -> BidEvaluationState:
    """Gather external company insights."""
    observer = get_observer()
    
    companies = [b.company_name for b in state["parsed_bids"]]
    
    with observer.trace_run(
        name="gather_insights",
        run_type=RunType.CHAIN,
        inputs={"companies": companies},
        tags=["search", "external"]
    ) as run:
        searcher = Searcher()
        memory = get_memory_store()
        
        insights = {}
        companies_to_search = []
        cached_companies = []
        
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
                cached_companies.append(bid.company_name)
            else:
                companies_to_search.append(bid.company_name)
        
        # Search for uncached companies
        if companies_to_search:
            for company in companies_to_search:
                search_start = time.perf_counter()
                search_results = searcher.search_batch([company])
                search_latency = (time.perf_counter() - search_start) * 1000
                
                # Log individual search as tool call
                if company in search_results:
                    insight = search_results[company]
                    insights[company] = insight
                    
                    observer.log_tool_call(
                        name=f"search_{company}",
                        tool_name="tavily_serpapi_search",
                        tool_input={"company": company, "query": f"{company} construction company"},
                        tool_output={
                            "sources_count": len(insight.sources),
                            "us_experience": insight.key_signals.us_commercial_experience,
                            "scale_alignment": insight.key_signals.project_scale_alignment,
                            "confidence": insight.confidence_score
                        },
                        latency_ms=round(search_latency, 2)
                    )
        
        run.outputs = {
            "total_companies": len(companies),
            "cached_count": len(cached_companies),
            "searched_count": len(companies_to_search),
            "insights_gathered": {
                company: {
                    "confidence": insights[company].confidence_score,
                    "us_experience": insights[company].key_signals.us_commercial_experience
                }
                for company in insights
            }
        }
        
        return {"insights": insights}


def gather_historical_node(state: BidEvaluationState) -> BidEvaluationState:
    """Gather historical insights from RAG."""
    observer = get_observer()
    
    with observer.trace_run(
        name="gather_historical",
        run_type=RunType.RETRIEVER,
        inputs={"companies": [b.company_name for b in state["parsed_bids"]]},
        tags=["rag", "historical"]
    ) as run:
        historical_rag = get_historical_rag()
        
        if not historical_rag:
            run.outputs = {"enabled": False, "reason": "rag_not_available"}
            return {"historical_insights": {}}
        
        historical_insights = {}
        for bid in state["parsed_bids"]:
            scope_items = bid.scope_coverage.included + bid.scope_coverage.subcontracted
            
            rag_start = time.perf_counter()
            hist = historical_rag.get_historical_insight(bid.company_name, scope_items)
            rag_latency = (time.perf_counter() - rag_start) * 1000
            
            historical_insights[bid.company_name] = {
                "win_rate": hist.historical_win_rate,
                "avg_score": hist.avg_score,
                "risk_patterns": hist.risk_patterns,
                "confidence": hist.confidence,
                "pricing_benchmark": hist.pricing_benchmark
            }
            
            # Log as retriever call
            observer.log_retriever_call(
                name=f"historical_rag_{bid.company_name}",
                query=f"Company: {bid.company_name}, Scope: {', '.join(scope_items[:3])}",
                documents=[
                    {"type": "company_profile", "win_rate": hist.historical_win_rate},
                    {"type": "similar_projects", "count": len(hist.similar_projects)},
                    {"type": "pricing_benchmark", "data": hist.pricing_benchmark}
                ],
                latency_ms=round(rag_latency, 2)
            )
        
        run.outputs = {
            "companies_processed": len(historical_insights),
            "insights_summary": {
                company: {
                    "win_rate": data["win_rate"],
                    "avg_score": data["avg_score"],
                    "risk_patterns_count": len(data["risk_patterns"])
                }
                for company, data in historical_insights.items()
            }
        }
        
        return {"historical_insights": historical_insights}


def score_bids_node(state: BidEvaluationState) -> BidEvaluationState:
    """Score all bids."""
    observer = get_observer()
    
    with observer.trace_run(
        name="score_bids",
        run_type=RunType.CHAIN,
        inputs={
            "bid_count": len(state["parsed_bids"]),
            "companies": [b.company_name for b in state["parsed_bids"]]
        },
        tags=["scoring", "evaluation"]
    ) as run:
        scorer = Scorer(use_historical=True)
        historical_rag = get_historical_rag()
        
        scores = []
        score_details = {}
        
        for bid in state["parsed_bids"]:
            score_start = time.perf_counter()
            
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
            
            score_latency = (time.perf_counter() - score_start) * 1000
            
            # Log individual scoring as tool call
            observer.log_tool_call(
                name=f"score_{bid.company_name}",
                tool_name="weighted_scorer",
                tool_input={
                    "bid_id": bid.bid_id,
                    "company": bid.company_name,
                    "total_cost": bid.total_cost,
                    "timeline_months": bid.timeline.estimated_months,
                    "confidence_level": bid.timeline.confidence_level
                },
                tool_output={
                    "final_score": score.final_weighted_score,
                    "cost_score": score.score_breakdown.cost_score,
                    "timeline_score": score.score_breakdown.timeline_score,
                    "scope_score": score.score_breakdown.scope_fit_score,
                    "risk_score": score.score_breakdown.risk_score,
                    "reputation_score": score.score_breakdown.external_reputation_score,
                    "flags": score.flags
                },
                latency_ms=round(score_latency, 2)
            )
            
            score_details[bid.company_name] = {
                "final_score": score.final_weighted_score,
                "flags": score.flags
            }
        
        run.outputs = {
            "scores_calculated": len(scores),
            "score_ranking": sorted(
                [(s.company_name, s.final_weighted_score) for s in scores],
                key=lambda x: x[1],
                reverse=True
            ),
            "details": score_details
        }
        
        return {"scores": scores}


def rank_bids_node(state: BidEvaluationState) -> BidEvaluationState:
    """Rank bids by score and select winner."""
    observer = get_observer()
    
    with observer.trace_run(
        name="rank_bids",
        run_type=RunType.CHAIN,
        inputs={"scores_count": len(state["scores"])},
        tags=["ranking", "selection"]
    ) as run:
        ranked = sorted(state["scores"], key=lambda s: s.final_weighted_score, reverse=True)
        winner = ranked[0] if ranked else None
        
        run.outputs = {
            "ranking": [
                {
                    "rank": i + 1,
                    "company": s.company_name,
                    "score": s.final_weighted_score,
                    "flags": s.flags
                }
                for i, s in enumerate(ranked)
            ],
            "winner": {
                "company": winner.company_name,
                "score": winner.final_weighted_score,
                "bid_id": winner.bid_id
            } if winner else None,
            "score_gap": (ranked[0].final_weighted_score - ranked[1].final_weighted_score) if len(ranked) > 1 else 0
        }
        
        return {"ranked_scores": ranked, "winner": winner}


def generate_explanation_node(state: BidEvaluationState) -> BidEvaluationState:
    """Generate LLM explanation for the recommendation."""
    observer = get_observer()
    
    if not state.get("winner"):
        return {"explanation": "No winner could be determined."}
    
    winner = state["winner"]
    
    with observer.trace_run(
        name="generate_explanation",
        run_type=RunType.LLM,
        inputs={
            "winner": winner.company_name,
            "winner_score": winner.final_weighted_score,
            "all_scores_count": len(state["scores"])
        },
        tags=["llm", "explanation"]
    ) as run:
        from config import GROQ_MODEL
        
        explainer = Explainer()
        
        llm_start = time.perf_counter()
        explanation = explainer.explain(
            winner,
            state["scores"],
            state["insights"]
        )
        llm_latency = (time.perf_counter() - llm_start) * 1000
        
        # Log LLM call with details
        observer.log_llm_call(
            name="groq_explanation",
            model=GROQ_MODEL,
            prompts=[
                {"role": "system", "content": "You are a senior construction bid evaluation analyst..."},
                {"role": "user", "content": f"Analyze winner: {winner.company_name} with score {winner.final_weighted_score}"}
            ],
            response=explanation[:500] + "..." if len(explanation) > 500 else explanation,
            token_usage={
                "prompt_tokens": len(explanation.split()) // 4,  # Estimate
                "completion_tokens": len(explanation.split()),
                "total_tokens": len(explanation.split()) * 5 // 4
            },
            model_parameters={"temperature": 0.1, "max_tokens": 65536},
            latency_ms=round(llm_latency, 2)
        )
        
        run.outputs = {
            "explanation_length": len(explanation),
            "model": GROQ_MODEL,
            "latency_ms": round(llm_latency, 2)
        }
        run.model = GROQ_MODEL
        run.latency_ms = round(llm_latency, 2)
        
        return {"explanation": explanation}


def build_result_node(state: BidEvaluationState) -> BidEvaluationState:
    """Build the final evaluation result."""
    observer = get_observer()
    
    winner = state["winner"]
    ranked = state["ranked_scores"]
    insights = state["insights"]
    
    if not winner:
        return {"result": None, "error": "No winner determined"}
    
    with observer.trace_run(
        name="build_result",
        run_type=RunType.CHAIN,
        inputs={"winner": winner.company_name},
        tags=["output", "result"]
    ) as run:
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
        
        run.outputs = {
            "winner": winner.company_name,
            "confidence": round(rec_confidence, 2),
            "score_gap": round(score_gap, 4),
            "rationale_count": len(rationale),
            "total_bids_evaluated": len(state["scores"])
        }
        
        return {"result": result}


def persist_result_node(state: BidEvaluationState) -> BidEvaluationState:
    """Persist the evaluation result."""
    observer = get_observer()
    
    if not state.get("result"):
        return state
    
    with observer.trace_run(
        name="persist_result",
        run_type=RunType.TOOL,
        inputs={"has_result": True},
        tags=["persistence", "storage"]
    ) as run:
        memory = get_memory_store()
        historical_rag = get_historical_rag()
        
        result_dict = state["result"].to_dict()
        
        # Save evaluation
        eval_id = memory.save_evaluation(
            input_data=state["bids"],
            result=result_dict,
            traces=observer.get_runs()
        )
        
        # Cache company insights
        for company, insight in state["insights"].items():
            memory.cache_company_insight(company, insight.to_dict())
        
        # Index into Historical RAG
        if historical_rag:
            historical_rag.index_evaluation(result_dict, state["bids"])
        
        run.outputs = {
            "evaluation_id": eval_id,
            "companies_cached": list(state["insights"].keys()),
            "rag_indexed": historical_rag is not None
        }
        
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
    
    workflow.add_conditional_edges(
        "check_cache",
        should_skip_evaluation,
        {
            "cached": END,
            "error": END,
            "continue": "gather_insights"
        }
    )
    
    workflow.add_edge("gather_insights", "gather_historical")
    workflow.add_edge("gather_historical", "score_bids")
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

_graph = create_bid_evaluation_graph()
bid_evaluation_app = _graph.compile()
