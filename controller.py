# controller.py
"""Main workflow controller for bid evaluation with LangGraph orchestration."""

from typing import List, Dict, Optional

from models import Bid, BidScore, EvaluationResult
from services.graph import bid_evaluation_app
from services.observer import Observer
from services.memory import get_memory_store
from services.historical_rag import get_historical_rag


class Controller:
    """Orchestrate the bid evaluation workflow with LangGraph."""
    
    def __init__(self):
        self.observer = Observer()
        self.memory = get_memory_store()
        self.historical_rag = get_historical_rag()
        self._last_evaluation_id: Optional[str] = None
        self._last_historical_insights: Dict[str, dict] = {}
        self._last_graph_state: Optional[Dict] = None
    
    def evaluate(self, bids: List[Bid], use_cached_result: bool = False) -> EvaluationResult:
        """
        Run full evaluation pipeline using LangGraph workflow.
        
        LangGraph Pipeline:
        1. parse_bids - Parse raw bid data
        2. check_cache - Check for cached result (optional)
        3. gather_insights - Search for company insights
        4. gather_historical - Get historical RAG insights
        5. score_bids - Score all bids
        6. rank_bids - Rank and select winner
        7. generate_explanation - Generate LLM explanation
        8. build_result - Build final result
        9. persist_result - Persist to memory
        
        Args:
            bids: List of Bid objects to evaluate
            use_cached_result: If True, return cached result if same input exists
            
        Raises:
            ValueError: If no bids provided
        """
        if not bids:
            raise ValueError("No bids provided for evaluation")
        
        # Convert bids to dict for LangGraph state
        input_data = [self._bid_to_dict(b) for b in bids]
        
        self.observer.clear()
        self.observer.log("request", "input", {"bid_count": len(input_data)})
        
        # Prepare initial state
        initial_state = {
            "bids": input_data,
            "use_cached_result": use_cached_result,
            "parsed_bids": [],
            "insights": {},
            "scores": [],
            "ranked_scores": [],
            "winner": None,
            "historical_insights": {},
            "explanation": "",
            "result": None,
            "error": None,
            "cache_hit": False,
            "evaluation_id": None
        }
        
        # Run the graph
        with self.observer.span("langgraph_evaluation", {"bid_count": len(input_data)}) as root_span:
            final_state = bid_evaluation_app.invoke(initial_state)
            self._last_graph_state = final_state
            
            if final_state.get("error"):
                raise ValueError(final_state["error"])
            
            result = final_state["result"]
            self._last_evaluation_id = final_state.get("evaluation_id")
            self._last_historical_insights = final_state.get("historical_insights", {})
            
            if result:
                root_span.set_attribute("winner.company", result.final_recommendation.company_name)
                root_span.set_attribute("winner.score", result.final_recommendation.confidence)
                root_span.set_attribute("graph.cache_hit", final_state.get("cache_hit", False))
            
            self.observer.log("complete", "output", {
                "winner": result.final_recommendation.company_name if result else None,
                "cache_hit": final_state.get("cache_hit", False)
            })
        
        return result
    
    def _bid_to_dict(self, bid: Bid) -> Dict:
        """Convert Bid object to dictionary."""
        return {
            "bid_id": bid.bid_id,
            "company_name": bid.company_name,
            "cost": bid.cost,
            "timeline": {
                "estimated_months": bid.timeline.estimated_months,
                "confidence_level": bid.timeline.confidence_level,
                "critical_path_risk": bid.timeline.critical_path_risk
            },
            "scope_coverage": {
                "included": bid.scope_coverage.included,
                "excluded": bid.scope_coverage.excluded,
                "subcontracted": bid.scope_coverage.subcontracted
            }
        }
    
    # ==================== PUBLIC API ====================
    
    def get_events(self) -> List[Dict]:
        """Get all logged events."""
        return self.observer.get_events()
    
    def get_summary(self) -> Dict:
        """Get observability summary."""
        return self.observer.get_summary()
    
    def get_last_evaluation_id(self) -> Optional[str]:
        """Get the ID of the last evaluation."""
        return self._last_evaluation_id
    
    def get_evaluation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent evaluation history."""
        return self.memory.get_recent_evaluations(limit)
    
    def get_evaluation_by_id(self, eval_id: str) -> Optional[Dict]:
        """Get a specific evaluation by ID."""
        record = self.memory.get_evaluation(eval_id)
        if record:
            return {
                "id": record.id,
                "timestamp": record.timestamp,
                "bid_count": record.bid_count,
                "winner_id": record.winner_id,
                "winner_company": record.winner_company,
                "winner_score": record.winner_score,
                "result": record.result_data,
                "traces": record.trace_data
            }
        return None
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        return {
            "evaluations": self.memory.get_evaluation_stats(),
            "errors": self.memory.get_error_stats(),
            "feedback": self.memory.get_feedback_stats(),
            "cached_companies": len(self.memory.get_all_cached_companies())
        }
    
    def submit_feedback(
        self, 
        rating: int = None, 
        comment: str = None,
        correct_winner_id: str = None
    ):
        """Submit feedback for the last evaluation."""
        if self._last_evaluation_id:
            feedback_type = "rating" if rating else ("correction" if correct_winner_id else "comment")
            self.memory.save_feedback(
                evaluation_id=self._last_evaluation_id,
                feedback_type=feedback_type,
                rating=rating,
                comment=comment,
                correct_winner_id=correct_winner_id
            )
    
    def get_historical_insights(self) -> Dict[str, dict]:
        """Get historical insights from the last evaluation."""
        return self._last_historical_insights
    
    def get_historical_stats(self) -> Dict:
        """Get historical RAG statistics."""
        if self.historical_rag:
            return {
                "enabled": True,
                **self.historical_rag.get_stats()
            }
        return {"enabled": False}
    
    def get_top_performers(self, limit: int = 5) -> List[Dict]:
        """Get top performing companies from historical data."""
        if self.historical_rag:
            return self.historical_rag.get_top_performers(limit)
        return []
    
    def get_company_history(self, company_name: str) -> Optional[Dict]:
        """Get historical performance for a specific company."""
        if self.historical_rag:
            return self.historical_rag.get_company_history(company_name)
        return None
    
    def get_graph_state(self) -> Optional[Dict]:
        """Get the last LangGraph execution state."""
        return self._last_graph_state
    
    def get_workflow_info(self) -> Dict:
        """Get information about the workflow configuration."""
        return {
            "using_langgraph": True,
            "nodes": [
                "parse_bids",
                "check_cache", 
                "gather_insights",
                "gather_historical",
                "score_bids",
                "rank_bids",
                "generate_explanation",
                "build_result",
                "persist_result"
            ]
        }
