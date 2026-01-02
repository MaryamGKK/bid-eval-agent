# controller.py
"""Main workflow controller for bid evaluation with OpenTelemetry tracing and persistence."""

from typing import List, Dict, Optional

from models import (
    Bid, BidScore, EvaluationResult, 
    FinalRecommendation, CompanyInsight, KeySignals
)
from services.searcher import Searcher
from services.scorer import Scorer
from services.explainer import Explainer
from services.observer import Observer
from services.memory import get_memory_store, MemoryStore
from services.historical_rag import get_historical_rag


class Controller:
    """Orchestrate the bid evaluation workflow with tracing and persistence."""
    
    def __init__(self, use_cache: bool = True, use_historical: bool = True):
        self.searcher = Searcher()
        self.scorer = Scorer(use_historical=use_historical)
        self.explainer = Explainer()
        self.observer = Observer()
        self.memory = get_memory_store()
        self.historical_rag = get_historical_rag()
        self.use_cache = use_cache
        self.use_historical = use_historical
        self._last_evaluation_id: Optional[str] = None
        self._last_historical_insights: Dict[str, dict] = {}
    
    def evaluate(self, bids: List[Bid], use_cached_result: bool = False) -> EvaluationResult:
        """
        Run full evaluation pipeline with OpenTelemetry tracing.
        
        Pipeline:
        1. Check for cached result (optional)
        2. Search for company insights (with caching)
        3. Score all bids
        4. Rank and select winner
        5. Generate explanation
        6. Persist result
        
        Args:
            bids: List of Bid objects to evaluate
            use_cached_result: If True, return cached result if same input exists
            
        Raises:
            ValueError: If no bids provided
        """
        if not bids:
            raise ValueError("No bids provided for evaluation")
        
        # Convert bids to dict for caching/comparison
        input_data = [self._bid_to_dict(b) for b in bids]
        
        # Check for existing evaluation with same input
        if use_cached_result:
            existing = self.memory.find_similar_evaluation(input_data)
            if existing:
                # Return cached result
                return self._dict_to_result(existing["result"])
        
        self.observer.clear()
        
        with self.observer.span("bid_evaluation", {"bid_count": len(bids)}) as root_span:
            self.observer.log("request", "input", {"bid_count": len(bids)})
            
            # Step 1: Gather company insights (with caching)
            insights = self._gather_insights(bids)
            
            # Step 2: Score all bids
            scores = self._score_bids(bids, insights)
            
            # Step 3: Rank bids
            ranked = sorted(scores, key=lambda s: s.final_weighted_score, reverse=True)
            winner = ranked[0]
            
            # Step 4: Generate output
            result = self._generate_output(winner, scores, insights, ranked)
            
            self.observer.log("complete", "output", {
                "winner": winner.company_name,
                "score": winner.final_weighted_score
            })
            
            root_span.set_attribute("winner.company", winner.company_name)
            root_span.set_attribute("winner.score", winner.final_weighted_score)
        
        # Step 5: Persist result
        result_dict = result.to_dict()
        traces = self.observer.get_events()
        
        self._last_evaluation_id = self.memory.save_evaluation(
            input_data=input_data,
            result=result_dict,
            traces=traces
        )
        
        # Cache company insights
        for company, insight in insights.items():
            self.memory.cache_company_insight(company, insight.to_dict())
        
        # Step 6: Index into Historical RAG for future learning
        if self.historical_rag:
            try:
                self.historical_rag.index_evaluation(result_dict, input_data)
                self.observer.log("historical_rag", "indexed", {
                    "evaluation_id": self._last_evaluation_id,
                    "companies": list(insights.keys())
                })
            except Exception as e:
                self.observer.log("historical_rag", "index_error", {
                    "error": str(e)
                }, status="warning")
        
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
    
    def _dict_to_result(self, data: Dict) -> EvaluationResult:
        """Convert dictionary back to EvaluationResult."""
        # Reconstruct the result from stored data
        from models import RAGTrace
        
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
        
        from models import ScoreBreakdown
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
    
    def _gather_insights(self, bids: List[Bid]) -> Dict[str, CompanyInsight]:
        """Gather external insights for all companies with caching."""
        with self.observer.span("gather_insights", {"company_count": len(bids)}):
            self.observer.start_timer("search_all")
            
            insights = {}
            companies_to_search = []
            
            # Check cache first
            if self.use_cache:
                for bid in bids:
                    cached = self.memory.get_cached_company_insight(bid.company_name)
                    if cached:
                        # Use cached data
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
                        self.observer.log("search", "cache_hit", {
                            "company": bid.company_name,
                            "cached_at": cached["cached_at"]
                        })
                    else:
                        companies_to_search.append(bid.company_name)
            else:
                companies_to_search = [bid.company_name for bid in bids]
            
            # Search for companies not in cache
            if companies_to_search:
                search_results = self.searcher.search_batch(companies_to_search)
                insights.update(search_results)
            
            self.observer.stop_timer(
                "search_all",
                "search",
                "batch_complete",
                {
                    "companies_searched": len(companies_to_search),
                    "cache_hits": len(bids) - len(companies_to_search),
                    "total_sources": sum(len(i.sources) for i in insights.values())
                }
            )
            
            # Log individual results
            for company, insight in insights.items():
                self.observer.log("search", "result", {
                    "company": company,
                    "sources_found": len(insight.sources),
                    "confidence": insight.confidence_score
                })
        
        return insights
    
    def _score_bids(
        self, 
        bids: List[Bid], 
        insights: Dict[str, CompanyInsight]
    ) -> List[BidScore]:
        """Score all bids against evaluation criteria."""
        scores = []
        self._last_historical_insights = {}
        
        with self.observer.span("score_bids", {"bid_count": len(bids)}):
            for bid in bids:
                with self.observer.span(f"score_{bid.bid_id}", {"company": bid.company_name}):
                    insight = insights[bid.company_name]
                    
                    # Get historical insight if available
                    historical_insight = None
                    if self.historical_rag:
                        scope_items = bid.scope_coverage.included + bid.scope_coverage.subcontracted
                        historical_insight = self.historical_rag.get_historical_insight(
                            bid.company_name,
                            scope_items
                        )
                        # Store for later access
                        self._last_historical_insights[bid.company_name] = {
                            "win_rate": historical_insight.historical_win_rate,
                            "avg_score": historical_insight.avg_score,
                            "risk_patterns": historical_insight.risk_patterns,
                            "confidence": historical_insight.confidence,
                            "pricing_benchmark": historical_insight.pricing_benchmark
                        }
                        
                        self.observer.log("historical", "insight", {
                            "company": bid.company_name,
                            "win_rate": historical_insight.historical_win_rate,
                            "confidence": historical_insight.confidence
                        })
                    
                    score = self.scorer.score(bid, bids, insight, historical_insight)
                    scores.append(score)
                    
                    self.observer.log("score", "complete", {
                        "bid_id": bid.bid_id,
                        "company": bid.company_name,
                        "final_score": score.final_weighted_score,
                        "flags": score.flags,
                        "historical_used": historical_insight is not None
                    })
        
        return scores
    
    def _generate_output(
        self,
        winner: BidScore,
        scores: List[BidScore],
        insights: Dict[str, CompanyInsight],
        ranked: List[BidScore]
    ) -> EvaluationResult:
        """Generate the final evaluation output."""
        
        with self.observer.span("generate_explanation"):
            self.observer.start_timer("llm_explain")
            explanation = self.explainer.explain(winner, scores, insights)
            self.observer.stop_timer("llm_explain", "llm", "complete", {"task": "explanation"})
        
        winner_insight = insights[winner.company_name]
        rationale = self.explainer.generate_rationale(winner, winner_insight)
        
        # Calculate recommendation confidence
        runner_up = ranked[1] if len(ranked) > 1 else None
        score_gap = (winner.final_weighted_score - runner_up.final_weighted_score) if runner_up else 0.1
        
        rec_confidence = min(
            0.95,
            winner.final_weighted_score * 0.5 + 
            min(score_gap * 2, 0.3) + 
            winner_insight.confidence_score * 0.2
        )
        
        return EvaluationResult(
            external_company_insights=insights,
            retrieved_context_used=True,
            rag_trace=self.observer.get_rag_trace(),
            bid_scores=scores,
            ranked_recommendations=[s.bid_id for s in ranked],
            final_recommendation=FinalRecommendation(
                bid_id=winner.bid_id,
                company_name=winner.company_name,
                confidence=round(rec_confidence, 2),
                decision_rationale=rationale
            ),
            explanation=explanation
        )
    
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
    
    def clear_cache(self):
        """Clear the search cache."""
        self.searcher.clear_cache()
    
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
