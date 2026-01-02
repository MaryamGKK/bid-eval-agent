# services/historical_rag.py
"""Historical RAG for learning from past bid evaluations."""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@dataclass
class HistoricalInsight:
    """Insight derived from historical data."""
    company_name: str
    historical_win_rate: float
    avg_score: float
    similar_projects: List[Dict]
    pricing_benchmark: Dict[str, float]
    risk_patterns: List[str]
    confidence: float


class HistoricalRAG:
    """RAG system for learning from historical bid evaluations."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Run: pip install chromadb"
            )
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use sentence-transformers for better semantic search
        # Load from local path if available (faster startup)
        import os
        local_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "all-MiniLM-L6-v2")
        
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        if os.path.exists(local_model_path):
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=local_model_path
            )
        else:
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        # Collections for different data types
        self.evaluations = self.client.get_or_create_collection(
            name="evaluations",
            metadata={"description": "Past bid evaluation results"},
            embedding_function=self._embedding_fn
        )
        
        self.company_profiles = self.client.get_or_create_collection(
            name="company_profiles",
            metadata={"description": "Company performance profiles"},
            embedding_function=self._embedding_fn
        )
        
        self.bid_patterns = self.client.get_or_create_collection(
            name="bid_patterns",
            metadata={"description": "Bid patterns and benchmarks"},
            embedding_function=self._embedding_fn
        )
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _create_evaluation_text(self, evaluation: Dict) -> str:
        """Create searchable text from evaluation."""
        winner = evaluation.get("final_recommendation", {})
        scores = evaluation.get("bid_scores", [])
        
        text_parts = [
            f"Winner: {winner.get('company_name', 'Unknown')}",
            f"Confidence: {winner.get('confidence', 0):.2f}",
            f"Rationale: {', '.join(winner.get('decision_rationale', []))}",
        ]
        
        for score in scores:
            sb = score.get("score_breakdown", {})
            text_parts.append(
                f"Bid {score['company_name']}: "
                f"cost={sb.get('cost_score', 0):.2f}, "
                f"timeline={sb.get('timeline_score', 0):.2f}, "
                f"scope={sb.get('scope_fit_score', 0):.2f}, "
                f"risk={sb.get('risk_score', 0):.2f}, "
                f"reputation={sb.get('external_reputation_score', 0):.2f}, "
                f"final={score.get('final_weighted_score', 0):.2f}, "
                f"flags={score.get('flags', [])}"
            )
        
        return " | ".join(text_parts)
    
    def _create_company_text(self, company_data: Dict) -> str:
        """Create searchable text from company data."""
        return (
            f"Company: {company_data.get('company_name', 'Unknown')} | "
            f"Wins: {company_data.get('total_wins', 0)} | "
            f"Evaluations: {company_data.get('total_evaluations', 0)} | "
            f"Avg Score: {company_data.get('avg_score', 0):.2f} | "
            f"Common Flags: {company_data.get('common_flags', [])} | "
            f"Strengths: {company_data.get('strengths', [])} | "
            f"Weaknesses: {company_data.get('weaknesses', [])}"
        )
    
    def _create_bid_pattern_text(self, bid: Dict) -> str:
        """Create searchable text from bid pattern."""
        return (
            f"Company: {bid.get('company_name', 'Unknown')} | "
            f"Cost: ${bid.get('total_cost', 0):,.0f} | "
            f"Timeline: {bid.get('timeline_months', 0)} months | "
            f"Scope: {bid.get('scope_items', [])} | "
            f"Risk: {bid.get('critical_path_risk', 'Unknown')} | "
            f"Confidence: {bid.get('confidence_level', 0):.2f}"
        )
    
    # ==================== INDEX OPERATIONS ====================
    
    def index_evaluation(self, evaluation: Dict, input_bids: List[Dict]):
        """Index a completed evaluation for future reference."""
        eval_id = self._generate_id(json.dumps(evaluation, sort_keys=True))
        eval_text = self._create_evaluation_text(evaluation)
        
        # Index evaluation
        self.evaluations.upsert(
            ids=[eval_id],
            documents=[eval_text],
            metadatas=[{
                "timestamp": datetime.now().isoformat(),
                "winner_company": evaluation.get("final_recommendation", {}).get("company_name", ""),
                "winner_score": evaluation.get("final_recommendation", {}).get("confidence", 0),
                "bid_count": len(evaluation.get("bid_scores", []))
            }]
        )
        
        # Update company profiles
        for score in evaluation.get("bid_scores", []):
            self._update_company_profile(score, evaluation)
        
        # Index bid patterns
        for bid in input_bids:
            self._index_bid_pattern(bid)
    
    def _update_company_profile(self, score: Dict, evaluation: Dict):
        """Update company profile based on evaluation result."""
        company = score["company_name"]
        is_winner = company == evaluation.get("final_recommendation", {}).get("company_name")
        
        # Get existing profile
        existing = self.company_profiles.get(
            where={"company_name": company},
            include=["metadatas"]
        )
        
        if existing["ids"]:
            # Update existing
            meta = existing["metadatas"][0]
            total_evals = meta.get("total_evaluations", 0) + 1
            total_wins = meta.get("total_wins", 0) + (1 if is_winner else 0)
            avg_score = (
                (meta.get("avg_score", 0) * meta.get("total_evaluations", 0) + 
                 score.get("final_weighted_score", 0)) / total_evals
            )
            
            # Track flags
            all_flags = meta.get("all_flags", [])
            all_flags.extend(score.get("flags", []))
            
            profile_data = {
                "company_name": company,
                "total_evaluations": total_evals,
                "total_wins": total_wins,
                "win_rate": total_wins / total_evals,
                "avg_score": avg_score,
                "common_flags": list(set(all_flags)),
                "all_flags": all_flags,
                "last_updated": datetime.now().isoformat()
            }
            
            self.company_profiles.update(
                ids=[existing["ids"][0]],
                documents=[self._create_company_text(profile_data)],
                metadatas=[profile_data]
            )
        else:
            # Create new profile
            profile_id = self._generate_id(company)
            profile_data = {
                "company_name": company,
                "total_evaluations": 1,
                "total_wins": 1 if is_winner else 0,
                "win_rate": 1.0 if is_winner else 0.0,
                "avg_score": score.get("final_weighted_score", 0),
                "common_flags": score.get("flags", []),
                "all_flags": score.get("flags", []),
                "last_updated": datetime.now().isoformat()
            }
            
            self.company_profiles.add(
                ids=[profile_id],
                documents=[self._create_company_text(profile_data)],
                metadatas=[profile_data]
            )
    
    def _index_bid_pattern(self, bid: Dict):
        """Index bid pattern for pricing/timeline benchmarks."""
        bid_id = self._generate_id(json.dumps(bid, sort_keys=True))
        
        pattern_data = {
            "company_name": bid.get("company_name", ""),
            "total_cost": bid.get("cost", {}).get("total_usd", 0),
            "timeline_months": bid.get("timeline", {}).get("estimated_months", 0),
            "confidence_level": bid.get("timeline", {}).get("confidence_level", 0),
            "critical_path_risk": bid.get("timeline", {}).get("critical_path_risk", ""),
            "scope_items": bid.get("scope_coverage", {}).get("included", []),
            "timestamp": datetime.now().isoformat()
        }
        
        self.bid_patterns.upsert(
            ids=[bid_id],
            documents=[self._create_bid_pattern_text(pattern_data)],
            metadatas=[pattern_data]
        )
    
    # ==================== QUERY OPERATIONS ====================
    
    def get_company_history(self, company_name: str) -> Optional[Dict]:
        """Get historical performance for a company."""
        results = self.company_profiles.get(
            where={"company_name": company_name},
            include=["metadatas"]
        )
        
        if results["ids"]:
            return results["metadatas"][0]
        return None
    
    def find_similar_evaluations(self, query: str, n_results: int = 5) -> List[Dict]:
        """Find similar past evaluations."""
        results = self.evaluations.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        similar = []
        for i, doc in enumerate(results["documents"][0]):
            similar.append({
                "document": doc,
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return similar
    
    def get_pricing_benchmark(self, scope_items: List[str]) -> Dict[str, float]:
        """Get pricing benchmarks for given scope items."""
        # Query for similar bids
        scope_query = " ".join(scope_items)
        results = self.bid_patterns.query(
            query_texts=[f"Scope: {scope_query}"],
            n_results=10,
            include=["metadatas"]
        )
        
        if not results["ids"][0]:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        
        costs = [m.get("total_cost", 0) for m in results["metadatas"][0] if m.get("total_cost", 0) > 0]
        
        if not costs:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        
        sorted_costs = sorted(costs)
        return {
            "min": min(costs),
            "max": max(costs),
            "avg": sum(costs) / len(costs),
            "median": sorted_costs[len(sorted_costs) // 2],
            "sample_size": len(costs)
        }
    
    def get_timeline_benchmark(self, scope_items: List[str]) -> Dict[str, float]:
        """Get timeline benchmarks for given scope items."""
        scope_query = " ".join(scope_items)
        results = self.bid_patterns.query(
            query_texts=[f"Scope: {scope_query}"],
            n_results=10,
            include=["metadatas"]
        )
        
        if not results["ids"][0]:
            return {"min": 0, "max": 0, "avg": 0}
        
        timelines = [m.get("timeline_months", 0) for m in results["metadatas"][0] if m.get("timeline_months", 0) > 0]
        
        if not timelines:
            return {"min": 0, "max": 0, "avg": 0}
        
        return {
            "min": min(timelines),
            "max": max(timelines),
            "avg": sum(timelines) / len(timelines),
            "sample_size": len(timelines)
        }
    
    def get_historical_insight(self, company_name: str, scope_items: List[str]) -> HistoricalInsight:
        """Get comprehensive historical insight for a company."""
        # Get company history
        history = self.get_company_history(company_name)
        
        # Get similar evaluations
        similar = self.find_similar_evaluations(
            f"Company: {company_name} Scope: {' '.join(scope_items)}",
            n_results=3
        )
        
        # Get benchmarks
        pricing = self.get_pricing_benchmark(scope_items)
        
        # Identify risk patterns from flags
        risk_patterns = []
        if history:
            flags = history.get("common_flags", [])
            if "f1" in flags:
                risk_patterns.append("Frequently subcontracts critical work")
            if "f2" in flags:
                risk_patterns.append("History of timeline overruns")
            if "f3" in flags:
                risk_patterns.append("Often has low confidence levels")
            if "f4" in flags:
                risk_patterns.append("Scope coverage issues in past bids")
            if "x1" in flags:
                risk_patterns.append("Limited US commercial experience")
        
        return HistoricalInsight(
            company_name=company_name,
            historical_win_rate=history.get("win_rate", 0) if history else 0,
            avg_score=history.get("avg_score", 0) if history else 0,
            similar_projects=[s["metadata"] for s in similar],
            pricing_benchmark=pricing,
            risk_patterns=risk_patterns,
            confidence=min(0.9, (history.get("total_evaluations", 0) / 10)) if history else 0.1
        )
    
    # ==================== STATISTICS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        return {
            "total_evaluations": self.evaluations.count(),
            "total_companies": self.company_profiles.count(),
            "total_bid_patterns": self.bid_patterns.count()
        }
    
    def get_top_performers(self, limit: int = 5) -> List[Dict]:
        """Get top performing companies by win rate."""
        results = self.company_profiles.get(
            include=["metadatas"],
            limit=100  # Get all, then sort
        )
        
        if not results["ids"]:
            return []
        
        # Sort by win rate (with min evaluations threshold)
        companies = [
            m for m in results["metadatas"] 
            if m.get("total_evaluations", 0) >= 2  # Min 2 evaluations
        ]
        
        sorted_companies = sorted(
            companies, 
            key=lambda x: (x.get("win_rate", 0), x.get("avg_score", 0)),
            reverse=True
        )
        
        return sorted_companies[:limit]


# Singleton instance
_historical_rag: Optional[HistoricalRAG] = None


def get_historical_rag(persist_directory: str = "./chroma_db") -> Optional[HistoricalRAG]:
    """Get or create historical RAG singleton."""
    global _historical_rag
    
    if not CHROMADB_AVAILABLE:
        return None
    
    if _historical_rag is None:
        try:
            _historical_rag = HistoricalRAG(persist_directory)
        except Exception:
            return None
    
    return _historical_rag

