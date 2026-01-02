# models.py
"""Data models for Bid Evaluation Agent POC."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union


# ============ INPUT MODELS ============

@dataclass
class CostBreakdown:
    labor: float
    materials: float
    contingency: float


@dataclass
class Timeline:
    estimated_months: int
    confidence_level: float
    critical_path_risk: str  # "Low", "Medium", "High"


@dataclass
class ScopeCoverage:
    included: List[str]
    excluded: List[str]
    subcontracted: List[str]


@dataclass
class DeliveryHistory:
    on_time_percentage: float
    on_budget_percentage: float
    known_delays: List[str]


@dataclass
class LegalCompliance:
    open_litigation: bool
    safety_violations_last_5_years: int


@dataclass
class BidMetadata:
    submission_channel: str
    submission_timestamp: str
    bid_revision: int


@dataclass
class Bid:
    """Full bid input model."""
    bid_id: str
    company_name: str
    cost: Dict[str, Any]  # Contains total_usd, currency, cost_breakdown
    timeline: Timeline
    scope_coverage: ScopeCoverage
    assumptions: List[str]
    dependencies: List[str]
    prior_similar_projects_count: int
    delivery_history: DeliveryHistory
    legal_and_compliance: LegalCompliance
    bid_metadata: BidMetadata
    
    @property
    def total_cost(self) -> float:
        return self.cost.get("total_usd", 0)
    
    @property
    def cost_breakdown(self) -> CostBreakdown:
        cb = self.cost.get("cost_breakdown", {})
        return CostBreakdown(
            labor=cb.get("labor", 0),
            materials=cb.get("materials", 0),
            contingency=cb.get("contingency", 0)
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> "Bid":
        """Create Bid from JSON dict. Handles optional fields gracefully."""
        # Parse delivery history (optional)
        delivery_data = data.get("delivery_history", {})
        delivery_history = DeliveryHistory(
            on_time_percentage=delivery_data.get("on_time_percentage", 0.8),
            on_budget_percentage=delivery_data.get("on_budget_percentage", 0.8),
            known_delays=delivery_data.get("known_delays", [])
        )
        
        # Parse legal compliance (optional)
        legal_data = data.get("legal_and_compliance", {})
        legal_compliance = LegalCompliance(
            open_litigation=legal_data.get("open_litigation", False),
            safety_violations_last_5_years=legal_data.get("safety_violations_last_5_years", 0)
        )
        
        # Parse bid metadata (optional)
        meta_data = data.get("bid_metadata", {})
        bid_metadata = BidMetadata(
            submission_channel=meta_data.get("submission_channel", "unknown"),
            submission_timestamp=meta_data.get("submission_timestamp", ""),
            bid_revision=meta_data.get("bid_revision", 1)
        )
        
        return cls(
            bid_id=data["bid_id"],
            company_name=data["company_name"],
            cost=data["cost"],
            timeline=Timeline(
                estimated_months=data["timeline"]["estimated_months"],
                confidence_level=data["timeline"]["confidence_level"],
                critical_path_risk=data["timeline"].get("critical_path_risk", "Medium")
            ),
            scope_coverage=ScopeCoverage(
                included=data["scope_coverage"]["included"],
                excluded=data["scope_coverage"].get("excluded", []),
                subcontracted=data["scope_coverage"].get("subcontracted", [])
            ),
            assumptions=data.get("assumptions", []),
            dependencies=data.get("dependencies", []),
            prior_similar_projects_count=data.get("prior_similar_projects_count", 0),
            delivery_history=delivery_history,
            legal_and_compliance=legal_compliance,
            bid_metadata=bid_metadata
        )


# ============ OUTPUT MODELS ============

@dataclass
class KeySignals:
    us_commercial_experience: Union[bool, str]  # True, False, or "Limited"
    project_scale_alignment: str  # "High", "Medium", "Low"
    recent_negative_news: bool


@dataclass
class CompanyInsight:
    sources: List[str]
    key_signals: KeySignals
    confidence_score: float
    
    def to_dict(self) -> dict:
        return {
            "sources": self.sources,
            "key_signals": {
                "us_commercial_experience": self.key_signals.us_commercial_experience,
                "project_scale_alignment": self.key_signals.project_scale_alignment,
                "recent_negative_news": self.key_signals.recent_negative_news
            },
            "confidence_score": self.confidence_score
        }


@dataclass
class ScoreBreakdown:
    cost_score: float
    timeline_score: float
    scope_fit_score: float
    risk_score: float
    external_reputation_score: float
    
    def to_dict(self) -> dict:
        return {
            "cost_score": self.cost_score,
            "timeline_score": self.timeline_score,
            "scope_fit_score": self.scope_fit_score,
            "risk_score": self.risk_score,
            "external_reputation_score": self.external_reputation_score
        }


@dataclass
class BidScore:
    bid_id: str
    company_name: str
    score_breakdown: ScoreBreakdown
    final_weighted_score: float
    flags: List[str]
    
    def to_dict(self) -> dict:
        return {
            "bid_id": self.bid_id,
            "company_name": self.company_name,
            "score_breakdown": self.score_breakdown.to_dict(),
            "final_weighted_score": self.final_weighted_score,
            "flags": self.flags
        }


@dataclass
class FinalRecommendation:
    bid_id: str
    company_name: str
    confidence: float
    decision_rationale: List[str]
    
    def to_dict(self) -> dict:
        return {
            "bid_id": self.bid_id,
            "company_name": self.company_name,
            "confidence": self.confidence,
            "decision_rationale": self.decision_rationale
        }


@dataclass
class RAGTrace:
    vector_store: str
    embedding_model: str
    documents_retrieved_per_company: int
    retrieval_confidence_threshold: float
    
    def to_dict(self) -> dict:
        return {
            "vector_store": self.vector_store,
            "embedding_model": self.embedding_model,
            "documents_retrieved_per_company": self.documents_retrieved_per_company,
            "retrieval_confidence_threshold": self.retrieval_confidence_threshold
        }


@dataclass
class EvaluationResult:
    """Full evaluation output model."""
    external_company_insights: Dict[str, CompanyInsight]
    retrieved_context_used: bool
    rag_trace: RAGTrace
    bid_scores: List[BidScore]
    ranked_recommendations: List[str]
    final_recommendation: FinalRecommendation
    explanation: str
    
    def to_dict(self) -> dict:
        return {
            "external_company_insights": {
                k: v.to_dict() for k, v in self.external_company_insights.items()
            },
            "retrieved_context_used": self.retrieved_context_used,
            "rag_trace": self.rag_trace.to_dict(),
            "bid_scores": [s.to_dict() for s in self.bid_scores],
            "ranked_recommendations": self.ranked_recommendations,
            "final_recommendation": self.final_recommendation.to_dict(),
            "explanation": self.explanation
        }

