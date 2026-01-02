# services/scorer.py
"""Scoring service for bid evaluation with historical learning."""

from typing import List, Dict, Optional
from models import Bid, BidScore, ScoreBreakdown, CompanyInsight
from config import WEIGHTS, MANDATORY_SCOPE, TARGET_MONTHS, CONFIDENCE_THRESHOLD

# Try to import historical RAG
try:
    from services.historical_rag import get_historical_rag, HistoricalInsight
    HISTORICAL_RAG_AVAILABLE = True
except ImportError:
    HISTORICAL_RAG_AVAILABLE = False
    HistoricalInsight = None


class Scorer:
    """Score bids based on multiple criteria with historical learning."""
    
    def __init__(self, use_historical: bool = True):
        self.use_historical = use_historical and HISTORICAL_RAG_AVAILABLE
        self._historical_rag = None
        
        if self.use_historical:
            self._historical_rag = get_historical_rag()
    
    def score(
        self, 
        bid: Bid, 
        all_bids: List[Bid], 
        insight: CompanyInsight,
        historical_insight: Optional[HistoricalInsight] = None
    ) -> BidScore:
        """Score a single bid against all criteria."""
        
        all_costs = [b.total_cost for b in all_bids]
        
        # Get historical insight if not provided
        if historical_insight is None and self._historical_rag:
            scope_items = bid.scope_coverage.included + bid.scope_coverage.subcontracted
            historical_insight = self._historical_rag.get_historical_insight(
                bid.company_name, 
                scope_items
            )
        
        # Calculate individual scores
        cost_score = self._score_cost(bid.total_cost, all_costs, historical_insight)
        timeline_score = self._score_timeline(bid, historical_insight)
        scope_score = self._score_scope(bid)
        risk_score = self._score_risk(bid, historical_insight)
        reputation_score = self._score_reputation(insight, historical_insight)
        
        # Calculate weighted final score
        final = (
            cost_score * WEIGHTS["cost"] +
            timeline_score * WEIGHTS["timeline"] +
            scope_score * WEIGHTS["scope"] +
            risk_score * WEIGHTS["risk"] +
            reputation_score * WEIGHTS["reputation"]
        )
        
        # Apply historical adjustment
        if historical_insight and historical_insight.confidence > 0.3:
            final = self._apply_historical_adjustment(final, historical_insight)
        
        # Generate flags
        flags = self._generate_flags(bid, insight, scope_score, historical_insight)
        
        return BidScore(
            bid_id=bid.bid_id,
            company_name=bid.company_name,
            score_breakdown=ScoreBreakdown(
                cost_score=round(cost_score, 2),
                timeline_score=round(timeline_score, 2),
                scope_fit_score=round(scope_score, 2),
                risk_score=round(risk_score, 2),
                external_reputation_score=round(reputation_score, 2)
            ),
            final_weighted_score=round(final, 2),
            flags=flags
        )
    
    def _score_cost(
        self, 
        cost: float, 
        all_costs: List[float],
        historical: Optional[HistoricalInsight] = None
    ) -> float:
        """Score cost - lower is better, with historical benchmark."""
        if not all_costs or len(all_costs) == 1:
            return 0.8
        
        min_c, max_c = min(all_costs), max(all_costs)
        if max_c == min_c:
            return 1.0
        
        # Base score: normalized against current bids
        normalized = 1 - (cost - min_c) / (max_c - min_c)
        base_score = 0.5 + (normalized * 0.5)
        
        # Historical benchmark adjustment
        if historical and historical.pricing_benchmark.get("sample_size", 0) >= 3:
            benchmark = historical.pricing_benchmark
            avg_price = benchmark.get("avg", 0)
            
            if avg_price > 0:
                # Penalize if significantly above historical average
                if cost > avg_price * 1.2:
                    base_score *= 0.9  # 10% penalty
                # Bonus if below historical average
                elif cost < avg_price * 0.9:
                    base_score = min(1.0, base_score * 1.05)  # 5% bonus
        
        return base_score
    
    def _score_timeline(
        self, 
        bid: Bid,
        historical: Optional[HistoricalInsight] = None
    ) -> float:
        """Score timeline based on duration, confidence, and history."""
        months = bid.timeline.estimated_months
        confidence = bid.timeline.confidence_level
        
        # Base score: how close to target
        if months <= TARGET_MONTHS:
            duration_score = 1.0
        else:
            overage = months - TARGET_MONTHS
            duration_score = max(0.4, 1.0 - (overage * 0.15))
        
        # Factor in confidence level
        base_score = duration_score * confidence
        
        # Historical adjustment: if company has history of timeline issues
        if historical and historical.risk_patterns:
            if "History of timeline overruns" in historical.risk_patterns:
                base_score *= 0.85  # 15% penalty for past timeline issues
        
        return base_score
    
    def _score_scope(self, bid: Bid) -> float:
        """Score scope coverage against mandatory items."""
        included = set(bid.scope_coverage.included)
        subcontracted = set(bid.scope_coverage.subcontracted)
        covered = included | subcontracted
        mandatory = set(MANDATORY_SCOPE)
        
        if not mandatory:
            return 1.0
        
        coverage = len(covered & mandatory) / len(mandatory)
        
        # Penalty for subcontracting
        subcontracted_mandatory = subcontracted & mandatory
        if subcontracted_mandatory:
            coverage *= 0.85
        
        return coverage
    
    def _score_risk(
        self, 
        bid: Bid,
        historical: Optional[HistoricalInsight] = None
    ) -> float:
        """Score risk based on delivery history, compliance, and historical patterns."""
        history = bid.delivery_history
        legal = bid.legal_and_compliance
        timeline = bid.timeline
        
        # Delivery performance (40% weight)
        delivery = (history.on_time_percentage + history.on_budget_percentage) / 2
        
        # Critical path risk factor (25% weight)
        risk_factors = {"Low": 1.0, "Medium": 0.85, "High": 0.65}
        path_risk = risk_factors.get(timeline.critical_path_risk, 0.75)
        
        # Legal/compliance factor (20% weight)
        legal_score = 1.0
        if legal.open_litigation:
            legal_score -= 0.3
        if legal.safety_violations_last_5_years > 0:
            legal_score -= min(legal.safety_violations_last_5_years * 0.1, 0.3)
        legal_score = max(legal_score, 0.4)
        
        # Known delays factor (15% weight)
        delay_score = 1.0 if not history.known_delays else max(0.6, 1.0 - len(history.known_delays) * 0.15)
        
        base_score = (
            delivery * 0.40 +
            path_risk * 0.25 +
            legal_score * 0.20 +
            delay_score * 0.15
        )
        
        # Historical risk pattern adjustment
        if historical and historical.risk_patterns:
            risk_penalty = len(historical.risk_patterns) * 0.03  # 3% per pattern
            base_score = max(0.4, base_score - risk_penalty)
        
        return base_score
    
    def _score_reputation(
        self, 
        insight: CompanyInsight,
        historical: Optional[HistoricalInsight] = None
    ) -> float:
        """Score external reputation with historical win rate."""
        signals = insight.key_signals
        
        # US experience (40% weight - reduced to make room for historical)
        if signals.us_commercial_experience is True:
            us_score = 1.0
        elif signals.us_commercial_experience == "Limited":
            us_score = 0.6
        else:
            us_score = 0.3
        
        # Scale alignment (25% weight)
        scale_scores = {"High": 1.0, "Medium": 0.7, "Low": 0.4}
        scale_score = scale_scores.get(signals.project_scale_alignment, 0.5)
        
        # Negative news penalty (15% weight)
        news_score = 0.4 if signals.recent_negative_news else 1.0
        
        # Historical win rate (20% weight - new!)
        historical_score = 0.5  # Default neutral
        if historical and historical.confidence > 0.3:
            # Win rate contributes positively
            historical_score = 0.3 + (historical.historical_win_rate * 0.7)
        
        base = (
            us_score * 0.40 + 
            scale_score * 0.25 + 
            news_score * 0.15 +
            historical_score * 0.20
        )
        
        # Adjust by insight confidence
        return base * (0.7 + insight.confidence_score * 0.3)
    
    def _apply_historical_adjustment(
        self, 
        score: float, 
        historical: HistoricalInsight
    ) -> float:
        """Apply final adjustment based on historical performance."""
        # Strong historical performers get a small boost
        if historical.historical_win_rate > 0.5 and historical.avg_score > 0.75:
            score = min(1.0, score * 1.03)  # 3% boost
        
        # Poor historical performers get a small penalty
        elif historical.historical_win_rate < 0.2 and historical.avg_score < 0.6:
            score = max(0.3, score * 0.97)  # 3% penalty
        
        return score
    
    def _generate_flags(
        self, 
        bid: Bid, 
        insight: CompanyInsight, 
        scope_score: float,
        historical: Optional[HistoricalInsight] = None
    ) -> List[str]:
        """Generate warning flags for the bid."""
        flags = []
        
        # f1: Subcontracted critical work
        if bid.scope_coverage.subcontracted:
            flags.append("f1")
        
        # f2: Timeline exceeds target
        if bid.timeline.estimated_months > TARGET_MONTHS:
            flags.append("f2")
        
        # f3: Low confidence
        if bid.timeline.confidence_level < CONFIDENCE_THRESHOLD:
            flags.append("f3")
        
        # f4: Scope gap
        if scope_score < 1.0:
            flags.append("f4")
        
        # x1: No US commercial experience
        if insight.key_signals.us_commercial_experience is False:
            flags.append("x1")
        
        # h1: Historical risk patterns (NEW)
        if historical and len(historical.risk_patterns) >= 2:
            flags.append("h1")
        
        # h2: Low historical win rate (NEW)
        if historical and historical.confidence > 0.3:
            if historical.historical_win_rate < 0.2 and historical.avg_score < 0.6:
                flags.append("h2")
        
        return flags
