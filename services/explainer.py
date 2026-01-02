# services/explainer.py
"""LLM explanation service using Groq."""

from typing import List
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from models import BidScore, FinalRecommendation, CompanyInsight


class Explainer:
    """Generate explanations using Groq LLM."""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Please configure it in .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL
    
    def explain(
        self, 
        winner: BidScore, 
        all_scores: List[BidScore],
        insights: dict
    ) -> str:
        """Generate natural language explanation for bid selection."""
        prompt = self._build_prompt(winner, all_scores, insights)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a senior construction bid evaluation analyst with expertise in "
                        "commercial renovation projects. Analyze bid data systematically, considering "
                        "cost, risk, timeline, scope coverage, and contractor reputation. "
                        "Provide clear, formal analysis suitable for executive decision-makers. "
                        "Be precise, objective, and data-driven in your assessments."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=65536  # Max for gpt-oss-120b
        )
        return response.choices[0].message.content
    
    def generate_rationale(
        self, 
        winner: BidScore, 
        insight: CompanyInsight
    ) -> List[str]:
        """Generate decision rationale points."""
        rationale = []
        sb = winner.score_breakdown
        
        # Scope assessment
        if sb.scope_fit_score >= 0.95:
            rationale.append("Complete scope coverage")
        elif sb.scope_fit_score >= 0.7:
            rationale.append("Adequate scope coverage")
        
        # US experience
        if insight.key_signals.us_commercial_experience is True:
            rationale.append("Verified US commercial experience")
        elif insight.key_signals.us_commercial_experience == "Limited":
            rationale.append("Partial US market presence")
        
        # Cost-risk balance
        if sb.cost_score >= 0.7 and sb.risk_score >= 0.7:
            rationale.append("Balanced cost-risk profile")
        elif sb.cost_score >= 0.85:
            rationale.append("Competitive cost structure")
        
        # External confidence
        if insight.confidence_score >= 0.85:
            rationale.append("High data confidence")
        elif insight.confidence_score >= 0.7:
            rationale.append("Validated external data")
        
        # Risk score
        if sb.risk_score >= 0.85:
            rationale.append("Strong delivery record")
        elif sb.risk_score >= 0.75:
            rationale.append("Reliable delivery history")
        
        # Timeline
        if sb.timeline_score >= 0.8:
            rationale.append("Schedule confidence")
        
        return rationale[:4]
    
    def _build_prompt(
        self, 
        winner: BidScore, 
        all_scores: List[BidScore],
        insights: dict
    ) -> str:
        """Build the LLM prompt."""
        sorted_scores = sorted(all_scores, key=lambda s: s.final_weighted_score, reverse=True)
        
        scores_summary = "\n".join([
            f"- {s.company_name}: {s.final_weighted_score} "
            f"(cost={s.score_breakdown.cost_score}, "
            f"timeline={s.score_breakdown.timeline_score}, "
            f"scope={s.score_breakdown.scope_fit_score}, "
            f"risk={s.score_breakdown.risk_score}, "
            f"reputation={s.score_breakdown.external_reputation_score})"
            + (f" [FLAGS: {', '.join(s.flags)}]" if s.flags else "")
            for s in sorted_scores
        ])
        
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else None
        
        return f"""Analyze this construction bid evaluation and provide a formal explanation for the selection of {winner.company_name}.

PROJECT CONTEXT:
- Commercial renovation in Downtown Los Angeles
- Target budget: $1.2M
- Target timeline: 6 months
- Priority: Risk minimization over cost savings
- Mandatory scope: Electrical, HVAC, Interior

SCORING RESULTS (ranked by final score):
{scores_summary}

SELECTED BID DETAILS:
- Company: {winner.company_name}
- Final Score: {winner.final_weighted_score}
- Cost Score: {winner.score_breakdown.cost_score}
- Timeline Score: {winner.score_breakdown.timeline_score}
- Scope Coverage: {winner.score_breakdown.scope_fit_score}
- Risk Score: {winner.score_breakdown.risk_score}
- Reputation: {winner.score_breakdown.external_reputation_score}
- Flags: {', '.join(winner.flags) if winner.flags else 'None'}

RUNNER-UP: {runner_up.company_name if runner_up else 'N/A'} (Score: {runner_up.final_weighted_score if runner_up else 'N/A'})

Provide a formal 2-3 paragraph analysis that:
1. States the selected contractor and justifies the selection
2. Highlights key strengths relative to project requirements
3. Addresses any identified risk flags and mitigation considerations
4. Explains the comparative advantage over other bidders

Use formal, professional language suitable for executive review."""
