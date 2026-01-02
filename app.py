# app.py
"""Streamlit UI for Bid Evaluation Agent."""

import streamlit as st
import json
from typing import List

from models import Bid
from controller import Controller

# Page config
st.set_page_config(
    page_title="Bid Evaluation System",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: #0a0a0f;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main-header {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        border-bottom: 1px solid #1e1e2e;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #a0a0b0;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: #12121a;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        padding: 1.25rem;
    }
    
    .winner-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
        border: 1px solid #2a2a3e;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .winner-label {
        color: #6366f1;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }
    
    .winner-name {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .winner-score {
        color: #a0a0b0;
        font-size: 1rem;
    }
    
    .flag-tag {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.5rem;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .status-positive {
        color: #22c55e;
    }
    
    .status-warning {
        color: #f59e0b;
    }
    
    .status-negative {
        color: #ef4444;
    }
    
    div[data-testid="stSidebar"] {
        background: #0a0a0f;
        border-right: 1px solid #1e1e2e;
    }
    
    .stButton > button {
        background: #6366f1;
        color: white;
        border: none;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: #4f46e5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize controller
@st.cache_resource
def get_controller():
    return Controller()

controller = get_controller()

# Sidebar navigation
st.sidebar.markdown("### Bid Evaluation System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Upload", "Results", "History", "Observability"],
    label_visibility="collapsed"
)

# Show stats in sidebar
try:
    stats = controller.get_stats()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Statistics**")
    st.sidebar.metric("Total Evaluations", stats["evaluations"]["total_evaluations"])
    st.sidebar.metric("Cached Companies", stats["cached_companies"])
except Exception:
    pass

# Flag definitions
FLAG_DEFINITIONS = {
    "f1": ("Subcontracted Work", "Critical scope items are subcontracted to third parties"),
    "f2": ("Timeline Overrun", "Estimated completion exceeds target duration"),
    "f3": ("Low Confidence", "Contractor confidence level below acceptable threshold"),
    "f4": ("Scope Gap", "Mandatory scope items are not fully covered"),
    "x1": ("No US Experience", "No verified US commercial project experience"),
    "h1": ("Historical Risk", "Multiple risk patterns identified from past evaluations"),
    "h2": ("Poor Track Record", "Low historical win rate and average scores")
}


def parse_bids(data: List[dict]) -> List[Bid]:
    """Parse JSON data into Bid objects."""
    return [Bid.from_dict(b) for b in data]


# ==================== UPLOAD PAGE ====================
if page == "Upload":
    st.markdown('<h1 class="main-header">Upload</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="section-header">Data Input</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select JSON file containing bid submissions",
            type=["json", "txt"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                bids = parse_bids(data)
                
                st.success(f"Loaded {len(bids)} bid submissions")
                
                st.markdown('<p class="section-header">Bid Summary</p>', unsafe_allow_html=True)
                
                for bid in bids:
                    with st.expander(f"{bid.company_name} ({bid.bid_id})"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Cost", f"${bid.total_cost:,.0f}")
                        c2.metric("Timeline", f"{bid.timeline.estimated_months} months")
                        c3.metric("Confidence", f"{bid.timeline.confidence_level:.0%}")
                        
                        st.markdown(f"**Scope Coverage:** {', '.join(bid.scope_coverage.included)}")
                        if bid.scope_coverage.subcontracted:
                            st.markdown(f"**Subcontracted:** {', '.join(bid.scope_coverage.subcontracted)}")
                
                st.markdown("---")
                
                # Check for cached result
                from services.memory import get_memory_store
                memory = get_memory_store()
                cached = memory.find_similar_evaluation(data)
                
                if cached:
                    st.info(f"Similar evaluation found from {cached['timestamp'][:10]}. You can use cached result or run fresh.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Use Cached Result", use_container_width=True):
                            st.session_state["result"] = cached["result"]
                            st.session_state["bids"] = data
                            st.success("Cached result loaded. Navigate to Results.")
                    with col2:
                        if st.button("Run Fresh Evaluation", type="primary", use_container_width=True):
                            with st.spinner("Processing bid evaluation..."):
                                result = controller.evaluate(bids, use_cached_result=False)
                                st.session_state["result"] = result.to_dict()
                                st.session_state["bids"] = data
                            st.success("Evaluation complete. Navigate to Results.")
                else:
                    if st.button("Run Evaluation", type="primary", use_container_width=True):
                        with st.spinner("Processing bid evaluation..."):
                            result = controller.evaluate(bids)
                            st.session_state["result"] = result.to_dict()
                            st.session_state["bids"] = data
                        st.success("Evaluation complete. Navigate to Results.")
                    
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")
            except KeyError as e:
                st.error(f"Missing required field: {e}")
            except Exception as e:
                st.error(f"Error processing bids: {str(e)}")
    
    with col2:
        st.markdown('<p class="section-header">Input Schema</p>', unsafe_allow_html=True)
        st.code('''{
  "bid_id": "string",
  "company_name": "string",
  "cost": {
    "total_usd": number,
    "currency": "USD",
    "cost_breakdown": {...}
  },
  "timeline": {
    "estimated_months": number,
    "confidence_level": float,
    "critical_path_risk": "Low|Medium|High"
  },
  "scope_coverage": {
    "included": [...],
    "excluded": [...],
    "subcontracted": [...]
  },
  ...
}''', language="json")


# ==================== RESULTS PAGE ====================
elif page == "Results":
    st.markdown('<h1 class="main-header">Evaluation Results</h1>', unsafe_allow_html=True)
    
    if "result" not in st.session_state:
        st.info("No evaluation data available. Please upload and evaluate bids first.")
        st.stop()
    
    result = st.session_state["result"]
    winner = result["final_recommendation"]
    
    # Winner announcement
    st.markdown(f"""
    <div class="winner-banner">
        <div class="winner-label">Recommended Selection</div>
        <div class="winner-name">{winner['company_name']}</div>
        <div class="winner-score">Confidence Score: {winner['confidence']:.0%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Decision rationale
    st.markdown('<p class="section-header">Decision Rationale</p>', unsafe_allow_html=True)
    cols = st.columns(len(winner["decision_rationale"]))
    for i, rationale in enumerate(winner["decision_rationale"]):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; height: 80px; 
                        display: flex; align-items: center; justify-content: center;">
                {rationale}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed scores
    st.markdown('<p class="section-header">Bid Score Analysis</p>', unsafe_allow_html=True)
    
    scores = sorted(result["bid_scores"], key=lambda x: x["final_weighted_score"], reverse=True)
    
    for rank, score in enumerate(scores, 1):
        is_winner = score["bid_id"] == winner["bid_id"]
        sb = score["score_breakdown"]
        
        with st.expander(
            f"#{rank} {score['company_name']} — Score: {score['final_weighted_score']:.2f}" + 
            (" [SELECTED]" if is_winner else ""),
            expanded=is_winner
        ):
            # Score breakdown
            cols = st.columns(5)
            cols[0].metric("Cost", f"{sb['cost_score']:.2f}")
            cols[1].metric("Timeline", f"{sb['timeline_score']:.2f}")
            cols[2].metric("Scope Fit", f"{sb['scope_fit_score']:.2f}")
            cols[3].metric("Risk", f"{sb['risk_score']:.2f}")
            cols[4].metric("Reputation", f"{sb['external_reputation_score']:.2f}")
            
            # Flags
            if score["flags"]:
                st.markdown("**Risk Flags:**")
                for flag in score["flags"]:
                    if flag in FLAG_DEFINITIONS:
                        name, desc = FLAG_DEFINITIONS[flag]
                        st.markdown(f"- **{flag.upper()}** — {name}: {desc}")
            
            # External insights
            company = score["company_name"]
            if company in result["external_company_insights"]:
                insight = result["external_company_insights"][company]
                
                st.markdown("**External Validation:**")
                c1, c2, c3 = st.columns(3)
                
                us_exp = insight["key_signals"]["us_commercial_experience"]
                if us_exp is True:
                    us_display = "Verified"
                    us_class = "status-positive"
                elif us_exp == "Limited":
                    us_display = "Limited"
                    us_class = "status-warning"
                else:
                    us_display = "Not Verified"
                    us_class = "status-negative"
                
                c1.markdown(f"US Experience: <span class='{us_class}'>{us_display}</span>", unsafe_allow_html=True)
                c2.markdown(f"Scale Alignment: {insight['key_signals']['project_scale_alignment']}")
                c3.markdown(f"Data Confidence: {insight['confidence_score']:.0%}")
                
                if insight["sources"]:
                    st.markdown("**Sources:**")
                    for src in insight["sources"][:3]:
                        st.markdown(f"- {src}")
            
            # Historical insights
            try:
                historical_insights = controller.get_historical_insights()
                if company in historical_insights:
                    hist = historical_insights[company]
                    st.markdown("**Historical Performance:**")
                    h1, h2, h3 = st.columns(3)
                    h1.metric("Win Rate", f"{hist.get('win_rate', 0):.0%}")
                    h2.metric("Avg Score", f"{hist.get('avg_score', 0):.2f}")
                    h3.metric("Data Confidence", f"{hist.get('confidence', 0):.0%}")
                    
                    if hist.get("risk_patterns"):
                        st.markdown("**Risk Patterns from History:**")
                        for pattern in hist["risk_patterns"]:
                            st.markdown(f"- {pattern}")
            except Exception:
                pass  # Historical RAG not available
    
    st.markdown("---")
    
    # Explanation
    st.markdown('<p class="section-header">Analysis Summary</p>', unsafe_allow_html=True)
    st.markdown(result["explanation"])
    
    # Feedback section
    st.markdown("---")
    st.markdown('<p class="section-header">Feedback</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        feedback_rating = st.slider(
            "Rate this recommendation", 
            1, 5, 3, 
            help="1 = Poor, 5 = Excellent"
        )
        feedback_comment = st.text_input("Optional comment")
    
    with col2:
        if st.button("Submit Feedback", use_container_width=True):
            controller.submit_feedback(rating=feedback_rating, comment=feedback_comment)
            st.success("Feedback recorded")
    
    # Export
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Export Results (JSON)",
            json.dumps(result, indent=2),
            "bid_evaluation_results.json",
            "application/json",
            use_container_width=True
        )
    with col2:
        eval_id = controller.get_last_evaluation_id()
        if eval_id:
            st.markdown(f"**Evaluation ID:** `{eval_id[:12]}...`")


# ==================== HISTORY PAGE ====================
elif page == "History":
    st.markdown('<h1 class="main-header">Evaluation History</h1>', unsafe_allow_html=True)
    
    # Get history
    history = controller.get_evaluation_history(limit=20)
    stats = controller.get_stats()
    
    # Stats overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Evaluations", stats["evaluations"]["total_evaluations"])
    col2.metric("Avg Confidence", f"{stats['evaluations']['average_winner_score']:.0%}")
    col3.metric("Cached Companies", stats["cached_companies"])
    col4.metric("Total Feedback", stats["feedback"]["total_feedback"])
    
    st.markdown("---")
    
    # Winner distribution
    if stats["evaluations"]["winner_distribution"]:
        st.markdown('<p class="section-header">Winner Distribution</p>', unsafe_allow_html=True)
        
        winner_dist = stats["evaluations"]["winner_distribution"]
        max_wins = max(winner_dist.values()) if winner_dist else 1
        
        for company, wins in sorted(winner_dist.items(), key=lambda x: x[1], reverse=True):
            pct = (wins / max_wins) * 100
            st.markdown(f"""
            <div style="margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: #fafafa; font-size: 0.875rem;">{company}</span>
                    <span style="color: #a1a1aa; font-size: 0.875rem;">{wins} wins</span>
                </div>
                <div style="background: #27272a; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: #6366f1; height: 100%; width: {pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Historical RAG stats
    try:
        hist_stats = controller.get_historical_stats()
        if hist_stats.get("enabled"):
            st.markdown('<p class="section-header">Historical Learning (RAG)</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Indexed Evaluations", hist_stats.get("total_evaluations", 0))
            col2.metric("Company Profiles", hist_stats.get("total_companies", 0))
            col3.metric("Bid Patterns", hist_stats.get("total_bid_patterns", 0))
            
            # Top performers
            top_performers = controller.get_top_performers(limit=5)
            if top_performers:
                st.markdown("**Top Performing Companies:**")
                for i, perf in enumerate(top_performers, 1):
                    st.markdown(
                        f"{i}. **{perf.get('company_name')}** — "
                        f"Win Rate: {perf.get('win_rate', 0):.0%}, "
                        f"Avg Score: {perf.get('avg_score', 0):.2f}, "
                        f"Evaluations: {perf.get('total_evaluations', 0)}"
                    )
            
            st.markdown("---")
    except Exception:
        pass  # Historical RAG not available
    
    # Recent evaluations
    st.markdown('<p class="section-header">Recent Evaluations</p>', unsafe_allow_html=True)
    
    if history:
        for eval_record in history:
            with st.expander(
                f"{eval_record['timestamp'][:10]} — {eval_record['winner_company']} "
                f"(Confidence: {eval_record['winner_score']:.0%})"
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Winner", eval_record["winner_company"])
                col2.metric("Bids Evaluated", eval_record["bid_count"])
                col3.metric("Confidence", f"{eval_record['winner_score']:.0%}")
                
                st.markdown(f"**Evaluation ID:** `{eval_record['id']}`")
                st.markdown(f"**Timestamp:** {eval_record['timestamp']}")
                
                # Load full evaluation button
                if st.button(f"Load Evaluation", key=f"load_{eval_record['id']}"):
                    full_eval = controller.get_evaluation_by_id(eval_record['id'])
                    if full_eval:
                        st.session_state["result"] = full_eval["result"]
                        st.session_state["loaded_eval_id"] = eval_record['id']
                        st.success("Evaluation loaded. Go to Results tab to view.")
                        st.rerun()
    else:
        st.info("No evaluation history yet. Run your first evaluation.")
    
    st.markdown("---")
    
    # Feedback section
    st.markdown('<p class="section-header">Submit Feedback</p>', unsafe_allow_html=True)
    
    if controller.get_last_evaluation_id():
        with st.form("feedback_form"):
            rating = st.slider("Rate the recommendation quality", 1, 5, 3)
            comment = st.text_area("Comments (optional)")
            correct_winner = st.text_input("Correct winner ID (if different)")
            
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                controller.submit_feedback(
                    rating=rating,
                    comment=comment if comment else None,
                    correct_winner_id=correct_winner if correct_winner else None
                )
                st.success("Feedback submitted. Thank you!")
    else:
        st.info("Run an evaluation first to submit feedback.")
    
    # Export section
    st.markdown("---")
    st.markdown('<p class="section-header">Data Export</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export All Data"):
            from services.memory import get_memory_store
            memory = get_memory_store()
            export_data = memory.export_all()
            st.download_button(
                "Download Export (JSON)",
                json.dumps(export_data, indent=2, default=str),
                "bid_eval_export.json",
                "application/json"
            )
    
    with col2:
        if st.button("Clear Old Data (30+ days)"):
            from services.memory import get_memory_store
            memory = get_memory_store()
            memory.clear_old_data(days=30)
            st.success("Old data cleared.")


# ==================== OBSERVABILITY PAGE ====================
elif page == "Observability":
    # LangSmith-style CSS
    st.markdown("""
    <style>
        .trace-header {
            background: #18181b;
            border: 1px solid #27272a;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }
        .trace-title {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .trace-name {
            font-size: 1.25rem;
            font-weight: 600;
            color: #fafafa;
        }
        .status-badge {
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .status-badge-error {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .trace-id {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #71717a;
        }
        .trace-meta {
            display: flex;
            gap: 2rem;
            margin-top: 0.75rem;
        }
        .meta-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .meta-label {
            font-size: 0.7rem;
            color: #71717a;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .meta-value {
            font-size: 0.875rem;
            color: #fafafa;
            font-weight: 500;
        }
        .span-row {
            background: #18181b;
            border: 1px solid #27272a;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            overflow: hidden;
        }
        .span-header {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            gap: 0.75rem;
            cursor: pointer;
        }
        .span-header:hover {
            background: #1f1f23;
        }
        .span-indent {
            width: 20px;
            border-left: 2px solid #3f3f46;
            height: 100%;
        }
        .span-icon {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            flex-shrink: 0;
        }
        .span-icon-chain { background: #3b82f6; color: white; }
        .span-icon-llm { background: #8b5cf6; color: white; }
        .span-icon-tool { background: #f59e0b; color: white; }
        .span-icon-retriever { background: #10b981; color: white; }
        .span-name {
            font-size: 0.875rem;
            color: #fafafa;
            font-weight: 500;
            flex-grow: 1;
        }
        .span-type {
            font-size: 0.7rem;
            color: #71717a;
            text-transform: uppercase;
            background: #27272a;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .span-duration {
            font-size: 0.8rem;
            color: #a1a1aa;
            font-family: 'IBM Plex Mono', monospace;
        }
        .duration-bar-container {
            width: 120px;
            height: 6px;
            background: #27272a;
            border-radius: 3px;
            overflow: hidden;
        }
        .duration-bar {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            border-radius: 3px;
        }
        .span-details {
            background: #0f0f12;
            border-top: 1px solid #27272a;
            padding: 1rem;
        }
        .detail-section {
            margin-bottom: 1rem;
        }
        .detail-label {
            font-size: 0.7rem;
            color: #71717a;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .detail-content {
            background: #18181b;
            border: 1px solid #27272a;
            border-radius: 4px;
            padding: 0.75rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #d4d4d8;
            overflow-x: auto;
        }
        .io-tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .io-tab {
            padding: 0.4rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            border: 1px solid #27272a;
            background: transparent;
            color: #a1a1aa;
        }
        .io-tab-active {
            background: #27272a;
            color: #fafafa;
        }
        .token-stat {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            background: #27272a;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            color: #a1a1aa;
            margin-right: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if "result" not in st.session_state:
        st.info("No evaluation data available. Run an evaluation first.")
        st.stop()
    
    result = st.session_state["result"]
    events = controller.get_events()
    summary = controller.get_summary()
    
    # Trace Header (LangSmith style)
    trace_id = summary.get("trace_ids", ["N/A"])[0] if summary.get("trace_ids") else "N/A"
    total_duration = summary.get("total_duration_ms", 0)
    
    st.markdown(f"""
    <div class="trace-header">
        <div class="trace-title">
            <span class="trace-name">bid_evaluation</span>
            <span class="status-badge">Success</span>
        </div>
        <div class="trace-id">Trace ID: {trace_id}</div>
        <div class="trace-meta">
            <div class="meta-item">
                <span class="meta-label">Latency</span>
                <span class="meta-value">{total_duration:.0f}ms</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">LLM Calls</span>
                <span class="meta-value">{summary.get('llm_calls', 0)}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Tool Calls</span>
                <span class="meta-value">{summary.get('search_count', 0)}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Total Steps</span>
                <span class="meta-value">{summary.get('total_events', 0)}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different views
    trace_tab, metrics_tab, raw_tab = st.tabs(["Trace", "Metrics", "Raw Events"])
    
    with trace_tab:
        st.markdown('<p class="section-header">Run Trace</p>', unsafe_allow_html=True)
        
        if events:
            # Calculate max duration for scaling bars
            max_duration = max([e.get("duration_ms", 0) for e in events] + [1])
            
            # Group events into hierarchy
            def get_span_icon(event_type):
                icons = {
                    "request": ("▶", "chain"),
                    "search": ("⚡", "tool"),
                    "score": ("◆", "retriever"),
                    "llm": ("◈", "llm"),
                    "complete": ("✓", "chain")
                }
                return icons.get(event_type, ("•", "chain"))
            
            def get_indent_level(node):
                if node in ["input", "output", "batch_complete"]:
                    return 0
                elif node in ["start", "result"]:
                    return 1
                elif node == "complete":
                    return 1
                return 0
            
            for i, event in enumerate(events):
                event_type = event["type"]
                node = event["node"]
                duration = event.get("duration_ms", 0)
                duration_pct = min((duration / max_duration) * 100, 100) if max_duration > 0 else 0
                icon, icon_class = get_span_icon(event_type)
                indent = get_indent_level(node)
                
                # Create span row
                span_name = f"{event_type}.{node}"
                
                with st.expander(f"{span_name}", expanded=False):
                    # Header row with custom styling
                    col1, col2, col3, col4 = st.columns([3, 1, 2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span class="span-icon span-icon-{icon_class}">{icon}</span>
                            <span style="color: #fafafa; font-weight: 500;">{event_type}</span>
                            <span style="color: #71717a;">→</span>
                            <span style="color: #a1a1aa;">{node}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'<span class="span-type">{event_type.upper()}</span>', unsafe_allow_html=True)
                    
                    with col3:
                        if duration > 0:
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <div class="duration-bar-container">
                                    <div class="duration-bar" style="width: {duration_pct}%;"></div>
                                </div>
                                <span class="span-duration">{duration:.0f}ms</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col4:
                        if event.get("span_id"):
                            st.markdown(f'<span class="trace-id">{event["span_id"][:8]}...</span>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Input/Output sections
                    io_col1, io_col2 = st.columns(2)
                    
                    with io_col1:
                        st.markdown('<div class="detail-label">INPUT</div>', unsafe_allow_html=True)
                        st.json(event.get("data", {}))
                    
                    with io_col2:
                        st.markdown('<div class="detail-label">METADATA</div>', unsafe_allow_html=True)
                        metadata = {
                            "timestamp": event.get("timestamp", ""),
                            "trace_id": event.get("trace_id", "N/A"),
                            "span_id": event.get("span_id", "N/A"),
                            "duration_ms": event.get("duration_ms", 0)
                        }
                        st.json(metadata)
        else:
            st.info("No trace data available")
    
    with metrics_tab:
        st.markdown('<p class="section-header">Performance Metrics</p>', unsafe_allow_html=True)
        
        # Latency breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Latency Breakdown**")
            
            # Calculate latencies by type
            search_events = [e for e in events if e["type"] == "search" and e.get("duration_ms")]
            llm_events = [e for e in events if e["type"] == "llm" and e.get("duration_ms")]
            score_events = [e for e in events if e["type"] == "score"]
            
            search_latency = sum(e.get("duration_ms", 0) for e in search_events)
            llm_latency = sum(e.get("duration_ms", 0) for e in llm_events)
            
            st.markdown(f"""
            <div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 1rem;">
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #a1a1aa; font-size: 0.8rem;">Search Operations</span>
                        <span style="color: #fafafa; font-size: 0.8rem;">{search_latency:.0f}ms</span>
                    </div>
                    <div style="background: #27272a; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: #f59e0b; height: 100%; width: {min(search_latency/max(total_duration,1)*100, 100):.0f}%;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #a1a1aa; font-size: 0.8rem;">LLM Inference</span>
                        <span style="color: #fafafa; font-size: 0.8rem;">{llm_latency:.0f}ms</span>
                    </div>
                    <div style="background: #27272a; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: #8b5cf6; height: 100%; width: {min(llm_latency/max(total_duration,1)*100, 100):.0f}%;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #a1a1aa; font-size: 0.8rem;">Total</span>
                        <span style="color: #fafafa; font-size: 0.8rem;">{total_duration:.0f}ms</span>
                    </div>
                    <div style="background: #27272a; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%; width: 100%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Run Statistics**")
            
            st.markdown(f"""
            <div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 1rem;">
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #27272a;">
                    <span style="color: #71717a; font-size: 0.8rem;">Total Events</span>
                    <span style="color: #fafafa; font-size: 0.8rem; font-weight: 500;">{summary.get('total_events', 0)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #27272a;">
                    <span style="color: #71717a; font-size: 0.8rem;">Search Calls</span>
                    <span style="color: #fafafa; font-size: 0.8rem; font-weight: 500;">{summary.get('search_count', 0)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #27272a;">
                    <span style="color: #71717a; font-size: 0.8rem;">Score Calculations</span>
                    <span style="color: #fafafa; font-size: 0.8rem; font-weight: 500;">{summary.get('score_count', 0)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                    <span style="color: #71717a; font-size: 0.8rem;">LLM Calls</span>
                    <span style="color: #fafafa; font-size: 0.8rem; font-weight: 500;">{summary.get('llm_calls', 0)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # RAG Configuration
        st.markdown('<p class="section-header">RAG Configuration</p>', unsafe_allow_html=True)
        rag = result["rag_trace"]
        
        st.markdown(f"""
        <div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 1rem;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                <div>
                    <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Vector Store</div>
                    <div style="color: #fafafa; font-size: 0.875rem;">{rag["vector_store"] or "Not Configured"}</div>
                </div>
                <div>
                    <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Embedding Model</div>
                    <div style="color: #fafafa; font-size: 0.875rem;">{rag["embedding_model"]}</div>
                </div>
                <div>
                    <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Docs/Company</div>
                    <div style="color: #fafafa; font-size: 0.875rem;">{rag["documents_retrieved_per_company"]}</div>
                </div>
                <div>
                    <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Threshold</div>
                    <div style="color: #fafafa; font-size: 0.875rem;">{rag["retrieval_confidence_threshold"]:.0%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with raw_tab:
        st.markdown('<p class="section-header">Raw Event Data</p>', unsafe_allow_html=True)
        
        if events:
            st.json(events)
            
            st.download_button(
                "Export Events (JSON)",
                json.dumps(events, indent=2),
                "evaluation_events.json",
                "application/json"
            )
        else:
            st.info("No events recorded")


# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
if "result" in st.session_state:
    st.sidebar.markdown("Evaluation: Active")
else:
    st.sidebar.markdown("Evaluation: Pending")
