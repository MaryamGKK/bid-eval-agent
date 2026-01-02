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

# Professional CSS with skeleton loading
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
    
    /* Skeleton Loading Animation */
    @keyframes skeleton-pulse {
        0% { opacity: 0.4; }
        50% { opacity: 0.7; }
        100% { opacity: 0.4; }
    }
    
    .skeleton {
        background: linear-gradient(90deg, #1e1e2e 0%, #2a2a3e 50%, #1e1e2e 100%);
        background-size: 200% 100%;
        animation: skeleton-pulse 1.5s ease-in-out infinite;
        border-radius: 4px;
    }
    
    .skeleton-text {
        height: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .skeleton-text-sm {
        height: 0.75rem;
        margin-bottom: 0.25rem;
    }
    
    .skeleton-title {
        height: 2rem;
        width: 60%;
        margin-bottom: 1rem;
    }
    
    .skeleton-card {
        background: #12121a;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .skeleton-metric {
        height: 3rem;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    .skeleton-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
        border: 1px solid #2a2a3e;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def show_skeleton_card():
    """Display a skeleton loading card."""
    st.markdown("""
    <div class="skeleton-card">
        <div class="skeleton skeleton-text" style="width: 40%;"></div>
        <div class="skeleton skeleton-metric"></div>
        <div class="skeleton skeleton-text-sm" style="width: 80%;"></div>
        <div class="skeleton skeleton-text-sm" style="width: 60%;"></div>
    </div>
    """, unsafe_allow_html=True)


def show_skeleton_banner():
    """Display a skeleton loading banner."""
    st.markdown("""
    <div class="skeleton-banner">
        <div class="skeleton skeleton-text-sm" style="width: 30%; margin: 0 auto 0.5rem auto;"></div>
        <div class="skeleton skeleton-title" style="margin: 0 auto 0.5rem auto;"></div>
        <div class="skeleton skeleton-text" style="width: 40%; margin: 0 auto;"></div>
    </div>
    """, unsafe_allow_html=True)


def show_skeleton_table(rows=3):
    """Display a skeleton loading table."""
    for _ in range(rows):
        cols = st.columns(5)
        for col in cols:
            with col:
                st.markdown('<div class="skeleton skeleton-metric" style="height: 2rem;"></div>', unsafe_allow_html=True)

# Initialize loading state
if "app_loaded" not in st.session_state:
    st.session_state["app_loaded"] = False

# Show initial loading skeleton while controller initializes
if not st.session_state["app_loaded"]:
    with st.sidebar:
        st.markdown("### Bid Evaluation System")
        st.markdown("---")
        st.markdown('<div class="skeleton skeleton-text" style="width: 80%; height: 2rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="skeleton skeleton-text" style="width: 60%; height: 2rem; margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="skeleton skeleton-text" style="width: 70%; height: 2rem; margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="skeleton skeleton-text" style="width: 50%; height: 2rem; margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
    
    # Main content skeleton
    st.markdown('<div class="skeleton skeleton-title" style="width: 30%; height: 2.5rem; margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        show_skeleton_card()
    with col2:
        show_skeleton_card()
    with col3:
        show_skeleton_card()
    show_skeleton_banner()
    show_skeleton_table(4)

# Initialize controller
@st.cache_resource
def get_controller():
    return Controller()

controller = get_controller()

# Mark app as loaded
if not st.session_state["app_loaded"]:
    st.session_state["app_loaded"] = True
    st.rerun()

# Sidebar navigation
st.sidebar.markdown("### Bid Evaluation System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Upload", "Results", "History", "Observability"],
    label_visibility="collapsed"
)

# Show stats in sidebar
stats = controller.get_stats()
st.sidebar.markdown("---")
st.sidebar.markdown("**Statistics**")
st.sidebar.metric("Total Evaluations", stats["evaluations"]["total_evaluations"])
st.sidebar.metric("Cached Companies", stats["cached_companies"])

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
                            # Show skeleton loading
                            loading_placeholder = st.empty()
                            with loading_placeholder.container():
                                st.markdown('<p class="section-header">Processing Evaluation...</p>', unsafe_allow_html=True)
                                show_skeleton_banner()
                                cols = st.columns(3)
                                for col in cols:
                                    with col:
                                        show_skeleton_card()
                            
                            result = controller.evaluate(bids, use_cached_result=False)
                            st.session_state["result"] = result.to_dict()
                            st.session_state["bids"] = data
                            loading_placeholder.empty()
                            st.success("Evaluation complete. Navigate to Results.")
                else:
                    if st.button("Run Evaluation", type="primary", use_container_width=True):
                        # Show skeleton loading
                        loading_placeholder = st.empty()
                        with loading_placeholder.container():
                            st.markdown('<p class="section-header">Processing Evaluation...</p>', unsafe_allow_html=True)
                            show_skeleton_banner()
                            cols = st.columns(3)
                            for col in cols:
                                with col:
                                    show_skeleton_card()
                        
                        result = controller.evaluate(bids)
                        st.session_state["result"] = result.to_dict()
                        st.session_state["bids"] = data
                        loading_placeholder.empty()
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
        # Show skeleton placeholder
        show_skeleton_banner()
        st.markdown('<p class="section-header">Awaiting Evaluation</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        for col in cols:
            with col:
                st.markdown('<div class="skeleton skeleton-metric" style="height: 4rem;"></div>', unsafe_allow_html=True)
        st.markdown("---")
        show_skeleton_table(3)
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
    
    # Show skeleton while loading
    loading_container = st.empty()
    with loading_container.container():
        cols = st.columns(4)
        for col in cols:
            with col:
                st.markdown('<div class="skeleton skeleton-metric"></div>', unsafe_allow_html=True)
        st.markdown("---")
        show_skeleton_card()
        show_skeleton_card()
    
    # Get history
    history = controller.get_evaluation_history(limit=20)
    stats = controller.get_stats()
    
    # Clear skeleton
    loading_container.empty()
    
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
        .run-row {
            background: #18181b;
            border: 1px solid #27272a;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            overflow: hidden;
        }
        .run-icon {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            flex-shrink: 0;
        }
        .run-icon-chain { background: #3b82f6; color: white; }
        .run-icon-llm { background: #8b5cf6; color: white; }
        .run-icon-tool { background: #f59e0b; color: white; }
        .run-icon-retriever { background: #10b981; color: white; }
        .run-icon-parser { background: #06b6d4; color: white; }
        .run-icon-prompt { background: #ec4899; color: white; }
        .run-type-badge {
            font-size: 0.65rem;
            color: #71717a;
            text-transform: uppercase;
            background: #27272a;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-weight: 500;
        }
        .duration-bar-container {
            width: 100px;
            height: 6px;
            background: #27272a;
            border-radius: 3px;
            overflow: hidden;
        }
        .duration-bar {
            height: 100%;
            border-radius: 3px;
        }
        .token-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            background: rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(139, 92, 246, 0.3);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            color: #a78bfa;
        }
        .detail-label {
            font-size: 0.7rem;
            color: #71717a;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if "result" not in st.session_state:
        st.markdown("""
        <div class="trace-header">
            <div class="skeleton skeleton-text" style="width: 40%; height: 1.5rem;"></div>
            <div class="skeleton skeleton-text-sm" style="width: 60%; margin-top: 0.5rem;"></div>
            <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                <div class="skeleton" style="width: 80px; height: 3rem;"></div>
                <div class="skeleton" style="width: 80px; height: 3rem;"></div>
                <div class="skeleton" style="width: 80px; height: 3rem;"></div>
                <div class="skeleton" style="width: 80px; height: 3rem;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        show_skeleton_card()
        show_skeleton_card()
        st.info("No evaluation data available. Run an evaluation first.")
        st.stop()
    
    result = st.session_state["result"]
    summary = controller.get_summary()
    trace_data = controller.get_trace()
    runs = controller.get_runs()
    
    # Trace Header (LangSmith style)
    trace_id = trace_data.get("trace_id", "N/A") if trace_data else "N/A"
    total_duration = trace_data.get("total_latency_ms", 0) if trace_data else summary.get("total_duration_ms", 0)
    total_tokens = trace_data.get("total_tokens", 0) if trace_data else summary.get("total_tokens", 0)
    trace_status = trace_data.get("status", "success") if trace_data else "success"
    
    status_class = "status-badge" if trace_status == "success" else "status-badge status-badge-error"
    
    st.markdown(f"""
    <div class="trace-header">
        <div class="trace-title">
            <span class="trace-name">bid_evaluation_pipeline</span>
            <span class="{status_class}">{trace_status.upper()}</span>
        </div>
        <div class="trace-id">Trace ID: {trace_id[:32] if len(str(trace_id)) > 32 else trace_id}</div>
        <div class="trace-meta">
            <div class="meta-item">
                <span class="meta-label">Latency</span>
                <span class="meta-value">{total_duration:.0f}ms</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Total Tokens</span>
                <span class="meta-value">{total_tokens:,}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Runs</span>
                <span class="meta-value">{summary.get('total_runs', len(runs))}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">LLM Calls</span>
                <span class="meta-value">{summary.get('llm_calls', 0)}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Workflow badge
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); 
                border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; 
                align-items: center; gap: 0.5rem;">
        <span style="color: #6366f1; font-weight: 600;">⬡ LangGraph</span>
        <span style="color: #a1a1aa; font-size: 0.8rem;">Active workflow orchestration</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different views
    trace_tab, runs_tab, metrics_tab, graph_tab, raw_tab = st.tabs(["Trace Tree", "All Runs", "Metrics", "Graph", "Raw"])
    
    with trace_tab:
        st.markdown('<p class="section-header">Run Trace (LangSmith Style)</p>', unsafe_allow_html=True)
        
        run_tree = controller.get_run_tree()
        
        if runs:
            max_duration = max([r.get("latency_ms", 0) or 0 for r in runs] + [1])
            
            def get_run_icon(run_type: str) -> tuple:
                icons = {
                    "chain": ("⛓️", "chain"),
                    "llm": ("🤖", "llm"),
                    "tool": ("🔧", "tool"),
                    "retriever": ("🔍", "retriever"),
                    "parser": ("📋", "parser"),
                    "prompt": ("💬", "prompt"),
                    "embedding": ("📊", "retriever")
                }
                return icons.get(run_type, ("•", "chain"))
            
            def get_duration_color(duration: float, max_dur: float) -> str:
                if not duration or not max_dur:
                    return "#3b82f6"
                ratio = duration / max_dur
                if ratio > 0.7:
                    return "#ef4444"  # Red for slow
                elif ratio > 0.4:
                    return "#f59e0b"  # Amber for medium
                return "#22c55e"  # Green for fast
            
            def render_run(run: dict, depth: int = 0):
                """Render a single run with indentation."""
                run_type = run.get("run_type", "chain")
                icon, icon_class = get_run_icon(run_type)
                duration = run.get("latency_ms") or 0
                duration_pct = min((duration / max_duration) * 100, 100) if max_duration > 0 else 0
                duration_color = get_duration_color(duration, max_duration)
                status = run.get("status", "success")
                
                # Token usage display
                token_html = ""
                if run.get("token_usage"):
                    tokens = run["token_usage"]
                    token_html = f"""
                    <span class="token-badge">
                        ◈ {tokens.get('total_tokens', 0):,} tokens
                    </span>
                    """
                
                indent_px = depth * 24
                
                with st.expander(f"{'│  ' * depth}{run.get('name', 'unknown')}", expanded=depth == 0):
                    # Header
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                        <div class="run-icon run-icon-{icon_class}">{icon}</div>
                        <div style="flex-grow: 1;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="color: #fafafa; font-weight: 600;">{run.get('name', 'unknown')}</span>
                                <span class="run-type-badge">{run_type.upper()}</span>
                                {token_html}
                            </div>
                            <div style="color: #71717a; font-size: 0.75rem; margin-top: 0.25rem;">
                                ID: {run.get('id', 'N/A')[:12]}...
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <div class="duration-bar-container">
                                    <div class="duration-bar" style="width: {duration_pct}%; background: {duration_color};"></div>
                                </div>
                                <span style="color: #a1a1aa; font-size: 0.8rem; font-family: monospace; min-width: 60px;">
                                    {duration:.0f}ms
                                </span>
                            </div>
                            <span style="color: {'#22c55e' if status == 'success' else '#ef4444'}; font-size: 0.7rem;">
                                {status.upper()}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Input/Output tabs
                    in_tab, out_tab, meta_tab = st.tabs(["Input", "Output", "Metadata"])
                    
                    with in_tab:
                        inputs = run.get("inputs", {})
                        if inputs:
                            st.json(inputs)
                        else:
                            st.markdown("*No input data*")
                    
                    with out_tab:
                        outputs = run.get("outputs")
                        if outputs:
                            st.json(outputs)
                        else:
                            st.markdown("*No output data*")
                    
                    with meta_tab:
                        meta = {
                            "id": run.get("id"),
                            "run_type": run_type,
                            "status": status,
                            "start_time": run.get("start_time"),
                            "end_time": run.get("end_time"),
                            "latency_ms": duration,
                            "parent_run_id": run.get("parent_run_id"),
                            "child_runs": len(run.get("child_run_ids", [])),
                            "tags": run.get("tags", [])
                        }
                        if run.get("model"):
                            meta["model"] = run["model"]
                        if run.get("token_usage"):
                            meta["token_usage"] = run["token_usage"]
                        if run.get("error"):
                            meta["error"] = run["error"]
                            meta["error_type"] = run.get("error_type")
                        st.json(meta)
                
                # Render children
                for child in run.get("children", []):
                    if child:
                        render_run(child, depth + 1)
            
            # Render the tree
            if run_tree:
                for root_run in run_tree:
                    if root_run:
                        render_run(root_run)
            else:
                # Fallback to flat list
                for run in runs:
                    render_run(run)
        else:
            st.info("No runs recorded yet. Run an evaluation to see traces.")
    
    with runs_tab:
        st.markdown('<p class="section-header">All Runs (Flat View)</p>', unsafe_allow_html=True)
        
        if runs:
            # Summary by type
            runs_by_type = summary.get("runs_by_type", {})
            if runs_by_type:
                cols = st.columns(len(runs_by_type))
                for i, (run_type, count) in enumerate(runs_by_type.items()):
                    with cols[i]:
                        icon, _ = get_run_icon(run_type)
                        st.metric(f"{icon} {run_type.title()}", count)
            
            st.markdown("---")
            
            # Table view
            run_data = []
            for run in runs:
                run_data.append({
                    "Name": run.get("name", ""),
                    "Type": run.get("run_type", "").upper(),
                    "Status": "✅" if run.get("status") == "success" else "❌",
                    "Latency (ms)": run.get("latency_ms") or 0,
                    "Tokens": run.get("token_usage", {}).get("total_tokens", 0) if run.get("token_usage") else 0
                })
            
            st.dataframe(
                run_data,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No runs recorded")
    
    with metrics_tab:
        st.markdown('<p class="section-header">Performance Metrics</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Latency Breakdown by Run Type**")
            
            if runs:
                latency_by_type = {}
                for run in runs:
                    rt = run.get("run_type", "other")
                    latency_by_type[rt] = latency_by_type.get(rt, 0) + (run.get("latency_ms") or 0)
                
                type_colors = {
                    "chain": "#3b82f6",
                    "llm": "#8b5cf6",
                    "tool": "#f59e0b",
                    "retriever": "#10b981",
                    "parser": "#06b6d4"
                }
                
                st.markdown('<div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 1rem;">', unsafe_allow_html=True)
                for rt, latency in sorted(latency_by_type.items(), key=lambda x: x[1], reverse=True):
                    pct = (latency / max(total_duration, 1)) * 100
                    color = type_colors.get(rt, "#71717a")
                    st.markdown(f"""
                    <div style="margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span style="color: #a1a1aa; font-size: 0.8rem;">{rt.title()}</span>
                            <span style="color: #fafafa; font-size: 0.8rem;">{latency:.0f}ms</span>
                        </div>
                        <div style="background: #27272a; height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background: {color}; height: 100%; width: {min(pct, 100):.0f}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Latency Percentiles**")
            
            latencies = [r.get("latency_ms") or 0 for r in runs if r.get("latency_ms")]
            if latencies:
                sorted_lat = sorted(latencies)
                p50 = sorted_lat[len(sorted_lat) // 2]
                p90 = sorted_lat[int(len(sorted_lat) * 0.9)]
                p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
                
                st.markdown(f"""
                <div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 1rem;">
                    <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #27272a;">
                        <span style="color: #71717a;">P50</span>
                        <span style="color: #fafafa; font-weight: 500;">{p50:.0f}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #27272a;">
                        <span style="color: #71717a;">P90</span>
                        <span style="color: #fafafa; font-weight: 500;">{p90:.0f}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #27272a;">
                        <span style="color: #71717a;">P99</span>
                        <span style="color: #fafafa; font-weight: 500;">{p99:.0f}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                        <span style="color: #71717a;">Avg</span>
                        <span style="color: #fafafa; font-weight: 500;">{sum(latencies)/len(latencies):.0f}ms</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Token usage
        st.markdown('<p class="section-header">Token Usage</p>', unsafe_allow_html=True)
        
        llm_runs = [r for r in runs if r.get("run_type") == "llm" and r.get("token_usage")]
        if llm_runs:
            total_prompt = sum(r["token_usage"].get("prompt_tokens", 0) for r in llm_runs)
            total_completion = sum(r["token_usage"].get("completion_tokens", 0) for r in llm_runs)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prompt Tokens", f"{total_prompt:,}")
            col2.metric("Completion Tokens", f"{total_completion:,}")
            col3.metric("Total Tokens", f"{total_tokens:,}")
        else:
            st.info("No LLM token data available")
        
        st.markdown("---")
        
        # RAG Config
        st.markdown('<p class="section-header">RAG Configuration</p>', unsafe_allow_html=True)
        rag = result["rag_trace"]
        
        st.markdown(f"""
        <div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 1rem;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                <div>
                    <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase;">Vector Store</div>
                    <div style="color: #fafafa; font-size: 0.875rem;">{rag["vector_store"] or "N/A"}</div>
                </div>
                <div>
                    <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase;">Embedding</div>
                    <div style="color: #fafafa; font-size: 0.875rem;">{rag["embedding_model"]}</div>
                </div>
                <div>
                    <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase;">Docs/Company</div>
                    <div style="color: #fafafa; font-size: 0.875rem;">{rag["documents_retrieved_per_company"]}</div>
                </div>
                <div>
                    <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase;">Threshold</div>
                    <div style="color: #fafafa; font-size: 0.875rem;">{rag["retrieval_confidence_threshold"]:.0%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with graph_tab:
        st.markdown('<p class="section-header">LangGraph Workflow</p>', unsafe_allow_html=True)
        
        workflow_info = controller.get_workflow_info()
        nodes = workflow_info.get("nodes", [])
        
        node_info = {
            "parse_bids": {"icon": "📥", "color": "#3b82f6", "desc": "Parse JSON bid data into objects"},
            "check_cache": {"icon": "💾", "color": "#6366f1", "desc": "Check for cached evaluation"},
            "gather_insights": {"icon": "🔍", "color": "#f59e0b", "desc": "Search Tavily + SerpAPI"},
            "gather_historical": {"icon": "📚", "color": "#10b981", "desc": "Query ChromaDB RAG"},
            "score_bids": {"icon": "📊", "color": "#8b5cf6", "desc": "Weighted multi-criteria scoring"},
            "rank_bids": {"icon": "🏆", "color": "#ec4899", "desc": "Rank and select winner"},
            "generate_explanation": {"icon": "🤖", "color": "#ef4444", "desc": "Groq LLM explanation"},
            "build_result": {"icon": "📝", "color": "#06b6d4", "desc": "Build final result object"},
            "persist_result": {"icon": "💿", "color": "#22c55e", "desc": "Save to SQLite + ChromaDB"}
        }
        
        st.markdown("""
        <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 1.5rem;">
            <div style="color: #71717a; font-size: 0.7rem; text-transform: uppercase; 
                        letter-spacing: 0.05em; margin-bottom: 1rem;">Workflow Pipeline</div>
        """, unsafe_allow_html=True)
        
        for i, node in enumerate(nodes):
            info = node_info.get(node, {"icon": "•", "color": "#71717a", "desc": ""})
            is_last = i == len(nodes) - 1
            
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; margin-bottom: {0 if is_last else '0.5rem'};">
                <div style="width: 40px; height: 40px; background: {info['color']}20; border: 2px solid {info['color']}; 
                            border-radius: 8px; display: flex; align-items: center; justify-content: center;
                            font-size: 1.1rem; margin-right: 1rem; flex-shrink: 0;">
                    {info['icon']}
                </div>
                <div style="flex-grow: 1;">
                    <div style="color: #fafafa; font-weight: 500; font-size: 0.9rem;">{node.replace('_', ' ').title()}</div>
                    <div style="color: #71717a; font-size: 0.75rem;">{info['desc']}</div>
                </div>
                <div style="color: #52525b; font-size: 0.7rem; font-family: monospace;">Node {i + 1}</div>
            </div>
            {'<div style="width: 2px; height: 12px; background: #3f3f46; margin-left: 19px;"></div>' if not is_last else ''}
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Graph state
        graph_state = controller.get_graph_state()
        if graph_state:
            st.markdown('<p class="section-header">Execution State</p>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cache Hit", "Yes" if graph_state.get("cache_hit") else "No")
            col2.metric("Bids Parsed", len(graph_state.get("parsed_bids", [])))
            col3.metric("Companies", len(graph_state.get("insights", {})))
            
            eval_id = graph_state.get("evaluation_id")
            col4.markdown(f"**Eval ID:**\n`{eval_id[:8] if eval_id else 'N/A'}...`")
    
    with raw_tab:
        st.markdown('<p class="section-header">Raw Data Export</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Runs (JSON)**")
            if runs:
                st.json(runs[:5])  # Show first 5
                st.download_button(
                    "Export All Runs",
                    json.dumps(runs, indent=2, default=str),
                    "evaluation_runs.json",
                    "application/json"
                )
        
        with col2:
            st.markdown("**Trace Summary**")
            if trace_data:
                st.json(trace_data)
                st.download_button(
                    "Export Trace",
                    json.dumps(trace_data, indent=2, default=str),
                    "trace_data.json",
                    "application/json"
                )


# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
if "result" in st.session_state:
    st.sidebar.markdown("Evaluation: Active")
else:
    st.sidebar.markdown("Evaluation: Pending")
