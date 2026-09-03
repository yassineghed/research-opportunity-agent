from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent import RecommendationAgent
from src.config import PipelineConfig


st.set_page_config(
    page_title="Research Opportunity Agent",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #17212b; --muted: #637381; --line: #dce5e8; --teal: #007f82; --coral: #e66f51; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .stApp { background: linear-gradient(135deg, #f4f8f5 0%, #fffdf8 58%, #f8eee7 100%); }
    [data-testid="stSidebar"] { background: #173f46; }
    [data-testid="stSidebar"] * { color: #edf7f3 !important; }
    [data-testid="stSidebar"] .stButton button { background: #e66f51; border: 0; color: white !important; }
    .eyebrow { color: var(--coral); font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero { padding: 1.2rem 0 1rem; border-bottom: 1px solid var(--line); margin-bottom: 1.5rem; }
    .hero h1 { font-size: clamp(2.2rem, 4vw, 4rem); line-height: 1; margin: .35rem 0 .8rem; }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 650px; }
    .metric { background: rgba(255,255,255,.68); border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.1rem; }
    .metric-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { font-family: 'Space Grotesk'; font-size: 1.6rem; font-weight: 700; margin-top: .25rem; }
    .opp { background: rgba(255,255,255,.82); border-left: 4px solid var(--teal); border-radius: 6px; padding: 1rem 1.15rem; margin: .7rem 0; }
    .opp h3 { margin: 0 0 .3rem; font-size: 1.1rem; }
    .opp-meta { color: var(--muted); font-size: .88rem; }
    .score { color: var(--teal); font-family: 'Space Grotesk'; font-weight: 700; float: right; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading researchers and building the opportunity index...")
def load_agent() -> RecommendationAgent:
    config = PipelineConfig()
    agent = RecommendationAgent(config)
    agent.load_data()
    if config.index_persist_path.exists():
        agent.load_index(config.index_persist_path)
    else:
        agent.build_index(show_progress=False, persist_dir=config.index_persist_path)
    return agent


def render_opportunity(item) -> None:
    score = f"{item.score:.1f}" if isinstance(item.score, float) else str(item.score)
    deadline = f" &nbsp;·&nbsp; Deadline: {item.deadline}" if item.deadline else ""
    st.markdown(
        f"<article class=\"opp\"><span class=\"score\">{score}/100</span>"
        f"<h3>{item.title}</h3><div class=\"opp-meta\">{item.organization}"
        f" &nbsp;|&nbsp; {item.opportunity_type or 'Opportunity'}{deadline}</div></article>",
        unsafe_allow_html=True,
    )
    if item.matching_areas:
        st.caption("Matching areas: " + ", ".join(item.matching_areas))
    if item.reason:
        st.write(item.reason)
    if item.url:
        st.link_button("Open opportunity", item.url)


st.markdown(
    '<div class="hero"><div class="eyebrow">Research intelligence</div>'
    '<h1>Find the work that fits.</h1>'
    '<p>Match researcher profiles to relevant funding and collaboration opportunities, with transparent retrieval and reranking signals.</p></div>',
    unsafe_allow_html=True,
)

try:
    agent = load_agent()
except Exception as exc:
    st.error(f"The recommendation pipeline could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## Research Opportunity Agent")
    st.caption("Select a profile to explore its strongest matches.")
    researcher_labels = {
        f"{researcher.fullname} | {researcher.institution}": researcher.id
        for researcher in agent.researchers
    }
    selected_label = st.selectbox("Researcher", list(researcher_labels))
    selected_researcher = agent.researchers_by_id[researcher_labels[selected_label]]
    st.divider()
    st.markdown("**Profile focus**")
    st.write(", ".join(selected_researcher.research_domains) or "No domains listed")
    st.caption("Change data or index settings in `.env`, then use the app menu to clear the cache.")

if st.button("Generate recommendations", type="primary", use_container_width=True):
    st.session_state["result"] = agent.query(selected_researcher.id)
    st.session_state["researcher_id"] = selected_researcher.id

result = st.session_state.get("result")
if result is None or st.session_state.get("researcher_id") != selected_researcher.id:
    st.info("Choose a researcher and generate recommendations to begin.")
    st.stop()

metric_columns = st.columns(4)
metrics = [
    ("Top matches", len(result.recommendations)),
    ("Candidates retrieved", len(result.retrieval_top)),
    ("Providers", len(result.provider_results)),
    ("Total time", f"{result.elapsed_total:.1f}s"),
]
for column, (label, value) in zip(metric_columns, metrics):
    column.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown(f"### Recommended for {result.researcher_name}")
st.caption(result.institution)
for item in result.recommendations:
    render_opportunity(item)

with st.expander("Compare provider rankings"):
    provider_tabs = st.tabs(list(result.provider_results) or ["No provider results"])
    for tab, provider in zip(provider_tabs, result.provider_results):
        with tab:
            if result.provider_errors.get(provider):
                st.warning(result.provider_errors[provider])
            st.caption(f"Reranking time: {result.provider_timing[provider]:.1f}s")
            for item in result.provider_results[provider]:
                render_opportunity(item)

with st.expander("View retrieval candidates"):
    st.dataframe(
        [
            {"Title": item.title, "Organization": item.organization, "Cosine score": item.score}
            for item in result.retrieval_top
        ],
        use_container_width=True,
        hide_index=True,
    )