from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent import RecommendationAgent
from src.config import PipelineConfig
from src.models.researcher import Researcher
from src.profile import CVProfileExtractor, extract_document_text


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


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()]


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
    st.caption("Add your research profile to explore relevant opportunities.")
    manual_tab, cv_tab = st.tabs(["Manual profile", "Upload CV"])
    with manual_tab:
        with st.form("researcher_profile"):
            fullname = st.text_input("Full name", key="profile_fullname")
            institution = st.text_input("Institution", key="profile_institution")
            research_domains = st.text_area(
                "Research domains",
                placeholder="Machine learning, climate science",
                key="profile_domains",
            )
            research_interests = st.text_area(
                "Research interests",
                placeholder="Use commas or one item per line",
                key="profile_interests",
            )
            skills = st.text_area(
                "Skills",
                placeholder="Python, remote sensing, statistical modelling",
                key="profile_skills",
            )
            keywords = st.text_area(
                "Keywords",
                placeholder="Use commas or one item per line",
                key="profile_keywords",
            )
            publications = st.text_area(
                "Publications",
                placeholder="One publication per line",
                key="profile_publications",
            )
            submitted = st.form_submit_button("Save profile", use_container_width=True)
    with cv_tab:
        uploaded_cv = st.file_uploader("CV document", type=["pdf", "docx", "txt"])
        extract_submitted = st.button("Extract profile", use_container_width=True)
        st.caption("The document is processed in memory and is not saved by the dashboard.")
    if st.button("Clear profile", use_container_width=True):
        for key in (
            "profile",
            "profile_source",
            "result",
            "refresh_summary",
            "researcher_id",
            "profile_fullname",
            "profile_institution",
            "profile_domains",
            "profile_interests",
            "profile_skills",
            "profile_keywords",
            "profile_publications",
        ):
            st.session_state.pop(key, None)
        st.rerun()

if submitted:
    domains = parse_list(research_domains)
    if not fullname.strip() or not institution.strip() or not domains:
        st.sidebar.error("Please provide your name, institution, and at least one research domain.")
    else:
        selected_researcher = Researcher(
            id="manual-profile",
            fullname=fullname.strip(),
            institution=institution.strip(),
            research_domains=domains,
            research_interests=parse_list(research_interests),
            skills=parse_list(skills),
            keywords=parse_list(keywords),
            publications=parse_list(publications),
        )
        st.session_state["profile"] = selected_researcher
        st.session_state["profile_source"] = "Manual entry"
        st.session_state.pop("result", None)
        st.session_state.pop("refresh_summary", None)
        st.sidebar.success("Profile saved. Review it, then generate recommendations.")

if extract_submitted:
    if uploaded_cv is None:
        st.sidebar.error("Upload a PDF, DOCX, or TXT CV first.")
    else:
        try:
            cv_text = extract_document_text(uploaded_cv.name, uploaded_cv.getvalue())
            extracted_profile = CVProfileExtractor().extract(cv_text)
            st.session_state["profile"] = extracted_profile
            st.session_state["profile_source"] = uploaded_cv.name
            st.session_state.pop("result", None)
            st.session_state.pop("refresh_summary", None)
            st.sidebar.success("Profile extracted. Review it before generating recommendations.")
        except Exception as exc:
            st.sidebar.error(f"CV extraction failed: {exc}")

selected_researcher = st.session_state.get("profile")

result = st.session_state.get("result")
if selected_researcher is None:
    st.info("Complete your profile or upload a CV to begin.")
    st.stop()

st.markdown("### Extracted profile" if st.session_state.get("profile_source", "").lower().endswith((".pdf", ".docx", ".txt")) else "### Researcher profile")
st.caption(f"Source: {st.session_state.get('profile_source', 'Manual entry')}")
with st.expander("Review profile data", expanded=True):
    profile_columns = st.columns(2)
    profile_columns[0].markdown(f"**Name**  \n{selected_researcher.fullname}")
    profile_columns[1].markdown(f"**Institution**  \n{selected_researcher.institution}")
    profile_columns[0].markdown(
        "**Research domains**  \n" + ", ".join(selected_researcher.research_domains)
    )
    profile_columns[1].markdown(
        "**Research interests**  \n" + (", ".join(selected_researcher.research_interests) or "Not provided")
    )
    profile_columns[0].markdown(
        "**Skills**  \n" + (", ".join(selected_researcher.skills) or "Not provided")
    )
    profile_columns[1].markdown(
        "**Keywords**  \n" + (", ".join(selected_researcher.keywords) or "Not provided")
    )
    st.markdown(
        "**Publications**  \n" + ("  \n".join(selected_researcher.publications) or "Not provided")
    )

refresh_col, recommend_col = st.columns(2)
with refresh_col:
    refresh_requested = st.button(
        "Search newer F&T opportunities",
        use_container_width=True,
        help="Search the Funding & Tenders Portal using your research domains and interests.",
    )
with recommend_col:
    recommend_requested = st.button(
        "Generate recommendations",
        type="primary",
        use_container_width=True,
    )

if refresh_requested:
    with st.spinner("Searching the Funding & Tenders Portal..."):
        refresh_summary = agent.refresh_for_profile(selected_researcher)
        st.session_state["refresh_summary"] = refresh_summary
        st.session_state["result"] = agent.query_from_profile(selected_researcher)
        st.session_state["researcher_id"] = selected_researcher.id

if st.session_state.get("refresh_summary"):
    summary = st.session_state["refresh_summary"]
    if summary["index_refreshed"]:
        st.success(
            f"Found {summary['fetched']} profile-specific opportunities; "
            f"added {summary['added']} new opportunities to the index."
        )
    else:
        st.info("No new profile-specific opportunities were found.")

if recommend_requested:
    with st.spinner("Finding relevant opportunities..."):
        st.session_state["result"] = agent.query_from_profile(selected_researcher)
        st.session_state["researcher_id"] = selected_researcher.id

result = st.session_state.get("result")
if result is None or st.session_state.get("researcher_id") != selected_researcher.id:
    st.info("Review your profile, then generate recommendations.")
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