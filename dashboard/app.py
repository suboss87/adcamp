"""
AdCamp - AI Video Ad Generation Dashboard
Premium multi-page shell with st.navigation()
"""
import streamlit as st

from config import (
    ACCENT, ACCENT_LIGHT,
    BORDER, BORDER_SUBTLE,
    SUCCESS, SUCCESS_LIGHT, ERROR, ERROR_LIGHT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    SHADOW_SM, SHADOW_MD,
    API_BASE,
)

# =============================================================================
#  PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AdCamp",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
#  GLOBAL CSS — Minimal overrides (theme handled by .streamlit/config.toml)
# =============================================================================

st.markdown(f"""
<style>
    /* ─── Inter font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
    }}
    .block-container {{
        padding: 2.5rem 2rem 2rem 2rem;
        max-width: 980px;
    }}

    /* ─── Sidebar nav items ─── */
    [data-testid="stSidebarNav"] a {{
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: {TEXT_SECONDARY} !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        margin: 1px 0.5rem !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stSidebarNav"] a:hover {{
        background: {BORDER_SUBTLE} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stSidebarNav"] a[aria-selected="true"] {{
        background: {ACCENT_LIGHT} !important;
        color: {ACCENT} !important;
        font-weight: 600 !important;
    }}

    /* ─── Cards (shadow-on-hover) ─── */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: none !important;
        border-radius: 12px !important;
        box-shadow: {SHADOW_SM} !important;
        transition: box-shadow 0.2s ease !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: {SHADOW_MD} !important;
    }}

    /* ─── Pipeline step component ─── */
    .ac-step {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: {TEXT_SECONDARY};
        border-radius: 10px;
        margin-bottom: 4px;
        transition: all 0.2s ease;
    }}
    .ac-step-num {{
        font-size: 0.7rem;
        font-weight: 700;
        color: {TEXT_TERTIARY};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        min-width: 3.5rem;
    }}
    .ac-step-msg {{ flex: 1; font-weight: 500; }}
    .ac-step-icon {{ font-size: 1rem; min-width: 1.5rem; text-align: center; }}
    .ac-step.running {{ background: {ACCENT_LIGHT}; }}
    .ac-step.running .ac-step-icon {{ color: {ACCENT}; font-weight: 700; }}
    .ac-step.complete {{ background: {SUCCESS_LIGHT}; }}
    .ac-step.complete .ac-step-icon {{ color: {SUCCESS}; }}
    .ac-step.failed {{ background: {ERROR_LIGHT}; }}
    .ac-step.failed .ac-step-icon {{ color: {ERROR}; }}

    /* ─── Metric label override ─── */
    [data-testid="stMetricLabel"] {{
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
    }}

    /* ─── Footer ─── */
    .ac-footer {{
        text-align: center;
        color: {TEXT_TERTIARY};
        font-size: 0.75rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid {BORDER};
        margin-top: 3rem;
    }}
    .ac-footer a {{ color: {ACCENT}; text-decoration: none; font-weight: 500; }}

    /* ─── Hide Streamlit defaults ─── */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
#  NAVIGATION
# =============================================================================

from pages.campaigns import page as campaigns_page
from pages.campaign_builder import page as builder_page
from pages.campaign_results import page as results_page
from pages.quick_test import page as quick_test_page
from pages.analytics import page as analytics_page

pages = {
    "Campaigns": [
        st.Page(campaigns_page, title="All Campaigns", icon=":material/campaign:", url_path="campaigns"),
        st.Page(builder_page, title="New Campaign", icon=":material/add_circle:", url_path="new-campaign"),
        st.Page(results_page, title="Results", icon=":material/play_circle:", url_path="results"),
    ],
    "Tools": [
        st.Page(quick_test_page, title="Quick Test", icon=":material/science:", url_path="quick-test"),
        st.Page(analytics_page, title="Analytics", icon=":material/analytics:", url_path="analytics"),
    ],
}

nav = st.navigation(pages)
nav.run()

# =============================================================================
#  FOOTER
# =============================================================================

api_docs_url = f"{API_BASE}/docs"
st.markdown(f"""
<div class="ac-footer">
    AdCamp v2.0 &middot; Powered by BytePlus ModelArk &middot;
    <a href="{api_docs_url}" target="_blank">API Docs</a>
</div>
""", unsafe_allow_html=True)
