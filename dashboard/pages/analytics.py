"""
Analytics — Production metrics, cost tracking, and performance insights
Native Streamlit components with st.metric delta indicators
"""
import requests
import streamlit as st

from config import API_BASE, COST_TARGET_PER_VIDEO


def page():
    st.markdown("### Analytics")
    st.caption("Production metrics and cost tracking")

    # Load data
    try:
        cost_resp = requests.get(f"{API_BASE}/api/cost-summary", timeout=10)
        cost_resp.raise_for_status()
        summary = cost_resp.json()
    except Exception as e:
        st.error(f"Unable to load analytics: {e}")
        return

    total_videos = summary.get("total_videos", 0)
    hero_videos = summary.get("hero_videos", 0)
    catalog_videos = summary.get("catalog_videos", 0)
    total_cost = summary.get("total_cost_usd", 0)
    avg_cost = summary.get("avg_cost_per_video", 0)

    # ── Key Metrics ──────────────────────────────────────────────────
    st.subheader("Overview", divider="gray")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Videos", total_videos)
    with m2:
        st.metric("Total Spend", f"${total_cost:.2f}")
    with m3:
        st.metric("Avg Cost / Video", f"${avg_cost:.4f}")
    with m4:
        st.metric("Hero / Catalog", f"{hero_videos} / {catalog_videos}")

    st.divider()

    # ── Tier Breakdown & Performance ─────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Tier Breakdown", divider="gray")

        with st.container(border=True):
            if total_videos > 0:
                hero_pct = (hero_videos / total_videos) * 100
                catalog_pct = (catalog_videos / total_videos) * 100

                st.caption(f"Hero SKUs: **{hero_videos}** ({hero_pct:.0f}%)")
                st.progress(hero_pct / 100)

                st.caption(f"Catalog SKUs: **{catalog_videos}** ({catalog_pct:.0f}%)")
                st.progress(catalog_pct / 100)

                st.caption("Hero: $0.13/video — Catalog: $0.08/video")
            else:
                st.info("No videos generated yet")

    with right:
        st.subheader("Performance vs Target", divider="gray")

        with st.container(border=True):
            if avg_cost > 0:
                savings = ((COST_TARGET_PER_VIDEO - avg_cost) / COST_TARGET_PER_VIDEO) * 100

                if savings > 0:
                    st.metric(
                        "Cost Efficiency",
                        f"${avg_cost:.4f}/video",
                        delta=f"{savings:.1f}% under target",
                    )
                else:
                    st.metric(
                        "Cost Efficiency",
                        f"${avg_cost:.4f}/video",
                        delta=f"{abs(savings):.1f}% over target",
                        delta_color="inverse",
                    )

                st.caption(
                    f"Target: ${COST_TARGET_PER_VIDEO}/video · "
                    f"Actual: ${avg_cost:.4f}/video · "
                    f"Saving ${max(COST_TARGET_PER_VIDEO - avg_cost, 0):.4f}/video"
                )
            else:
                st.info("Generate videos to see performance metrics")

    st.divider()
    st.caption("Data refreshes on page load")
