"""
Campaign Results — Video gallery with players, costs, and scripts
Native Streamlit components with st.page_link CTAs for empty states
"""
import requests
import streamlit as st

from config import API_BASE, status_badge


def page():
    st.markdown("### Campaign Results")

    campaign_id = st.session_state.get("view_campaign_id")

    if not campaign_id:
        campaign_id = st.text_input("Enter Campaign ID",
                                    placeholder="Paste a campaign ID to view results")
        if not campaign_id:
            st.markdown("#### No campaign selected")
            st.caption(
                "Select a campaign from All Campaigns, or enter a Campaign ID above."
            )
            st.page_link("campaigns", label="Go to Campaigns",
                          icon=":material/campaign:")
            return

    # Load campaign info
    try:
        camp_resp = requests.get(f"{API_BASE}/api/campaigns/{campaign_id}", timeout=10)
        camp_resp.raise_for_status()
        campaign = camp_resp.json()
    except Exception as e:
        st.error(f"Failed to load campaign: {e}")
        return

    # Campaign header
    status = campaign.get("status", "draft")
    st.markdown(f"**{campaign['name']}** {status_badge(status)}")
    theme_preview = campaign.get("theme", "")[:150]
    if theme_preview:
        st.caption(theme_preview)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Products", campaign.get("total_products", 0))
    with m2:
        st.metric("Completed", campaign.get("completed_videos", 0))
    with m3:
        st.metric("Failed", campaign.get("failed_videos", 0))
    with m4:
        st.metric("Total Cost", f"${campaign.get('total_cost_usd', 0):.2f}")

    st.divider()

    # Load results
    try:
        results_resp = requests.get(
            f"{API_BASE}/api/campaigns/{campaign_id}/results", timeout=10
        )
        results_resp.raise_for_status()
        results = results_resp.json()
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return

    if not results:
        st.markdown("#### No results yet")
        st.caption("Start generation from the Campaign Builder page.")
        st.page_link("new-campaign", label="Go to Campaign Builder",
                      icon=":material/add_circle:")
        return

    # Separate completed / failed
    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]

    # ── Completed Videos ─────────────────────────────────────────────
    if completed:
        st.subheader(f"Completed Videos · {len(completed)}", divider="gray")

        # 2-column grid
        for i in range(0, len(completed), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(completed):
                    break
                result = completed[idx]
                with col:
                    with st.container(border=True):
                        video_url = result.get("video_url")
                        if video_url:
                            st.video(video_url)

                        # Product info
                        product_id = result.get("product_id", "N/A")
                        st.caption(f"SKU: `{product_id}`")

                        # Script preview
                        script = result.get("script", {})
                        if script:
                            ad_copy = script.get("ad_copy", "")
                            if ad_copy:
                                st.markdown(f"**{ad_copy}**")
                            scene = script.get("scene_description", "")
                            if scene:
                                st.caption(scene[:120])

                        # Cost & model info
                        cost = result.get("cost", {})
                        model = result.get("model_used", "N/A")
                        cost_val = cost.get("total_cost_usd", 0) if cost else 0
                        st.caption(f"{model} — ${cost_val:.4f}")

                        if video_url:
                            st.link_button("Download", video_url,
                                           use_container_width=True)

    # ── Failed Videos ────────────────────────────────────────────────
    if failed:
        st.divider()
        with st.expander(f"Failed ({len(failed)})"):
            for r in failed:
                pid = r.get("product_id", "N/A")
                err = r.get("error", "Unknown error")
                st.caption(f"`{pid}`: {err}")
