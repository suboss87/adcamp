"""
All Campaigns — Campaign list with status, metrics, actions
Native Streamlit components with st.dialog for delete confirmation
"""
import requests
import streamlit as st

from config import API_BASE, status_badge


@st.dialog("Delete Campaign")
def _confirm_delete(campaign_id: str, campaign_name: str):
    st.write(f"Are you sure you want to delete **{campaign_name}**?")
    st.caption("This action cannot be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("Delete", type="primary", use_container_width=True):
            try:
                requests.delete(
                    f"{API_BASE}/api/campaigns/{campaign_id}", timeout=10
                ).raise_for_status()
                st.toast("Campaign deleted", icon=":material/check_circle:")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")


def page():
    st.markdown("### All Campaigns")

    try:
        resp = requests.get(f"{API_BASE}/api/campaigns", timeout=10)
        resp.raise_for_status()
        campaigns = resp.json()
    except Exception as e:
        st.error(f"Failed to load campaigns: {e}")
        return

    if not campaigns:
        st.markdown("#### No campaigns yet")
        st.caption(
            "Create your first campaign to start generating video ads at scale."
        )
        st.page_link("new-campaign", label="Create Campaign",
                      icon=":material/add_circle:")
        return

    # ── Summary metrics ──────────────────────────────────────────────
    total = len(campaigns)
    active = sum(1 for c in campaigns if c.get("status") == "generating")
    completed = sum(1 for c in campaigns if c.get("status") == "completed")
    total_cost = sum(c.get("total_cost_usd", 0) for c in campaigns)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Campaigns", total)
    with m2:
        st.metric("Active", active)
    with m3:
        st.metric("Completed", completed)
    with m4:
        st.metric("Total Spend", f"${total_cost:.2f}")

    st.divider()

    # ── Campaign cards ───────────────────────────────────────────────
    for c in campaigns:
        status = c.get("status", "draft")
        done = c.get("completed_videos", 0)
        failed = c.get("failed_videos", 0)
        total_p = c.get("total_products", 0)
        cost = c.get("total_cost_usd", 0)
        created = c.get("created_at", "")[:10]
        platforms = ", ".join(p.capitalize() for p in c.get("platforms", []))

        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**{c['name']}** {status_badge(status)}")
                theme_preview = c.get("theme", "")[:120]
                if theme_preview:
                    st.caption(theme_preview)
                meta_parts = []
                if platforms:
                    meta_parts.append(platforms)
                if c.get("duration"):
                    meta_parts.append(f'{c["duration"]}s')
                if c.get("resolution"):
                    meta_parts.append(c["resolution"])
                if meta_parts:
                    st.caption(" · ".join(meta_parts))

            with right:
                st.metric("Videos", f"{done}/{total_p}")
                st.metric("Spent", f"${cost:.2f}")
                st.caption(created)

            # Actions via popover
            with st.popover(":material/more_vert:", use_container_width=False):
                if done > 0:
                    if st.button("View Results", key=f"r_{c['id']}",
                                 use_container_width=True):
                        st.session_state["view_campaign_id"] = c["id"]
                        st.switch_page("results")
                if status == "draft" and total_p > 0:
                    if st.button("Generate", key=f"g_{c['id']}",
                                 use_container_width=True):
                        st.session_state["build_campaign_id"] = c["id"]
                        st.switch_page("new-campaign")
                if st.button("Delete", key=f"d_{c['id']}",
                             use_container_width=True):
                    _confirm_delete(c["id"], c["name"])
