"""
Campaign Builder — Create campaign, upload CSV, trigger batch generation
Clean wizard-style flow with @st.fragment auto-polling
"""
import requests
import streamlit as st

from config import API_BASE


def page():
    st.markdown("### New Campaign")

    # ── Step 1: Campaign Details ──────────────────────────────────────
    st.subheader("Step 1 — Campaign Details", divider="gray")

    with st.container(border=True):
        name = st.text_input("Campaign Name", placeholder="e.g. Summer 2025 Collection")
        theme = st.text_area(
            "Campaign Theme",
            placeholder="Describe the overall mood, aesthetic, and message.\nThis theme will be used to auto-generate briefs for every product...",
            height=100,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            platforms = st.multiselect(
                "Platforms", ["tiktok", "instagram", "youtube"],
                default=["tiktok"], format_func=str.capitalize,
            )
        with c2:
            duration = st.selectbox("Duration", [2, 4, 6, 8, 10, 12], index=3,
                                    format_func=lambda x: f"{x}s")
        with c3:
            resolution = st.selectbox("Resolution", ["480p", "720p", "1080p"], index=1)

    # Create campaign
    campaign_id = st.session_state.get("build_campaign_id")

    if not campaign_id:
        if st.button("Create Campaign", type="primary", disabled=not (name and theme)):
            try:
                resp = requests.post(f"{API_BASE}/api/campaigns", json={
                    "name": name, "theme": theme, "platforms": platforms,
                    "duration": duration, "resolution": resolution,
                }, timeout=10)
                resp.raise_for_status()
                st.session_state["build_campaign_id"] = resp.json()["id"]
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
        return

    st.toast(f"Campaign ready: {campaign_id}", icon=":material/check_circle:")
    st.info(f"Campaign ID: `{campaign_id}`")

    # ── Step 2: Upload Products ───────────────────────────────────────
    st.subheader("Step 2 — Product Catalog", divider="gray")
    st.caption(
        "CSV columns: `sku_id`, `product_name`, `description` (required) · "
        "`image_url`, `sku_tier`, `category` (optional)"
    )

    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_csv and st.button("Upload Products", type="primary"):
        with st.spinner("Parsing..."):
            try:
                files = {"file": (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")}
                resp = requests.post(
                    f"{API_BASE}/api/campaigns/{campaign_id}/products",
                    files=files, timeout=30,
                )
                resp.raise_for_status()
                r = resp.json()
                st.toast(f"{r['products_created']} products uploaded",
                         icon=":material/check_circle:")
                if r.get("errors"):
                    with st.expander(f"{r['products_skipped']} rows skipped"):
                        for err in r["errors"]:
                            st.caption(err)
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # Show products
    try:
        products = requests.get(
            f"{API_BASE}/api/campaigns/{campaign_id}/products", timeout=10
        ).json()
    except Exception:
        products = []

    if products:
        st.subheader(f"{len(products)} Products", divider="gray")
        for p in products[:15]:
            tier = "Hero" if p.get("sku_tier") == "hero" else "Catalog"
            st.markdown(f"`{p['sku_id']}` {p['product_name']} — {tier}")
        if len(products) > 15:
            st.caption(f"+ {len(products) - 15} more")

    # ── Step 3: Generate ──────────────────────────────────────────────
    if products:
        st.subheader("Step 3 — Generate Videos", divider="gray")

        pending_count = sum(1 for p in products if p.get("status") == "pending")
        if pending_count == 0:
            st.info("All products processed. Check the Results page.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                concurrency = st.slider("Parallel jobs", 1, 10, 3)
            with c2:
                st.caption(f"{pending_count} products ready")

            if st.button("Start Generation", type="primary", use_container_width=True):
                try:
                    requests.post(
                        f"{API_BASE}/api/campaigns/{campaign_id}/generate",
                        json={"concurrency": concurrency}, timeout=10,
                    ).raise_for_status()
                    st.session_state["polling_campaign_id"] = campaign_id
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    # ── Progress Polling (non-blocking fragment) ──────────────────────
    if st.session_state.get("polling_campaign_id") == campaign_id:
        _poll_progress(campaign_id)


@st.fragment(run_every=3)
def _poll_progress(campaign_id: str):
    """Auto-polling fragment — page stays interactive during generation."""
    st.subheader("Progress", divider="gray")

    try:
        p = requests.get(
            f"{API_BASE}/api/campaigns/{campaign_id}/progress", timeout=10
        ).json()
    except Exception:
        st.caption("Waiting for progress data...")
        return

    total = p.get("total_products", 1)
    done = p.get("completed_videos", 0) + p.get("failed_videos", 0)
    pct = p.get("progress_pct", 0)
    status = p.get("status", "generating")

    st.progress(pct / 100, text=f"{done}/{total} processed")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Completed", p.get("completed_videos", 0))
    with mc2:
        st.metric("Failed", p.get("failed_videos", 0))
    with mc3:
        st.metric("Cost", f"${p.get('total_cost_usd', 0):.4f}")

    if status in ("completed", "partial", "failed"):
        if status == "completed":
            st.toast("All videos generated successfully",
                     icon=":material/check_circle:")
            st.success("All videos generated successfully")
        elif status == "partial":
            st.toast("Generation completed with some failures",
                     icon=":material/warning:")
            st.warning(
                f"{p.get('completed_videos', 0)} succeeded, "
                f"{p.get('failed_videos', 0)} failed"
            )
        else:
            st.toast("Batch generation failed", icon=":material/error:")
            st.error("Batch generation failed")
        del st.session_state["polling_campaign_id"]
