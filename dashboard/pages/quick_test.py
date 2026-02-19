"""
Quick Test — Single video generation with SSE progress tracking
Native components + .ac-step pipeline step HTML (genuinely custom)
"""
import json

import requests
import streamlit as st
from PIL import Image

from config import API_BASE, COST_LABELS, step_indicator


def page():
    st.markdown("### Quick Test")
    st.caption("Generate a single video ad for testing the pipeline")

    # ── Campaign Brief ────────────────────────────────────────────────
    st.subheader("Campaign Brief", divider="gray")
    brief = st.text_area(
        "Campaign Brief",
        placeholder="Describe your product, target audience, mood, and key message...",
        height=140,
        label_visibility="collapsed",
    )

    # ── Configuration ─────────────────────────────────────────────────
    st.subheader("Configuration", divider="gray")

    with st.container(border=True):
        cfg1, cfg2, cfg3, cfg4 = st.columns(4)
        with cfg1:
            sku_tier = st.selectbox(
                "SKU Tier", ["catalog", "hero"], index=0,
                format_func=lambda x: "Hero" if x == "hero" else "Catalog",
            )
        with cfg2:
            duration = st.selectbox("Duration", [2, 4, 6, 8, 10, 12, 15], index=3,
                                    format_func=lambda x: f"{x}s")
        with cfg3:
            resolution = st.selectbox("Resolution", ["480p", "720p", "1080p"], index=1)
        with cfg4:
            sku_id = st.text_input("SKU ID", value="SKU-001")

        cfg5, cfg6 = st.columns(2)
        with cfg5:
            platforms = st.multiselect(
                "Platforms", ["tiktok", "instagram", "youtube"],
                default=["tiktok"], format_func=str.capitalize,
            )
        with cfg6:
            cost_label = COST_LABELS.get(sku_tier, "$0.08 / video")
            st.caption(f"Estimated cost: **{cost_label}**")

    # ── Product Image ─────────────────────────────────────────────────
    with st.expander("Product Image (optional)"):
        uploaded_file = st.file_uploader(
            "Upload product image", type=["jpg", "jpeg", "png"],
            help="Min height: 300px. Improves video quality.",
            label_visibility="collapsed",
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption=None, use_container_width=True)
            width, height = image.size
            if height < 300:
                st.warning(f"Image height ({height}px) is below the 300px minimum")
            else:
                st.caption(f"{width} x {height}px")

    # ── Generate ──────────────────────────────────────────────────────
    generate_btn = st.button(
        "Generate Video", type="primary",
        use_container_width=True, disabled=not brief,
    )

    if not brief and generate_btn:
        st.error("Please enter a campaign brief to continue.")

    # ── SSE Generation Flow ───────────────────────────────────────────
    if generate_btn and brief:
        image_url = None
        if uploaded_file:
            with st.spinner("Uploading image..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(),
                                      uploaded_file.type)}
                    upload_resp = requests.post(
                        f"{API_BASE}/api/upload-image", files=files, timeout=30,
                    )
                    upload_resp.raise_for_status()
                    image_url = upload_resp.json().get("url")
                except Exception as e:
                    st.warning(f"Image upload failed: {e}. Continuing without image.")

        payload = {
            "brief": brief, "product_image_url": image_url,
            "sku_tier": sku_tier, "sku_id": sku_id,
            "platforms": platforms, "duration": duration,
            "resolution": resolution,
        }

        progress_area = st.container()
        script_area = st.container()
        cost_area = st.container()
        video_area = st.container()

        with progress_area:
            st.subheader("Progress", divider="gray")
            progress_bar = st.progress(0)
            step_placeholders = {i: st.empty() for i in range(1, 6)}

            try:
                response = requests.post(
                    f"{API_BASE}/api/generate-stream",
                    json=payload, stream=True, timeout=360,
                )

                final_data = {}

                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue

                    data = json.loads(line[6:])
                    step = data.get("step")
                    status = data.get("status")
                    message = data.get("message", "")
                    progress = data.get("progress", 0)

                    progress_bar.progress(progress / 100)

                    if step and step in step_placeholders:
                        css_class = status if status in ("running", "complete", "failed") else ""
                        indicator = step_indicator(status)
                        step_placeholders[step].markdown(f"""
                        <div class="ac-step {css_class}">
                            <span class="ac-step-num">Step {step}</span>
                            <span class="ac-step-msg">{message}</span>
                            <span class="ac-step-icon">{indicator}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    if "data" in data:
                        final_data.update(data["data"])

                    if status == "complete" and step == 5:
                        break
                    elif status in ("failed", "error", "timeout"):
                        st.error(message)
                        st.stop()

                # ── Script preview ────────────────────────────────
                with script_area:
                    if "script" in final_data:
                        s = final_data["script"]
                        with st.expander("View Generated Script", expanded=False):
                            st.markdown(f"**{s.get('ad_copy', 'N/A')}**")
                            st.caption(s.get("scene_description", ""))

                # ── Cost breakdown ────────────────────────────────
                with cost_area:
                    if "cost" in final_data:
                        cost = final_data["cost"]
                        cc1, cc2, cc3 = st.columns(3)
                        with cc1:
                            st.metric("Script Cost",
                                      f"${cost.get('script_cost_usd', 0):.4f}")
                        with cc2:
                            st.metric("Video Cost",
                                      f"${cost.get('video_cost_usd', 0):.4f}")
                        with cc3:
                            st.metric("Total Cost",
                                      f"${cost.get('total_cost_usd', 0):.4f}")

                # ── Video result ──────────────────────────────────
                with video_area:
                    if "video_url" in final_data:
                        st.success("Video generated successfully!",
                                   icon=":material/check_circle:")
                        st.video(final_data["video_url"])
                        st.link_button("Download Video", final_data["video_url"],
                                       use_container_width=True)

            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"Error: {e}")
