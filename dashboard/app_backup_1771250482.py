"""
AdCamp - Modern Video Ad Generation Dashboard
AI-Powered Product Video Generation at Scale
"""
import time
import os
from typing import Optional

import requests
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Use production GCP Cloud Run API by default, fall back to localhost for local dev
API_BASE = os.getenv(
    "API_URL", 
    "https://adcamp-api-309502792454.asia-southeast1.run.app"
)

# Modern color palette
COLORS = {
    "primary": "#0066FF",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "hero": "#8B5CF6",
    "catalog": "#3B82F6",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AdCamp · AI Video Generation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimal, clean look
st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }
    
    /* Minimal cards */
    .result-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .result-card h4 {
        font-size: 0.875rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    /* Clean buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER SECTION
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🎬 AdCamp")
st.caption("AI-Powered Video Ad Generation")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR - CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Settings")
    
    sku_tier = st.selectbox(
        "SKU Tier",
        ["catalog", "hero"],
        index=0,
        format_func=lambda x: f"Hero (Premium)" if x == "hero" else "Catalog (Optimized)"
    )
    
    st.caption(f"💰 Cost: {'$0.13' if sku_tier == 'hero' else '$0.08'}/video")
    
    resolution = st.select_slider(
        "Resolution",
        options=["480p", "720p", "1080p"],
        value="720p"
    )
    
    duration = st.slider(
        "Duration (sec)",
        2, 15, 8
    )
    
    platforms = st.multiselect(
        "Platforms",
        ["tiktok", "instagram", "youtube"],
        default=["tiktok"],
        format_func=lambda x: x.title()
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT - TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🎬 Generate Video", "📊 Analytics", "📚 Library"])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1: VIDEO GENERATION
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    col_left, col_right = st.columns([2, 1], gap="large")
    
    with col_left:
        st.markdown("### 📝 Campaign Details")
        
        brief = st.text_area(
            "Campaign Brief",
            placeholder="Example: Summer collection launch featuring lightweight running shoes. Target audience: active millennials (25-35). Mood: energetic, aspirational. Setting: sunrise beach run with golden hour lighting. Key message: performance meets style.",
            height=140,
            help="Describe your campaign vision, target audience, mood, and key message"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            sku_id = st.text_input(
                "SKU ID",
                value="SKU-001",
                placeholder="e.g., SHOE-RUN-2024-001"
            )
        with col_b:
            campaign_name = st.text_input(
                "Campaign Name",
                placeholder="e.g., Summer Sprint 2024",
                help="Internal reference name"
            )
    
    with col_right:
        st.markdown("### 🖼️ Product Image")
        
        image_url = st.text_input(
            "Image URL (optional)",
            placeholder="https://example.com/product.jpg",
            help="Min height: 300px. Leave empty to generate without image.",
            label_visibility="collapsed"
        )
        
        if image_url:
            try:
                st.image(image_url, caption="Product Preview", use_container_width=True)
                st.success("✓ Image loaded successfully")
            except:
                st.error("⚠️ Unable to load image preview")
        else:
            st.info("💡 **Pro Tip**: Adding a product image improves video relevance and quality")
    
    st.divider()
    
    # Generate button with better styling
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_btn = st.button(
            "🚀 Generate Video Ad",
            type="primary",
            use_container_width=True,
            disabled=not brief
        )
    
    if not brief and generate_btn:
        st.error("⚠️ Please enter a campaign brief to continue")
    
    # ─── GENERATION FLOW ───
    if generate_btn and brief:
        payload = {
            "brief": brief,
            "product_image_url": image_url or None,
            "sku_tier": sku_tier,
            "sku_id": sku_id,
            "platforms": platforms,
            "duration": duration,
            "resolution": resolution,
        }
        
        # Create sections for results
        progress_container = st.container()
        script_container = st.container()
        routing_container = st.container()
        cost_container = st.container()
        video_container = st.container()
        
        with progress_container:
            with st.spinner("Generating ad script..."):
                try:
                    resp = requests.post(f"{API_BASE}/api/generate", json=payload, timeout=120)
                    resp.raise_for_status()
                    result = resp.json()
                    st.success("✓ Script generated, video creation started")
                
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ Generation failed: HTTP {e.response.status_code}")
                    try:
                        error_detail = e.response.json()
                        st.caption(str(error_detail.get('detail', '')))
                    except:
                        pass
                    st.stop()
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.stop()
        
        # ─── SCRIPT OUTPUT ───
        with script_container:
            st.divider()
            script = result.get("script", {})
            
            with st.expander("📝 View Generated Script", expanded=True):
                st.write("**Ad Copy:**", script.get('ad_copy', 'N/A'))
                st.caption(script.get('scene_description', ''))
        
        
        # ─── COST BREAKDOWN ───
        with cost_container:
            cost_info = result.get("cost", {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Script Cost", f"${cost_info.get('script_cost_usd', 0):.4f}")
            with col2:
                st.metric("Video Cost", f"${cost_info.get('video_cost_usd', 0):.4f}")
            with col3:
                st.metric("Total Cost", f"${cost_info.get('total_cost_usd', 0):.4f}")
        
        # ─── VIDEO OUTPUT ───
        with video_container:
            st.divider()
            task_id = result.get("task_id")
            if task_id:
                progress_bar = st.progress(0, text="⏳ Generating video...")
                
                for i in range(60):  # Max 5 minutes
                    progress = min((i + 1) * 3, 95)
                    progress_bar.progress(progress, text=f"⏳ Generating... {(i+1)*5}s")
                    
                    try:
                        status_resp = requests.get(f"{API_BASE}/api/status/{task_id}", timeout=30)
                        status = status_resp.json()
                    except:
                        time.sleep(5)
                        continue
                    
                    if status.get("status") == "Succeeded":
                        progress_bar.progress(100, text="✅ Complete")
                        video_url = status.get("video_url")
                        if video_url:
                            st.video(video_url)
                            st.link_button("⬇️ Download", video_url)
                        break
                    
                    elif status.get("status") == "Failed":
                        st.error(f"Generation failed: {status.get('error', '')}")
                        break
                    
                    time.sleep(5)
                else:
                    st.warning(f"Timeout. Task ID: {task_id}")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### 📊 Production Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔄 Refresh Analytics", use_container_width=True):
            try:
                cost_resp = requests.get(f"{API_BASE}/api/cost-summary", timeout=10)
                cost_resp.raise_for_status()
                summary = cost_resp.json()
                
                # Key metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Total Videos</div>
                    </div>
                    """.format(summary.get("total_videos", 0)), unsafe_allow_html=True)
                
                with col_b:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">${:.4f}</div>
                        <div class="metric-label">Avg Cost/Video</div>
                    </div>
                    """.format(summary.get("avg_cost_per_video", 0)), unsafe_allow_html=True)
                
                with col_c:
                    st.markdown("""
                    <div class="metric-card" style="border-left-color: #8B5CF6;">
                        <div class="metric-value" style="color: #7c3aed;">{}</div>
                        <div class="metric-label">Hero Videos</div>
                    </div>
                    """.format(summary.get("hero_videos", 0)), unsafe_allow_html=True)
                
                with col_d:
                    st.markdown("""
                    <div class="metric-card" style="border-left-color: #3B82F6;">
                        <div class="metric-value" style="color: #2563eb;">{}</div>
                        <div class="metric-label">Catalog Videos</div>
                    </div>
                    """.format(summary.get("catalog_videos", 0)), unsafe_allow_html=True)
                
                st.divider()
                
                # Additional insights
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("#### 💰 Cost Summary")
                    total_cost = summary.get('total_cost_usd', 0)
                    hero_videos = summary.get("hero_videos", 0)
                    catalog_videos = summary.get("catalog_videos", 0)
                    total_videos = hero_videos + catalog_videos
                    
                    if total_videos > 0:
                        hero_pct = (hero_videos / total_videos) * 100
                        catalog_pct = (catalog_videos / total_videos) * 100
                        
                        st.metric("Total Spend", f"${total_cost:.2f}")
                        st.progress(hero_pct / 100, text=f"Hero: {hero_pct:.1f}%")
                        st.progress(catalog_pct / 100, text=f"Catalog: {catalog_pct:.1f}%")
                    else:
                        st.info("No videos generated yet")
                
                with col_right:
                    st.markdown("#### 🎯 Performance vs Target")
                    target_cost = 0.16
                    avg_cost = summary.get('avg_cost_per_video', 0)
                    
                    if avg_cost > 0:
                        savings = ((target_cost - avg_cost) / target_cost) * 100
                        
                        if savings > 0:
                            st.success(f"✅ **{savings:.1f}% under target** (${target_cost - avg_cost:.4f} savings/video)")
                        else:
                            st.warning(f"⚠️ **{abs(savings):.1f}% over target**")
                        
                        st.caption(f"Target: ${target_cost}/video · Actual: ${avg_cost:.4f}/video")
                    else:
                        st.info("Generate videos to see performance metrics")
                
            except Exception as e:
                st.error(f"Unable to load analytics: {str(e)}")
    
    with col2:
        st.markdown("#### 🎯 Quick Stats")
        st.markdown("""
        - **Target Cost**: $0.16/video
        - **Target Savings**: 50%
        - **Pipeline Speed**: ~30s/video
        - **Quality Target**: 80%+ approval
        """)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3: VIDEO LIBRARY (Placeholder)
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### 📚 Video Library")
    st.info("🚧 **Coming Soon**: Browse and manage your generated videos")
    
    st.markdown("""
    **Planned Features:**
    - 🔍 Search and filter videos
    - 📁 Organize by campaign
    - 📊 Performance analytics per video
    - 🔄 Regenerate variations
    - 📤 Bulk export
    """)

# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🎬 AdCamp v1.0 · BytePlus ModelArk")

with footer_col2:
    monitoring_url = "http://localhost:3000" if "localhost" in API_BASE else "#"
    api_docs_url = f"{API_BASE}/docs"
    st.caption(f"📊 [View Monitoring]({monitoring_url}) · [API Docs]({api_docs_url})")

with footer_col3:
    st.caption("Powered by Seed 1.8 & Seedance")
