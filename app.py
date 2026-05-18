"""
VisionGate X — Streamlit command centre
Run: streamlit run app.py
"""

from __future__ import annotations

import html
import os
import sys
import tempfile
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from pipeline import Pipeline
from modules import Database
from utils.image_utils import bgr_to_rgb, resize_frame


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"{config.APP_TITLE} · Command Centre",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Premium theme (dark glass + accent) ───────────────────────────────────────

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
    /* Root */
    .stApp {
        background: radial-gradient(1200px 800px at 10% -10%, rgba(56, 189, 248, 0.12), transparent 55%),
                    radial-gradient(900px 600px at 95% 10%, rgba(167, 139, 250, 0.10), transparent 50%),
                    linear-gradient(165deg, #0b0f14 0%, #0f1419 45%, #0a0d11 100%);
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 20, 28, 0.98) 0%, rgba(10, 13, 18, 0.98) 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown { color: rgba(255,255,255,0.88); }
    [data-testid="stHeader"] { background: transparent; }

    /* Hero */
    .vgx-hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.55) 0%, rgba(15, 23, 42, 0.35) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 24px 48px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
        margin-bottom: 1.5rem;
    }
    .vgx-brand {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        background: linear-gradient(90deg, #e2e8f0 0%, #38bdf8 40%, #a78bfa 85%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.35rem 0;
        line-height: 1.15;
    }
    .vgx-tagline {
        color: rgba(148, 163, 184, 0.95);
        font-size: 1rem;
        margin: 0;
        font-weight: 500;
    }
    .vgx-pill {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(56, 189, 248, 0.12);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.28);
    }

    /* Cards */
    .vgx-card {
        background: rgba(24, 31, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.65rem;
        backdrop-filter: blur(10px);
    }
    .vgx-card-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(148, 163, 184, 0.85);
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .vgx-metric-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 600;
        color: #f1f5f9;
    }

    /* Live log rows */
    .vgx-log-scroll {
        max-height: 420px;
        overflow-y: auto;
        padding-right: 6px;
    }
    .vgx-log-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 0.75rem;
        align-items: start;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(71, 85, 105, 0.25);
        border-left: 3px solid #38bdf8;
    }
    .vgx-log-plate {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .vgx-log-cap {
        font-size: 0.88rem;
        color: rgba(203, 213, 225, 0.92);
        margin-top: 0.35rem;
        line-height: 1.45;
    }
    .vgx-badge {
        padding: 0.28rem 0.65rem;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }
    .vgx-b-helmet { background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }
    .vgx-b-no { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.35); }
    .vgx-b-un { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

    /* Upload zone hint */
    div[data-testid="stFileUploader"] section {
        border-radius: 12px !important;
        border: 1px dashed rgba(56, 189, 248, 0.35) !important;
        background: rgba(56, 189, 248, 0.04) !important;
    }

    /* Streamlit widgets — softer */
    .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #0284c7, #6366f1) !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }

    /* Detections vault — creative card grid */
    .vgx-vault-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .vgx-vault-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .vgx-vault-sub {
        color: #64748b;
        font-size: 0.88rem;
        margin: 0.25rem 0 0 0;
    }
    .vgx-det-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
    }
    .vgx-det-card {
        position: relative;
        overflow: hidden;
        border-radius: 16px;
        padding: 1.15rem 1.25rem 1.25rem;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .vgx-det-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.35);
    }
    .vgx-det-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
        opacity: 0.85;
    }
    .vgx-det-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .vgx-det-icon {
        font-size: 1.75rem;
        line-height: 1;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
    }
    .vgx-det-plate {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: 0.06em;
    }
    .vgx-det-veh {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        font-weight: 600;
        margin-top: 2px;
    }
    .vgx-det-cap {
        margin-top: 0.85rem;
        font-size: 0.9rem;
        line-height: 1.5;
        color: rgba(203, 213, 225, 0.92);
    }
    .vgx-det-meta {
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(71, 85, 105, 0.35);
        font-size: 0.72rem;
        color: #64748b;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .vgx-chip-conf {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(56, 189, 248, 0.1);
        color: #7dd3fc;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def _estimate_yield_count(video_path: str, max_frames: int | None) -> int:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    skip = max(1, int(config.FRAME_SKIP))
    est = (n + skip - 1) // skip if n > 0 else 400
    if max_frames is not None:
        est = min(est, int(max_frames))
    return max(est, 1)


def _helmet_badge_class(status: str) -> str:
    if status == "helmet":
        return "vgx-b-helmet"
    if status == "no_helmet":
        return "vgx-b-no"
    return "vgx-b-un"


def _helmet_label(status: str) -> str:
    if status == "helmet":
        return "Helmet OK"
    if status == "no_helmet":
        return "No helmet"
    return "Unknown"


def _frame_logs_html(entries: list[dict]) -> str:
    parts = []
    for e in reversed(entries[-40:]):
        plate = html.escape(e.get("plate") or "—")
        cap_txt = html.escape(e.get("caption") or "")
        hs = e.get("helmet", "unknown")
        badge = f'<span class="vgx-badge {_helmet_badge_class(hs)}">{html.escape(_helmet_label(hs))}</span>'
        veh = html.escape(e.get("vehicle", ""))
        ts = html.escape(e.get("time", ""))
        parts.append(
            f'<div class="vgx-log-row"><div><div class="vgx-log-plate">{plate}'
            f'<span style="opacity:0.55;font-size:0.78rem;margin-left:8px">{veh}</span></div>'
            f'<div class="vgx-log-cap">{cap_txt}</div>'
            f'<div style="font-size:0.72rem;color:#64748b;margin-top:6px">{ts}</div></div>{badge}</div>'
        )
    return '<div class="vgx-log-scroll">' + "".join(parts) + "</div>" if parts else "<p style='color:#64748b'>Waiting for detections…</p>"


VEHICLE_ICONS = {
    "motorcycle": "🏍",
    "car": "🚗",
    "bus": "🚌",
    "truck": "🚛",
    "bicycle": "🚲",
}


def _vehicle_glyph(kind: str) -> str:
    return VEHICLE_ICONS.get((kind or "").lower(), "🚙")


def _detections_vault_html(rows: list[dict]) -> str:
    parts = ['<div class="vgx-det-grid">']
    for r in rows:
        plate_raw = (r.get("plate_number") or "").strip() or "—"
        plate = html.escape(plate_raw)
        veh = html.escape((r.get("vehicle_type") or "vehicle").title())
        cap = html.escape(r.get("caption") or "")
        ts = html.escape(str(r.get("timestamp") or ""))
        src = html.escape(str(r.get("source") or ""))
        hid = r.get("id", "")
        hs = r.get("helmet_status") or "unknown"
        badge_cls = _helmet_badge_class(hs)
        badge_txt = html.escape(_helmet_label(hs))
        conf = r.get("confidence")
        try:
            cf = f"{float(conf):.2f}" if conf is not None else "—"
        except (TypeError, ValueError):
            cf = "—"
        glyph = _vehicle_glyph(str(r.get("vehicle_type") or ""))
        parts.append(
            f'<div class="vgx-det-card"><div class="vgx-det-top">'
            f'<div><div class="vgx-det-icon">{glyph}</div>'
            f'<div class="vgx-det-plate">{plate}</div>'
            f'<div class="vgx-det-veh">{veh}</div></div>'
            f'<span class="vgx-badge {badge_cls}">{badge_txt}</span></div>'
            f'<div class="vgx-det-cap">{cap}</div>'
            f'<div class="vgx-det-meta"><span>#{hid} · {ts}</span>'
            f'<span class="vgx-chip-conf">YOLO {cf}</span>'
            f"<span>{src}</span></div></div>"
        )
    parts.append("</div>")
    return "".join(parts)


# ── Cached resources ───────────────────────────────────────────────────────────


@st.cache_resource
def get_db() -> Database:
    return Database()


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline(db=get_db())


db = get_db()


# ── Session defaults ─────────────────────────────────────────────────────────

if "live_logs" not in st.session_state:
    st.session_state.live_logs = []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<p style="font-size:1.35rem;font-weight:700;color:#f1f5f9;margin:0">{config.APP_ICON} {config.APP_TITLE}</p>'
        f'<p style="color:#64748b;font-size:0.85rem;margin:4px 0 0 0">v{config.APP_VERSION}</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Monitor",
            "Detections log",
            "Violations",
            "Challans",
            "Analytics",
            "Settings",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    stats = db.get_summary_stats()
    st.markdown('<p class="vgx-card-title" style="margin-bottom:10px">Overview</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="vgx-card"><span class="vgx-card-title">Detections</span>'
        f'<div class="vgx-metric-num">{stats["total_detections"]}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="vgx-card"><span class="vgx-card-title">Violations</span>'
        f'<div class="vgx-metric-num">{stats["total_violations"]}</div></div>',
        unsafe_allow_html=True,
    )
    c3, c4 = st.columns(2)
    c3.markdown(
        f'<div class="vgx-card"><span class="vgx-card-title">Challans</span>'
        f'<div class="vgx-metric-num">{stats["total_challans"]}</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="vgx-card"><span class="vgx-card-title">Fines ₹</span>'
        f'<div class="vgx-metric-num">{int(stats["total_fines_inr"]):,}</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Monitor (upload + live activity)
# ══════════════════════════════════════════════════════════════════════════════

if page == "Monitor":
    st.markdown(
        f"""
<div class="vgx-hero">
  <h1 class="vgx-brand">{config.APP_TITLE.upper().replace(" ", " · ")}</h1>
  <p class="vgx-tagline">Intelligent vehicle monitoring, captioning &amp; security — analyse uploads, audit logs, issue challans.</p>
  <span class="vgx-pill"> Video upload</span>
</div>
""",
        unsafe_allow_html=True,
    )

    up_col, opt_col = st.columns([1.15, 1])
    with up_col:
        uploaded = st.file_uploader(
            "Upload traffic or CCTV recording",
            type=["mp4", "avi", "mov", "mkv"],
            help="Full pipeline: YOLOv8 vehicles → helmet → ANPR → captions → SQLite.",
        )
    with opt_col:
        st.markdown("**Processing**")
        save_snaps = st.checkbox("Save frame snapshots", value=True)
        max_frames = st.number_input("Max frames (0 = entire video)", 0, 10000, 300)
        mf = None if max_frames == 0 else int(max_frames)

    start = st.button("Run VisionGate pipeline", type="primary", use_container_width=False)

    if start:
        if not uploaded:
            st.warning("Choose a video file first.")
        else:
            st.session_state.live_logs = []
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            pipe = get_pipeline()
            total_est = _estimate_yield_count(tmp_path, mf)
            progress_bar = st.progress(0.0, text="Initialising models…")
            _pad_l, vgc, _pad_r = st.columns([2.35, 1.32, 2.35])
            with vgc:
                st.caption("Live inference · YOLO + overlays")
                preview = st.empty()
            log_panel = st.empty()
            violation_area = st.container()

            frame_count = 0
            violation_count = 0
            challan_count = 0

            try:
                for frame_result in pipe.process_video(
                    video_path=tmp_path,
                    max_frames=mf,
                    save_snaps=save_snaps,
                    source_label=uploaded.name or "upload.mp4",
                ):
                    frame_count += 1
                    progress_bar.progress(
                        min(frame_count / total_est, 1.0),
                        text=f"Frame {frame_result.frame_number} · tracking vehicles (one DB row per passage)",
                    )

                    if frame_result.annotated_frame is not None and frame_count % 2 == 0:
                        rgb = bgr_to_rgb(resize_frame(frame_result.annotated_frame, 520))
                        preview.image(rgb, channels="RGB", use_container_width=True)

                    ts = datetime.now().strftime("%H:%M:%S")
                    for evt in frame_result.new_db_events:
                        st.session_state.live_logs.append(
                            {
                                "time": ts,
                                "plate": evt.get("plate", "—"),
                                "vehicle": evt.get("vehicle", ""),
                                "helmet": evt.get("helmet", "unknown"),
                                "caption": evt.get("caption", ""),
                            }
                        )
                    log_panel.markdown(
                        _frame_logs_html(st.session_state.live_logs),
                        unsafe_allow_html=True,
                    )

                    for v in frame_result.violations:
                        violation_count += 1
                        with violation_area:
                            st.error(
                                f"**{v.plate_number or 'Unknown plate'}** — {v.description} · "
                                f"Fine ₹{int(v.fine_inr)} · Challan `{v.challan_id}`"
                            )
                    challan_count += len(frame_result.challan_ids)

                ts_end = datetime.now().strftime("%H:%M:%S")
                for evt in pipe.consume_video_tail_events():
                    st.session_state.live_logs.append(
                        {
                            "time": ts_end,
                            "plate": evt.get("plate", "—"),
                            "vehicle": evt.get("vehicle", ""),
                            "helmet": evt.get("helmet", "unknown"),
                            "caption": evt.get("caption", ""),
                        }
                    )
                log_panel.markdown(
                    _frame_logs_html(st.session_state.live_logs),
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Pipeline error: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                progress_bar.progress(1.0, text="Finished")

            st.success(
                f"Done — {frame_count} frames analysed · {violation_count} violations · "
                f"{challan_count} challans · SQLite stores **one consolidated row per vehicle passage** "
                f"(best plate + helmet over the track)."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Detections log
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Detections log":
    st.markdown(
        """
<div class="vgx-vault-head">
  <div>
    <h2 class="vgx-vault-title">Detections vault</h2>
    <p class="vgx-vault-sub">Each card is one vehicle passage — merged plate, helmet & caption (not every frame).</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        plate_filter = st.text_input("Search plate", "")
    with c2:
        helmet_filter = st.selectbox(
            "Helmet filter",
            ["", "helmet", "no_helmet", "unknown"],
        )
    with c3:
        limit = st.slider("Cards", 10, 200, 48)

    rows = db.get_detections(limit=limit, plate_filter=plate_filter, helmet_filter=helmet_filter)

    if not rows:
        st.info("No passages logged yet. Run **Monitor** with a video — rows appear when a vehicle track closes or a violation fires.")
    else:
        st.markdown(_detections_vault_html(rows), unsafe_allow_html=True)
        st.caption(f"{len(rows)} passage records · tune tracking in `config.py` (`TRACK_MAX_MISSED_FRAMES`)")

        with st.expander("Raw table export"):
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Violations
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Violations":
    st.markdown("### Violations")
    c1, c2 = st.columns(2)
    with c1:
        plate_filter = st.text_input("Filter by plate", "")
    with c2:
        limit = st.slider("Max rows", 20, 500, 100)

    rows = db.get_violations(limit=limit, plate_filter=plate_filter)

    if not rows:
        st.info("No violations recorded.")
    else:
        df = pd.DataFrame(rows)
        df = df[
            [
                "id",
                "timestamp",
                "plate_number",
                "violation_type",
                "description",
                "fine_inr",
                "challan_id",
                "notified",
            ]
        ].rename(
            columns={
                "id": "ID",
                "timestamp": "Time",
                "plate_number": "Plate",
                "violation_type": "Type",
                "description": "Description",
                "fine_inr": "Fine (₹)",
                "challan_id": "Challan ID",
                "notified": "Notified",
            }
        )
        df["Notified"] = df["Notified"].map({0: "No", 1: "Yes"})
        st.dataframe(df, use_container_width=True, height=500)
        st.caption(f"Total fines in view: ₹{int(pd.to_numeric(df['Fine (₹)']).sum()):,}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Challans
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Challans":
    st.markdown("### E-Challans")

    c1, c2 = st.columns(2)
    with c1:
        status_filter = st.selectbox("Status", ["", "pending", "paid", "cancelled"])
    with c2:
        limit = st.slider("Max rows", 20, 500, 100)

    rows = db.get_challans(limit=limit, status_filter=status_filter)

    if not rows:
        st.info("No challans yet.")
    else:
        for row in rows:
            with st.expander(
                f"{row['challan_id']}  ·  {row['plate_number']}  ·  ₹{int(row['fine_inr'])}  ·  {row['status'].upper()}",
                expanded=False,
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Challan ID:** `{row['challan_id']}`")
                    st.markdown(f"**Vehicle:** {row['plate_number']}")
                    st.markdown(f"**Violation:** {row['violation_type']}")
                    st.markdown(f"**Fine:** ₹{int(row['fine_inr'])}")
                    st.markdown(f"**Issued:** {row['issued_at']}")
                with col2:
                    st.markdown(f"**Status:** {row['status'].upper()}")
                    if row["pdf_path"] and os.path.exists(row["pdf_path"]):
                        with open(row["pdf_path"], "rb") as f:
                            st.download_button(
                                label="Download PDF",
                                data=f.read(),
                                file_name=f"{row['challan_id']}.pdf",
                                mime="application/pdf",
                                key=row["challan_id"],
                            )
                    new_status = st.selectbox(
                        "Update status",
                        ["pending", "paid", "cancelled"],
                        index=["pending", "paid", "cancelled"].index(row["status"]),
                        key=f"status_{row['challan_id']}",
                    )
                    if new_status != row["status"]:
                        if st.button("Save", key=f"save_{row['challan_id']}"):
                            db.update_challan_status(row["challan_id"], new_status)
                            st.success("Updated")
                            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Analytics
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Analytics":
    st.markdown("### Analytics")

    stats = db.get_summary_stats()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Detections", stats["total_detections"])
    a2.metric("Violations", stats["total_violations"])
    a3.metric("Challans", stats["total_challans"])
    a4.metric("Fines (₹)", f"₹{int(stats['total_fines_inr']):,}")

    st.divider()

    all_dets = db.get_detections(limit=2000)
    if all_dets:
        df_det = pd.DataFrame(all_dets)
        df_det["timestamp"] = pd.to_datetime(df_det["timestamp"])
        df_det["hour"] = df_det["timestamp"].dt.floor("h")

        st.subheader("Detections over time")
        det_ts = df_det.groupby("hour").size().reset_index(name="count")
        st.bar_chart(det_ts.set_index("hour")["count"])

        st.subheader("Vehicle mix")
        vtype = df_det["vehicle_type"].value_counts()
        st.bar_chart(vtype)

        st.subheader("Helmet compliance (2-wheelers)")
        hc = df_det[df_det["vehicle_type"].isin(["motorcycle", "bicycle"])]
        if not hc.empty:
            hc_counts = hc["helmet_status"].value_counts()
            st.bar_chart(hc_counts)
    else:
        st.info("Process a video on **Monitor** to populate charts.")

    all_viol = db.get_violations(limit=2000)
    if all_viol:
        df_viol = pd.DataFrame(all_viol)
        st.subheader("Fines per day")
        df_viol["timestamp"] = pd.to_datetime(df_viol["timestamp"])
        df_viol["day"] = df_viol["timestamp"].dt.floor("D")
        fines_ts = df_viol.groupby("day")["fine_inr"].sum().reset_index()
        st.line_chart(fines_ts.set_index("day")["fine_inr"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Settings
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Settings":
    st.markdown("### Settings")
    st.caption("Session-scoped tweaks — reload page for full reset.")

    with st.expander("Detection thresholds", expanded=True):
        new_conf = st.slider(
            "Confidence threshold",
            0.1,
            0.9,
            float(config.CONFIDENCE_THRESHOLD),
            0.05,
        )
        new_skip = st.slider(
            "Frame skip (higher = faster; DB rows are merged per vehicle track, not per frame)",
            1,
            10,
            int(config.FRAME_SKIP),
        )
        new_prox = st.slider("Rider–vehicle proximity (px)", 50, 300, int(config.RIDER_VEHICLE_PROXIMITY))
        if st.button("Apply detection settings"):
            config.CONFIDENCE_THRESHOLD = new_conf
            config.FRAME_SKIP = new_skip
            config.RIDER_VEHICLE_PROXIMITY = new_prox
            get_pipeline.clear()
            st.success("Applied. Pipeline will reload on next run.")

    with st.expander("Vehicle track merging (detections DB)"):
        st.caption(
            "One SQLite row per passage. **Helmet policy:** if helmet appears in *any* frame of the track, "
            "the stored row is compliant; purely unknown passages are stored as no-helmet. "
            "Plate text keeps the strongest OCR score across frames."
        )
        tm = st.slider(
            "Frames unseen before passage closes",
            4,
            45,
            int(config.TRACK_MAX_MISSED_FRAMES),
            help="Lower = more frequent finalization; higher = wait longer for OCR/helmet.",
        )
        tiou = st.slider(
            "BBox IoU to match same vehicle",
            0.10,
            0.45,
            float(config.TRACK_IOU_MATCH_THRESHOLD),
            0.02,
        )
        if st.button("Apply track settings"):
            config.TRACK_MAX_MISSED_FRAMES = int(tm)
            config.TRACK_IOU_MATCH_THRESHOLD = float(tiou)
            get_pipeline.clear()
            st.success("Track settings saved for this session.")

    with st.expander("Violation fines (₹)"):
        new_helmet_fine = st.number_input(
            "No-helmet fine (₹)",
            100,
            10000,
            int(config.VIOLATION_TYPES["NO_HELMET"]["fine_inr"]),
        )
        if st.button("Apply fine settings"):
            config.VIOLATION_TYPES["NO_HELMET"]["fine_inr"] = float(new_helmet_fine)
            st.success("Fine updated.")

    with st.expander("Database"):
        st.code(config.DB_PATH)
        if st.button("Clear all data", type="secondary"):
            import sqlite3

            with sqlite3.connect(config.DB_PATH) as conn:
                conn.execute("DELETE FROM detections")
                conn.execute("DELETE FROM violations")
                conn.execute("DELETE FROM challans")
            get_pipeline.clear()
            st.success("Database cleared.")
            st.rerun()

    with st.expander("About"):
        st.markdown(
            f"""
**{config.APP_TITLE}** — v{config.APP_VERSION}

YOLOv8 · EasyOCR · SQLite · Streamlit · ReportLab challans.
"""
        )
