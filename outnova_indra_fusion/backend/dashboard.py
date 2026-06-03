"""
OutNova Indra Fusion — Dashboard Streamlit para analistas
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="OutNova — Dashboard Analista",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* OutNova / Majito palette */
    :root {
        --on-primary: #2563EB;
        --on-success: #16A34A;
        --on-warning: #D97706;
        --on-error: #DC2626;
        --on-surface: #F8FAFC;
        --on-border: #E5E7EB;
    }
    .block-container { padding-top: 1.2rem; }
    [data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px 14px;
    }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    .risk-low  { color: #16A34A; font-weight: 700; font-size: 1.4rem; }
    .risk-mid  { color: #D97706; font-weight: 700; font-size: 1.4rem; }
    .risk-high { color: #DC2626; font-weight: 700; font-size: 1.4rem; }
    .coercion-alert {
        background: #FEF2F2;
        border: 1px solid #DC2626;
        border-radius: 8px;
        padding: 12px 16px;
        color: #991B1B;
        font-weight: 500;
    }
    /* Transcript message cards */
    .aura-msg {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
        color: #1E3A5F;
        font-size: 0.92rem;
    }
    .user-msg {
        background: #F8FAFC;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
        color: #111827;
        font-size: 0.92rem;
    }
    .msg-meta {
        font-size: 0.75rem;
        color: #6B7280;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge-ptt {
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        border-radius: 999px;
        padding: 1px 8px;
        font-size: 0.70rem;
        font-weight: 600;
    }
    .badge-typed {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 999px;
        padding: 1px 8px;
        font-size: 0.70rem;
        font-weight: 600;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] select,
    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)


def risk_color_class(score):
    if score is None:
        return "risk-mid"
    if score < 30:
        return "risk-low"
    elif score < 70:
        return "risk-mid"
    return "risk-high"


def risk_emoji(score):
    if score is None:
        return "🟡"
    if score < 30:
        return "🟢"
    elif score < 70:
        return "🟡"
    return "🔴"


def fetch_sessions(status_filter=None, limit=50, offset=0):
    params = {"limit": limit, "offset": offset}
    if status_filter:
        params["status"] = status_filter
    try:
        r = requests.get(f"{API_BASE}/sessions", params=params, timeout=5)
        return r.json()
    except Exception as e:
        st.error(f"No se pudo conectar al backend: {e}")
        return {"sessions": [], "total": 0}


def fetch_session(session_id: int):
    try:
        r = requests.get(f"{API_BASE}/sessions/{session_id}", timeout=5)
        return r.json()
    except Exception:
        return None


def patch_status(session_id: int, status: str, reason: str = ""):
    try:
        r = requests.patch(
            f"{API_BASE}/sessions/{session_id}/status",
            json={"status": status, "reason": reason},
            timeout=5,
        )
        return r.ok
    except Exception:
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/2563EB/FFFFFF?text=OutNova", use_container_width=True)
    st.title("Panel de Analista")
    st.divider()

    status_filter = st.selectbox(
        "Filtrar por estado",
        ["Todos", "Pendiente", "Procesando", "Aprobado", "Rechazado", "Recaptura"],
    )
    if st.button("🔄 Actualizar lista"):
        st.rerun()

    st.divider()
    filter_val = None if status_filter == "Todos" else status_filter
    data = fetch_sessions(status_filter=filter_val, limit=100)
    sessions = data.get("sessions", [])
    total = data.get("total", 0)

    st.caption(f"{total} sesiones totales")

    if not sessions:
        st.info("Sin sesiones disponibles.")
        st.stop()

    options = {
        f"#{s['id']} — {s['status']} — {(s.get('risk_score') or '?')}pts — {s['created_at'][:10]}": s["id"]
        for s in sessions
    }
    selected_label = st.radio("Seleccionar sesión", list(options.keys()))
    selected_id = options[selected_label]

# ── Main panel ─────────────────────────────────────────────────────────────
s = fetch_session(selected_id)
if not s:
    st.error("No se pudo cargar la sesión.")
    st.stop()

score = s.get("risk_score")
status = s.get("status", "Pendiente")
col_title, col_status = st.columns([3, 1])

with col_title:
    st.markdown(f"## Sesión #{s['id']}")
    st.caption(f"ID cliente: `{s.get('client_session_id', '—')}`")
    st.caption(f"Creada: {s.get('created_at', '—')[:19].replace('T', ' ')}")

with col_status:
    st.markdown(f"### {risk_emoji(score)} {score:.1f}/100" if score is not None else "### ⚪ Sin score")
    st.markdown(f"**Estado:** {status}")

# ── Decision ───────────────────────────────────────────────────────────────
reason = s.get("decision_reason") or s.get("status_reason") or "—"
if status == "Aprobado":
    st.success(f"✅ {reason}")
elif status == "Rechazado":
    st.error(f"❌ {reason}")
elif status == "Recaptura":
    st.warning(f"🔁 {reason}")
else:
    st.info(f"ℹ️ {reason}")

# ── Manual action buttons ──────────────────────────────────────────────────
st.subheader("Decisión Manual")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("✅ Aprobar", use_container_width=True):
        if patch_status(selected_id, "Aprobado", "Aprobado manualmente por analista."):
            st.success("Aprobado.")
            st.rerun()
with col2:
    if st.button("❌ Rechazar", use_container_width=True):
        if patch_status(selected_id, "Rechazado", "Rechazado manualmente por analista."):
            st.error("Rechazado.")
            st.rerun()
with col3:
    if st.button("🔁 Recaptura", use_container_width=True):
        if patch_status(selected_id, "Recaptura", "Enviado a recaptura por analista."):
            st.warning("Enviado a recaptura.")
            st.rerun()
with col4:
    if st.button("⏳ Pendiente", use_container_width=True):
        if patch_status(selected_id, "Pendiente", "Marcado pendiente por analista."):
            st.info("Marcado como pendiente.")
            st.rerun()

st.divider()

# ── Score Donut Chart ──────────────────────────────────────────────────────
breakdown = s.get("risk_breakdown") or {}
if isinstance(breakdown, str):
    try:
        breakdown = json.loads(breakdown)
    except Exception:
        breakdown = {}

deepface_avail = s.get("deepface_available", False)

donut_labels = ["Voice Behavioral", "Liveness & Spoof", "Audio Quality", "Multimodal"]
donut_values = [
    breakdown.get("voice_behavioral_risk_score") or s.get("voice_behavioral_risk_score") or 0,
    breakdown.get("liveness_spoof_risk_score") or s.get("liveness_spoof_risk_score") or 0,
    breakdown.get("audio_quality_risk_score") or s.get("audio_quality_risk_score") or 0,
    breakdown.get("multimodal_consistency_risk_score") or s.get("multimodal_consistency_risk_score") or 0,
]
if deepface_avail:
    donut_labels.append("DeepFace Emoción")
    donut_values.append(breakdown.get("deepface_emotion_risk_score") or s.get("deepface_emotion_risk_score") or 0)

col_donut, col_metrics = st.columns([1, 2])

with col_donut:
    st.subheader("Distribución de Riesgo")
    fig = go.Figure(go.Pie(
        labels=donut_labels,
        values=donut_values,
        hole=0.5,
        textinfo="label+percent",
        marker_colors=["#2563EB", "#DC2626", "#D97706", "#16A34A", "#7C3AED"],
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_metrics:
    st.subheader("Sub-scores")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Voice Behavioral", f"{(s.get('voice_behavioral_risk_score') or 0):.1f}")
        st.metric("Audio Quality", f"{(s.get('audio_quality_risk_score') or 0):.1f}")
        st.metric("Coerción score", f"{(s.get('coercion_score') or 0):.2f}")
    with m2:
        st.metric("Liveness & Spoof", f"{(s.get('liveness_spoof_risk_score') or 0):.1f}")
        st.metric("Multimodal", f"{(s.get('multimodal_consistency_risk_score') or 0):.1f}")
        st.metric("Quality conf.", f"{(s.get('quality_confidence') or 0):.0%}")

st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_video, tab_audio, tab_coercion, tab_aura, tab_deepface, tab_raw = st.tabs([
    "🎥 Video/Liveness", "🎙 Audio", "⚠️ Coerción", "💬 AURA", "🧠 DeepFace", "🔧 Raw JSON"
])

# ── VIDEO TAB ──────────────────────────────────────────────────────────────
with tab_video:
    st.subheader("Liveness & Parpadeos")
    c1, c2, c3, c4 = st.columns(4)
    blink_rate = s.get("blink_rate_per_min") or 0
    c1.metric("Total parpadeos", s.get("total_blinks") or 0)
    c2.metric("Rate/min", f"{blink_rate:.1f}", help="Normal: 8–25 /min")
    c3.metric("Cara detectada", f"{(s.get('face_detected_ratio') or 0):.0%}")
    c4.metric("Ojos detectados", f"{(s.get('eye_detected_ratio') or 0):.0%}")

    if blink_rate < 3 or blink_rate > 50:
        st.warning(f"⚠️ Blink rate anormal: {blink_rate:.1f}/min (normal 8–25)")
    elif 8 <= blink_rate <= 25:
        st.success(f"✅ Blink rate normal: {blink_rate:.1f}/min")

    if s.get("deepfake_flag"):
        st.error(f"🚨 Deepfake/No-Liveness: {s.get('deepfake_reason', '')}")
    else:
        st.success(f"✅ {s.get('deepfake_reason', 'Sin indicadores de deepfake.')}")

    st.metric("Duración video", f"{(s.get('video_duration_sec') or 0):.1f}s")

# ── AUDIO TAB ──────────────────────────────────────────────────────────────
with tab_audio:
    st.subheader("Transcripción Whisper")
    transcription = s.get("transcription", "")
    if transcription:
        st.text_area("Texto detectado", transcription, height=120)
        st.caption(f"{s.get('transcription_words', 0)} palabras · Idioma: {s.get('detected_language', '?')}")
    else:
        st.info("Sin transcripción disponible.")

    st.subheader("Métricas de audio")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality conf.", f"{(s.get('quality_confidence') or 0):.0%}")
    c2.metric("Silence ratio", f"{(s.get('silence_ratio') or 0):.0%}")
    c3.metric("Speech rate", f"{(s.get('speech_rate_wpm') or 0):.0f} wpm")
    c4.metric("Pitch CV", f"{(s.get('pitch_cv') or 0):.3f}")

    c5, c6 = st.columns(2)
    c5.metric("Energy CV", f"{(s.get('energy_cv') or 0):.3f}")
    c6.metric("First latency", f"{(s.get('first_voice_latency') or 0):.1f}s")

    # Voice emotion
    ve = s.get("voice_emotion") or {}
    if isinstance(ve, str):
        try:
            ve = json.loads(ve)
        except Exception:
            ve = {}
    if ve:
        st.subheader("Emoción de voz")
        st.write(f"Dominante: **{ve.get('dominant_emotion', '?')}** (confianza: {ve.get('confidence', 0):.2f})")
        st.caption(f"Motor: {ve.get('primary_engine', '?')}")

    # Voice antispoof
    va = s.get("voice_antispoof") or {}
    if isinstance(va, str):
        try:
            va = json.loads(va)
        except Exception:
            va = {}
    if va:
        st.subheader("Anti-spoof de voz")
        if va.get("synthetic_detected"):
            st.error(f"🚨 Voz sintética detectada (prob: {va.get('synthetic_probability', 0):.0%})")
        else:
            st.success("✅ Sin indicadores de voz sintética.")

# ── COERCION TAB ───────────────────────────────────────────────────────────
with tab_coercion:
    st.subheader("Detección de Coerción")
    if s.get("coercion_detected"):
        phrases = s.get("coercion_matches") or []
        if isinstance(phrases, str):
            try:
                phrases = json.loads(phrases)
            except Exception:
                phrases = []
        st.markdown(
            f'<div class="coercion-alert"><b>🚨 COERCIÓN DETECTADA</b><br>'
            f'Score: {s.get("coercion_score", 0):.2f}<br>'
            f'Frases: {", ".join(phrases[:5]) if phrases else "ver detalles"}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ Sin indicadores de coerción en transcripción.")
        st.metric("Coerción score", f"{(s.get('coercion_score') or 0):.3f}")

# ── AURA TAB ───────────────────────────────────────────────────────────────
with tab_aura:
    st.subheader("Transcript AURA")
    st.caption(f"Turnos: {s.get('aura_turn_count', 0)} · Modelo: {s.get('aura_model_status', '?')} · Fallback: {s.get('aura_used_fallback', True)}")

    transcript = s.get("aura_transcript") or []
    if isinstance(transcript, str):
        try:
            transcript = json.loads(transcript)
        except Exception:
            transcript = []

    if transcript:
        for msg in transcript:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            source = msg.get("source", "")
            ts = msg.get("timestamp", "")[:19].replace("T", " ")
            if role == "aura":
                source_badge = ""
                meta = f'🤖 <b>AURA</b> &nbsp;<span style="color:#6B7280;font-size:0.75rem">{ts}</span>'
                st.markdown(
                    f'<div class="aura-msg"><div class="msg-meta">{meta}</div>{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if source == "push_to_talk":
                    source_badge = '<span class="badge-ptt">🎙 voz</span>'
                elif source == "typed":
                    source_badge = '<span class="badge-typed">⌨ texto</span>'
                else:
                    source_badge = ""
                meta = f'👤 <b>Usuario</b> &nbsp;<span style="color:#6B7280;font-size:0.75rem">{ts}</span> {source_badge}'
                st.markdown(
                    f'<div class="user-msg"><div class="msg-meta">{meta}</div>{content}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Sin transcript de AURA.")

# ── DEEPFACE TAB ───────────────────────────────────────────────────────────
with tab_deepface:
    if deepface_avail:
        st.subheader("Análisis de Emociones DeepFace")
        ep = s.get("emotion_percentages") or {}
        if isinstance(ep, str):
            try:
                ep = json.loads(ep)
            except Exception:
                ep = {}
        if ep:
            df_chart = pd.DataFrame(ep.items(), columns=["Emoción", "Valor"])
            st.bar_chart(df_chart.set_index("Emoción"))
            st.metric("Emoción dominante", s.get("dominant_emotion", "?"))
            st.metric("Ratio stress", f"{(s.get('stress_emotion_ratio') or 0):.0%}")
        else:
            st.info("Sin datos de emociones.")
    else:
        st.info("DeepFace omitido. Pesos redistribuidos en score.")
        st.caption("Para activar DeepFace: instalar requirements-optional.txt y USE_DEEPFACE=1")

# ── RAW JSON TAB ───────────────────────────────────────────────────────────
with tab_raw:
    st.subheader("Risk Breakdown (JSON)")
    with st.expander("Ver risk_breakdown completo"):
        st.json(breakdown)

    aq = s.get("audio_quality") or {}
    if isinstance(aq, str):
        try:
            aq = json.loads(aq)
        except Exception:
            aq = {}
    with st.expander("Ver audio_quality"):
        st.json(aq)

    with st.expander("Ver sesión completa"):
        st.json({k: v for k, v in s.items() if k not in ("aura_transcript", "risk_breakdown", "audio_analysis_json")})

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"OutNova Security v2.4.1 · Dashboard Analista · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
