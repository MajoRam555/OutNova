"""
OutNova Indra Fusion — Dashboard Streamlit para analistas
"""
import json
from datetime import datetime

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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ── Light theme override ───────────────────────────────────────────────── */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main > div,
[data-testid="stVerticalBlock"] {
    background: #F8FAFC !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
*  { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }
p, span, div, label, li { color: #374151 !important; }
h1,h2,h3,h4,h5,h6 { color: #111827 !important; font-weight:700 !important; }
a { color: #2563EB !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display:none !important; }
[data-testid="stHeader"] { background:transparent !important; height:0 !important; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] * { color: #374151 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #111827 !important; }
[data-testid="stSidebar"] hr { border-color: #E5E7EB !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #2563EB !important; color: #FFFFFF !important;
    border: none !important; border-radius:8px !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background:#1D4ED8 !important; }

/* ── Tabs ───────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background:#F1F5F9 !important; border-radius:10px !important;
    padding:4px !important; border:1px solid #E5E7EB !important; gap:2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius:8px !important; color:#6B7280 !important;
    font-size:13px !important; font-weight:500 !important;
    padding:6px 14px !important; background:transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background:#FFFFFF !important; color:#2563EB !important;
    box-shadow:0 1px 3px rgba(0,0,0,.08) !important; font-weight:600 !important;
}
[data-testid="stTabContent"] { background:transparent !important; padding-top:14px !important; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius:8px !important; font-size:13px !important; font-weight:500 !important;
    padding:8px 16px !important; transition:all .15s ease !important;
    border:1px solid #E5E7EB !important; background:#FFFFFF !important;
    color:#374151 !important; box-shadow:0 1px 2px rgba(0,0,0,.04) !important; width:100% !important;
}
.stButton > button:hover { background:#F1F5F9 !important; border-color:#D1D5DB !important; }

/* Action button colors by column (approve / reject / recapture / pending) */
[data-testid="column"]:nth-child(1) .stButton > button {
    background:#F0FDF4 !important; color:#166534 !important; border-color:#BBF7D0 !important; }
[data-testid="column"]:nth-child(1) .stButton > button:hover { background:#DCFCE7 !important; }
[data-testid="column"]:nth-child(2) .stButton > button {
    background:#FEF2F2 !important; color:#991B1B !important; border-color:#FECACA !important; }
[data-testid="column"]:nth-child(2) .stButton > button:hover { background:#FEE2E2 !important; }
[data-testid="column"]:nth-child(3) .stButton > button {
    background:#FFFBEB !important; color:#92400E !important; border-color:#FDE68A !important; }
[data-testid="column"]:nth-child(3) .stButton > button:hover { background:#FEF3C7 !important; }
[data-testid="column"]:nth-child(4) .stButton > button {
    background:#EFF6FF !important; color:#1D4ED8 !important; border-color:#BFDBFE !important; }
[data-testid="column"]:nth-child(4) .stButton > button:hover { background:#DBEAFE !important; }

/* ── TextArea ───────────────────────────────────────────────────────────── */
.stTextArea textarea {
    background:#F8FAFC !important; border:1px solid #E5E7EB !important;
    border-radius:8px !important; color:#111827 !important; font-size:13px !important;
}
.stTextArea label { color:#6B7280 !important; font-size:12px !important; }

/* ── Alerts ─────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius:10px !important; border:1px solid !important; }

/* ── Divider ────────────────────────────────────────────────────────────── */
hr { border:none !important; border-top:1px solid #E5E7EB !important; margin:14px 0 !important; }

/* ── Caption ────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p { color:#9CA3AF !important; font-size:11px !important; }

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background:#FFFFFF !important; border:1px solid #E5E7EB !important;
    border-radius:10px !important; box-shadow:0 1px 2px rgba(0,0,0,.04) !important;
}
[data-testid="stExpander"] summary { color:#374151 !important; font-size:13px !important; }

/* ════════════════════════════════════
   OUTNOVA COMPONENTS
════════════════════════════════════ */
.on-header {
    background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px;
    padding:14px 22px; display:flex; align-items:center; justify-content:space-between;
    box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:20px;
}
.on-logo-mark {
    display:inline-flex; align-items:center; justify-content:center;
    width:34px; height:34px; background:#2563EB; border-radius:8px;
    font-size:12px; font-weight:700; color:#fff !important; letter-spacing:.02em; flex-shrink:0;
}
.on-logo-text { font-size:16px; font-weight:700; color:#111827 !important; letter-spacing:-.01em; }
.on-status-dot {
    display:inline-block; width:8px; height:8px;
    background:#16A34A; border-radius:50%;
}
.on-status-lbl { font-size:12px; font-weight:500; color:#6B7280 !important; }
.on-session-badge {
    font-size:11px; font-weight:500; color:#6B7280 !important;
    background:#F1F5F9; border:1px solid #E5E7EB; padding:3px 10px;
    border-radius:999px; font-family:'Courier New',monospace; letter-spacing:.03em;
}
.on-card {
    background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px;
    padding:20px 22px; box-shadow:0 1px 3px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.04);
    margin-bottom:14px;
}
.on-card-lg {
    background:#FFFFFF; border:1px solid #E5E7EB; border-radius:20px;
    padding:24px 26px; box-shadow:0 4px 24px rgba(0,0,0,.07),0 2px 8px rgba(0,0,0,.04);
    margin-bottom:18px;
}
.on-lbl {
    font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:.07em; color:#9CA3AF !important; margin-bottom:8px;
}
.on-card-title { font-size:16px; font-weight:700; color:#111827 !important; letter-spacing:-.01em; margin-bottom:2px; }
.on-card-sub   { font-size:12px; color:#6B7280 !important; }

.on-mc {
    background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px;
    padding:15px 16px; box-shadow:0 1px 3px rgba(0,0,0,.05);
}
.on-mc-lbl { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.07em; color:#9CA3AF !important; margin-bottom:7px; }
.on-mc-val { font-size:24px; font-weight:800; color:#111827 !important; line-height:1; margin-bottom:4px; font-variant-numeric:tabular-nums; }
.on-mc-val.c-ok  { color:#16A34A !important; }
.on-mc-val.c-mid { color:#D97706 !important; }
.on-mc-val.c-bad { color:#DC2626 !important; }
.on-mc-val.c-blu { color:#2563EB !important; }
.on-mc-sub { font-size:11px; color:#9CA3AF !important; }

.on-badge { display:inline-flex; align-items:center; gap:5px; padding:4px 11px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid; }
.on-ok  { background:#F0FDF4; color:#16A34A !important; border-color:#BBF7D0; }
.on-mid { background:#FFFBEB; color:#D97706 !important; border-color:#FDE68A; }
.on-bad { background:#FEF2F2; color:#DC2626 !important; border-color:#FECACA; }
.on-blu { background:#EFF6FF; color:#2563EB !important; border-color:#BFDBFE; }
.on-neu { background:#F1F5F9; color:#6B7280 !important; border-color:#E5E7EB; }

.on-risk { font-size:36px; font-weight:800; line-height:1; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
.on-risk.c-ok  { color:#16A34A !important; }
.on-risk.c-mid { color:#D97706 !important; }
.on-risk.c-bad { color:#DC2626 !important; }
.on-risk.c-non { color:#9CA3AF !important; }

/* Chat bubbles */
.on-chat { display:flex; flex-direction:column; gap:10px; padding:4px 2px; }
.on-msg-a { max-width:86%; align-self:flex-start; }
.on-msg-u { max-width:86%; align-self:flex-end; margin-left:auto; }
.on-meta  { font-size:10px; color:#9CA3AF !important; margin-bottom:4px; display:flex; align-items:center; gap:6px; }
.on-meta-r { justify-content:flex-end; }
.on-bbl   { border-radius:12px; padding:10px 14px; font-size:13px; line-height:1.55; }
.on-bbl-a { background:#EFF6FF; border:1px solid #BFDBFE; border-bottom-left-radius:4px; color:#1E3A5F !important; }
.on-bbl-t { background:#F1F5F9; border:1px solid #E5E7EB; border-bottom-right-radius:4px; color:#111827 !important; }
.on-bbl-p { background:#F0FDF4; border:1px solid #BBF7D0; border-bottom-right-radius:4px; color:#166534 !important; }
.on-mb-ptt   { background:#DCFCE7; color:#166534 !important; border:1px solid #86EFAC; border-radius:999px; padding:1px 7px; font-size:10px; font-weight:600; }
.on-mb-typed { background:#EFF6FF; color:#1D4ED8 !important; border:1px solid #BFDBFE; border-radius:999px; padding:1px 7px; font-size:10px; font-weight:600; }

.on-coercion {
    background:#FEF2F2; border:1px solid #FECACA;
    border-radius:12px; padding:16px 18px; color:#991B1B !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def status_cls(status: str) -> str:
    return {"Aprobado": "on-ok", "Rechazado": "on-bad",
            "Recaptura": "on-mid", "Procesando": "on-blu"}.get(status, "on-neu")

def risk_cls(score) -> str:
    if score is None: return "c-non"
    if score < 30: return "c-ok"
    if score < 85: return "c-mid"
    return "c-bad"

def mc(label: str, value: str, sub: str = "", val_cls: str = "") -> str:
    vc = f"on-mc-val {val_cls}" if val_cls else "on-mc-val"
    s = f'<div class="on-mc-sub">{sub}</div>' if sub else ""
    return f'<div class="on-mc"><div class="on-mc-lbl">{label}</div><div class="{vc}">{value}</div>{s}</div>'

def badge(text: str, cls: str) -> str:
    return f'<span class="on-badge {cls}">{text}</span>'

def chat_html(msgs: list) -> str:
    if not msgs:
        return '<p style="color:#9CA3AF;font-size:13px;">Sin transcript disponible.</p>'
    parts = ['<div class="on-chat">']
    for m in msgs:
        role    = m.get("role", "?")
        content = m.get("content", "")
        source  = m.get("source", "")
        ts      = m.get("timestamp", "")[:19].replace("T", " ")
        if role == "aura":
            parts.append(f'''<div class="on-msg-a">
  <div class="on-meta">🤖 <b style="color:#2563EB">AURA</b> &nbsp; {ts}</div>
  <div class="on-bbl on-bbl-a">{content}</div>
</div>''')
        else:
            if source == "push_to_talk":
                mini = '<span class="on-mb-ptt">🎙 PTT</span>'
                bbl  = "on-bbl-p"
            else:
                mini = '<span class="on-mb-typed">⌨ texto</span>'
                bbl  = "on-bbl-t"
            parts.append(f'''<div class="on-msg-u">
  <div class="on-meta on-meta-r">{ts} &nbsp; 👤 <b style="color:#374151">Usuario</b> {mini}</div>
  <div class="on-bbl {bbl}">{content}</div>
</div>''')
    parts.append('</div>')
    return "\n".join(parts)


# ── API ───────────────────────────────────────────────────────────────────────
def fetch_sessions(status_filter=None, limit=100, offset=0):
    params = {"limit": limit, "offset": offset}
    if status_filter:
        params["status"] = status_filter
    try:
        r = requests.get(f"{API_BASE}/sessions", params=params, timeout=5)
        return r.json()
    except Exception as e:
        st.error(f"No se pudo conectar al backend: {e}")
        return {"sessions": [], "total": 0}

def fetch_session(sid: int):
    try:
        return requests.get(f"{API_BASE}/sessions/{sid}", timeout=5).json()
    except Exception:
        return None

def patch_status(sid: int, status: str, reason: str = "") -> bool:
    try:
        r = requests.patch(f"{API_BASE}/sessions/{sid}/status",
                           json={"status": status, "reason": reason}, timeout=5)
        return r.ok
    except Exception:
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 18px;">
        <div class="on-logo-mark">ON</div>
        <div>
            <div class="on-logo-text">OutNova</div>
            <div style="font-size:10px;color:#9CA3AF !important;margin-top:1px;">Dashboard Analista</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="on-lbl">Filtrar por estado</div>', unsafe_allow_html=True)
    status_filter = st.selectbox("Estado", ["Todos","Pendiente","Procesando","Aprobado","Rechazado","Recaptura"],
                                 label_visibility="collapsed")
    if st.button("↺  Actualizar lista", use_container_width=True):
        st.rerun()

    st.divider()

    filter_val = None if status_filter == "Todos" else status_filter
    data     = fetch_sessions(status_filter=filter_val)
    sessions = data.get("sessions", [])
    total    = data.get("total", 0)

    st.markdown(f'<div style="font-size:11px;color:#9CA3AF !important;margin-bottom:10px;">{total} sesiones totales</div>',
                unsafe_allow_html=True)

    if not sessions:
        st.info("Sin sesiones disponibles.")
        st.stop()

    st.markdown('<div class="on-lbl">Seleccionar sesión</div>', unsafe_allow_html=True)
    options = {
        f"#{s['id']} · {s['status']} · {(s.get('risk_score') or '?')} pts · {s['created_at'][:10]}": s["id"]
        for s in sessions
    }
    selected_label = st.radio("Sesión", list(options.keys()), label_visibility="collapsed")
    selected_id    = options[selected_label]


# ── Load session ──────────────────────────────────────────────────────────────
s = fetch_session(selected_id)
if not s:
    st.error("No se pudo cargar la sesión.")
    st.stop()

score        = s.get("risk_score")
status       = s.get("status", "Pendiente")
short_id     = (s.get("client_session_id") or "—")[:8].upper()
reason       = s.get("decision_reason") or s.get("status_reason") or ""
r_cls        = risk_cls(score)
s_cls        = status_cls(status)

breakdown = s.get("risk_breakdown") or {}
if isinstance(breakdown, str):
    try:    breakdown = json.loads(breakdown)
    except: breakdown = {}

deepface_avail = bool(s.get("deepface_available"))


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="on-header">
  <div style="display:flex;align-items:center;gap:10px;">
    <div class="on-logo-mark">ON</div>
    <div><div class="on-logo-text">OutNova</div></div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div class="on-status-dot"></div>
    <span class="on-status-lbl">Dashboard activo</span>
  </div>
  <div><span class="on-session-badge">ID: {short_id}</span></div>
</div>
""", unsafe_allow_html=True)


# ── Session card ──────────────────────────────────────────────────────────────
score_str = f"{score:.1f}" if score is not None else "—"
reason_row = (f'<div style="margin-top:12px;padding:10px 14px;background:#F8FAFC;'
              f'border:1px solid #E5E7EB;border-radius:8px;font-size:13px;">{reason}</div>'
              if reason else "")

st.markdown(f"""
<div class="on-card-lg">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:14px;">
    <div>
      <div class="on-lbl">Sesión seleccionada</div>
      <div class="on-card-title">Dashboard INDRA · #{s['id']}</div>
      <div class="on-card-sub">Panel de revisión biométrica y conversacional</div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        {badge(status, s_cls)}
        <span style="font-size:11px;color:#9CA3AF;">
          ID: <code style="background:#F1F5F9;padding:1px 6px;border-radius:4px;color:#374151;font-size:11px;">
            {s.get('client_session_id','—')[:24]}</code>
        </span>
        <span style="font-size:11px;color:#9CA3AF;">{s.get('created_at','—')[:19].replace('T',' ')}</span>
      </div>
    </div>
    <div style="text-align:right;">
      <div class="on-risk {r_cls}">{score_str}<span style="font-size:16px;font-weight:500">/100</span></div>
      <div style="font-size:11px;color:#9CA3AF;margin-top:4px;">
        {'Bajo riesgo' if r_cls=='c-ok' else 'Riesgo medio' if r_cls=='c-mid' else 'Alto riesgo' if r_cls=='c-bad' else 'Sin score'}
      </div>
    </div>
  </div>
  {reason_row}
</div>
""", unsafe_allow_html=True)


# ── Manual actions ────────────────────────────────────────────────────────────
st.markdown('<div class="on-lbl">Decisión manual</div>', unsafe_allow_html=True)
ca, cb, cc, cd = st.columns(4)
with ca:
    if st.button("✅ Aprobar", use_container_width=True):
        if patch_status(selected_id, "Aprobado", "Aprobado manualmente."): st.rerun()
with cb:
    if st.button("❌ Rechazar", use_container_width=True):
        if patch_status(selected_id, "Rechazado", "Rechazado manualmente."): st.rerun()
with cc:
    if st.button("🔁 Recaptura", use_container_width=True):
        if patch_status(selected_id, "Recaptura", "Enviado a recaptura."): st.rerun()
with cd:
    if st.button("⏳ Pendiente", use_container_width=True):
        if patch_status(selected_id, "Pendiente", "Marcado pendiente."): st.rerun()


# ── Metrics row ───────────────────────────────────────────────────────────────
st.markdown('<div style="margin-top:6px;"></div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)

qc        = (s.get("quality_confidence") or 0)
coercion  = bool(s.get("coercion_detected"))
df_flag   = bool(s.get("deepfake_flag"))
turns     = s.get("aura_turn_count") or 0

with m1: st.markdown(mc("Risk Score", score_str, "de 100 pts", r_cls), unsafe_allow_html=True)
with m2: st.markdown(mc("Estado", status, "", "c-ok" if status=="Aprobado" else "c-bad" if status=="Rechazado" else "c-mid" if status=="Recaptura" else "c-blu"), unsafe_allow_html=True)
with m3: st.markdown(mc("Quality Conf.", f"{qc:.0%}", "umbral 70%", "c-ok" if qc>=0.7 else "c-bad"), unsafe_allow_html=True)
with m4: st.markdown(mc("Coerción", "Sí" if coercion else "No", f"score {(s.get('coercion_score') or 0):.3f}", "c-bad" if coercion else "c-ok"), unsafe_allow_html=True)
with m5: st.markdown(mc("Deepfake", "Detectado" if df_flag else "No", (s.get("deepfake_reason") or "")[:28], "c-bad" if df_flag else "c-ok"), unsafe_allow_html=True)
with m6: st.markdown(mc("Turnos AURA", str(turns), s.get("aura_model_status") or "?", "c-blu"), unsafe_allow_html=True)


# ── Risk distribution ─────────────────────────────────────────────────────────
st.markdown('<div style="margin-top:4px;"></div>', unsafe_allow_html=True)
col_d, col_s = st.columns([1, 2])

donut_labels = ["Voice Behavioral", "Liveness & Spoof", "Audio Quality", "Multimodal"]
donut_values = [
    (breakdown.get("voice_behavioral_risk_score") or s.get("voice_behavioral_risk_score") or 0),
    (breakdown.get("liveness_spoof_risk_score")   or s.get("liveness_spoof_risk_score")   or 0),
    (breakdown.get("audio_quality_risk_score")    or s.get("audio_quality_risk_score")    or 0),
    (breakdown.get("multimodal_consistency_risk_score") or s.get("multimodal_consistency_risk_score") or 0),
]
if deepface_avail:
    donut_labels.append("DeepFace")
    donut_values.append(breakdown.get("deepface_emotion_risk_score") or s.get("deepface_emotion_risk_score") or 0)

with col_d:
    st.markdown('<div class="on-card"><div class="on-card-title">Distribución de Riesgo</div><div class="on-card-sub" style="margin-bottom:8px;">Desglose por componente</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Pie(
        labels=donut_labels, values=donut_values, hole=0.55,
        textinfo="label+percent",
        marker_colors=["#2563EB","#DC2626","#D97706","#1D4ED8","#F43F5E"],
        textfont=dict(family="Inter, system-ui", size=11),
    ))
    fig.update_layout(showlegend=False, margin=dict(t=6,b=6,l=6,r=6), height=250,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui", color="#374151"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_s:
    st.markdown('<div class="on-card"><div class="on-card-title">Sub-scores</div><div class="on-card-sub" style="margin-bottom:12px;">Puntuación por componente</div>', unsafe_allow_html=True)
    sa, sb = st.columns(2)
    with sa:
        st.markdown(mc("Voice Behavioral", f"{(s.get('voice_behavioral_risk_score') or 0):.1f}", "peso 45%"), unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        st.markdown(mc("Audio Quality", f"{(s.get('audio_quality_risk_score') or 0):.1f}", "peso 15%"), unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        st.markdown(mc("Coerción score", f"{(s.get('coercion_score') or 0):.3f}", ""), unsafe_allow_html=True)
    with sb:
        st.markdown(mc("Liveness & Spoof", f"{(s.get('liveness_spoof_risk_score') or 0):.1f}", "peso 25%"), unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        st.markdown(mc("Multimodal", f"{(s.get('multimodal_consistency_risk_score') or 0):.1f}", "peso 10%"), unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        st.markdown(mc("Quality conf.", f"{qc:.0%}", "umbral 70%", "c-ok" if qc>=0.7 else "c-bad"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Detail tabs ───────────────────────────────────────────────────────────────
st.markdown('<div style="margin-top:4px;"></div>', unsafe_allow_html=True)
tv, ta, tco, tau, tdf, traw = st.tabs(["🎥 Video/Liveness","🎙 Audio","⚠️ Coerción","💬 AURA","🧠 DeepFace","🔧 Raw"])

# ── VIDEO ──────────────────────────────────────────────────────────────────────
with tv:
    blink_rate = (s.get("blink_rate_per_min") or 0)
    bk_cls = "c-ok" if 8<=blink_rate<=25 else "c-mid" if blink_rate>0 else "c-non"

    st.markdown('<div class="on-card"><div class="on-card-title">Liveness & Parpadeos</div></div>', unsafe_allow_html=True)
    vc1,vc2,vc3,vc4,vc5 = st.columns(5)
    with vc1: st.markdown(mc("Total parpadeos", str(s.get("total_blinks") or 0)), unsafe_allow_html=True)
    with vc2: st.markdown(mc("Rate / min", f"{blink_rate:.1f}", "normal 8–25", bk_cls), unsafe_allow_html=True)
    with vc3: st.markdown(mc("Cara detectada", f"{(s.get('face_detected_ratio') or 0):.0%}"), unsafe_allow_html=True)
    with vc4: st.markdown(mc("Ojos detectados", f"{(s.get('eye_detected_ratio') or 0):.0%}"), unsafe_allow_html=True)
    with vc5: st.markdown(mc("Duración video", f"{(s.get('video_duration_sec') or 0):.1f}s"), unsafe_allow_html=True)

    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    if s.get("deepfake_flag"):
        st.error(f"🚨 Deepfake/No-Liveness: {s.get('deepfake_reason','')}")
    else:
        st.success(f"✅ {s.get('deepfake_reason','Sin indicadores de deepfake.')}")

# ── AUDIO ──────────────────────────────────────────────────────────────────────
with ta:
    st.markdown('<div class="on-card-title" style="margin-bottom:10px;">Transcripción Whisper</div>', unsafe_allow_html=True)
    transcription = s.get("transcription", "")
    if transcription:
        st.text_area("Texto", transcription, height=110, label_visibility="collapsed")
        st.caption(f"{s.get('transcription_words',0)} palabras · Idioma: {s.get('detected_language','?')}")
    else:
        st.markdown('<p style="color:#9CA3AF;font-size:13px;">Sin transcripción disponible.</p>', unsafe_allow_html=True)

    st.markdown('<div class="on-card-title" style="margin:16px 0 10px;">Métricas de audio</div>', unsafe_allow_html=True)
    ac1,ac2,ac3,ac4 = st.columns(4)
    with ac1:
        st.markdown(mc("Quality conf.", f"{qc:.0%}", "umbral 70%", "c-ok" if qc>=0.7 else "c-bad"), unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        st.markdown(mc("Energy CV", f"{(s.get('energy_cv') or 0):.3f}"), unsafe_allow_html=True)
    with ac2:
        st.markdown(mc("Silence ratio", f"{(s.get('silence_ratio') or 0):.0%}"), unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        st.markdown(mc("First latency", f"{(s.get('first_voice_latency') or 0):.1f}s"), unsafe_allow_html=True)
    with ac3:
        st.markdown(mc("Speech rate", f"{(s.get('speech_rate_wpm') or 0):.0f} wpm"), unsafe_allow_html=True)
    with ac4:
        st.markdown(mc("Pitch CV", f"{(s.get('pitch_cv') or 0):.3f}"), unsafe_allow_html=True)

    ve = s.get("voice_emotion") or {}
    if isinstance(ve, str):
        try: ve = json.loads(ve)
        except: ve = {}
    if ve:
        st.markdown(f"""
        <div class="on-card" style="margin-top:14px;">
          <div class="on-card-title">Emoción de voz</div>
          <div style="margin-top:8px;font-size:13px;">
            Dominante: <strong style="color:#2563EB;">{ve.get('dominant_emotion','?')}</strong>
            &nbsp;·&nbsp; Confianza: <strong>{ve.get('confidence',0):.2f}</strong>
            &nbsp;·&nbsp; <span style="color:#9CA3AF;">Motor: {ve.get('primary_engine','?')}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    va = s.get("voice_antispoof") or {}
    if isinstance(va, str):
        try: va = json.loads(va)
        except: va = {}
    if va:
        st.markdown('<div class="on-card-title" style="margin:14px 0 8px;">Anti-spoof de voz</div>', unsafe_allow_html=True)
        if va.get("synthetic_detected"):
            st.error(f"🚨 Voz sintética (prob: {va.get('synthetic_probability',0):.0%})")
        else:
            st.success("✅ Sin indicadores de voz sintética.")

# ── COERCION ───────────────────────────────────────────────────────────────────
with tco:
    st.markdown('<div class="on-card-title" style="margin-bottom:12px;">Detección de Coerción</div>', unsafe_allow_html=True)
    if s.get("coercion_detected"):
        phrases = s.get("coercion_matches") or []
        if isinstance(phrases, str):
            try: phrases = json.loads(phrases)
            except: phrases = []
        ph = f'<div style="font-size:12px;margin-top:6px;">Frases: {", ".join(phrases[:5])}</div>' if phrases else ""
        st.markdown(f"""
        <div class="on-coercion">
          <div style="font-weight:700;font-size:15px;margin-bottom:6px;">🚨 COERCIÓN DETECTADA</div>
          <div style="font-size:13px;">Score: <strong>{(s.get('coercion_score') or 0):.2f}</strong></div>
          {ph}
        </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ Sin indicadores de coerción en la transcripción.")
        st.markdown(mc("Coerción score", f"{(s.get('coercion_score') or 0):.3f}", "sin umbral superado", "c-ok"), unsafe_allow_html=True)

# ── AURA ───────────────────────────────────────────────────────────────────────
with tau:
    st.markdown(f"""
    <div class="on-card-title">Transcript AURA</div>
    <div class="on-card-sub" style="margin-bottom:14px;">
        Turnos: {s.get('aura_turn_count',0)} &nbsp;·&nbsp;
        Modelo: {s.get('aura_model_status','?')} &nbsp;·&nbsp;
        Fallback: {s.get('aura_used_fallback',True)}
    </div>
    """, unsafe_allow_html=True)

    transcript = s.get("aura_transcript") or []
    if isinstance(transcript, str):
        try: transcript = json.loads(transcript)
        except: transcript = []
    st.markdown(chat_html(transcript), unsafe_allow_html=True)

# ── DEEPFACE ───────────────────────────────────────────────────────────────────
with tdf:
    st.markdown('<div class="on-card-title" style="margin-bottom:10px;">Análisis DeepFace</div>', unsafe_allow_html=True)
    if deepface_avail:
        ep = s.get("emotion_percentages") or {}
        if isinstance(ep, str):
            try: ep = json.loads(ep)
            except: ep = {}
        if ep:
            fig2 = go.Figure(go.Bar(
                x=list(ep.keys()), y=list(ep.values()),
                marker_color="#2563EB", marker_line_width=0,
            ))
            fig2.update_layout(
                margin=dict(t=8,b=8,l=8,r=8), height=220,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, system-ui", color="#374151"),
                xaxis=dict(gridcolor="#E5E7EB"), yaxis=dict(gridcolor="#E5E7EB"),
            )
            st.plotly_chart(fig2, use_container_width=True)
            df1, df2 = st.columns(2)
            with df1: st.markdown(mc("Emoción dominante", s.get("dominant_emotion","?"), ""), unsafe_allow_html=True)
            with df2: st.markdown(mc("Ratio stress", f"{(s.get('stress_emotion_ratio') or 0):.0%}", ""), unsafe_allow_html=True)
        else:
            st.info("Sin datos de emociones DeepFace.")
    else:
        st.info("DeepFace omitido. Pesos redistribuidos automáticamente.")
        st.caption("Para activar: instalar requirements-optional.txt y USE_DEEPFACE=1")

# ── RAW ────────────────────────────────────────────────────────────────────────
with traw:
    st.markdown('<div class="on-card-title" style="margin-bottom:10px;">Datos técnicos</div>', unsafe_allow_html=True)
    with st.expander("Risk breakdown (JSON)"):
        st.json(breakdown)
    aq = s.get("audio_quality") or {}
    if isinstance(aq, str):
        try: aq = json.loads(aq)
        except: aq = {}
    with st.expander("Audio quality (JSON)"):
        st.json(aq)
    with st.expander("Sesión completa"):
        st.json({k: v for k, v in s.items() if k not in ("aura_transcript","risk_breakdown","audio_analysis_json")})


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:24px;padding:14px 22px;background:#FFFFFF;border:1px solid #E5E7EB;
            border-radius:12px;display:flex;align-items:center;justify-content:space-between;
            box-shadow:0 1px 3px rgba(0,0,0,.05);">
  <div style="display:flex;align-items:center;gap:8px;">
    <div class="on-logo-mark" style="width:24px;height:24px;font-size:9px;">ON</div>
    <span style="font-size:12px;color:#6B7280 !important;">OutNova Security v2.4.1</span>
  </div>
  <span style="font-size:11px;color:#9CA3AF !important;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
</div>
""", unsafe_allow_html=True)
