"""
Narrative coherence analysis module.
Analyzes AURA conversation transcripts for evasion, contradictions, and incompleteness.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Evasion patterns ──────────────────────────────────────────────────────────

EVASION_PHRASES = [
    "no recuerdo", "no me acuerdo", "no sé", "no se",
    "no puedo decir", "no quiero hablar de eso",
    "eso no importa", "por qué preguntas eso",
    "prefiero no contestar", "no tengo por qué decirte",
    "es privado", "es personal", "no es relevante",
    "no sé de qué hablas", "no entiendo la pregunta",
    "es complicado", "es difícil de explicar",
    "mejor cambiemos el tema", "eso es otro asunto",
    "no me hagas esa pregunta", "no sé cómo explicarlo",
    "no recuerdo bien", "tal vez", "quizás", "quizás sí",
    "puede ser", "depende", "a veces", "no estoy seguro",
    "no estoy segura", "no lo tengo claro",
]

CONTRADICTION_MARKERS = [
    ("siempre", "nunca"),
    ("sí fui", "no fui"),
    ("sí lo hice", "no lo hice"),
    ("lo sé", "no lo sé"),
    ("estaba solo", "estaba acompañado"),
    ("estaba sola", "estaba acompañada"),
    ("fue ayer", "fue hace días"),
    ("fue hoy", "fue ayer"),
    ("trabajo ahí", "no trabajo ahí"),
    ("conozco", "no conozco"),
    ("tengo", "no tengo"),
    ("fui yo", "no fui yo"),
    ("es mío", "no es mío"),
    ("es mía", "no es mía"),
]

INCOMPLETENESS_MARKERS = [
    "...", "este...", "bueno...", "pues...", "o sea...",
    "es que...", "lo que pasa es...", "es que mira...",
    "mmm", "ehh", "uhh", "ahhh",
    "no sé cómo decirlo", "cómo te digo",
    "es largo de explicar", "sería muy largo",
    "la verdad es que", "lo cierto es que",
    "hay cosas que", "algunas cosas",
]

DIRECT_ANSWER_INDICATORS = [
    r"\bsí\b", r"\bno\b", r"\bfui\b", r"\bestuve\b", r"\bera\b",
    r"\bhice\b", r"\btuve\b", r"\btengo\b", r"\bpuedo\b",
]


def _normalize(text: str) -> str:
    import unicodedata
    text = text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\.]", " ", text)).strip()


def _extract_user_turns(aura_transcript: list) -> list:
    return [
        msg.get("content", "")
        for msg in aura_transcript
        if isinstance(msg, dict) and msg.get("role") == "user"
    ]


def _score_evasion(user_turns: list) -> tuple:
    matched = []
    total_turns = len(user_turns)
    if total_turns == 0:
        return 0.0, []

    for turn in user_turns:
        norm = _normalize(turn)
        for phrase in EVASION_PHRASES:
            if phrase in norm and phrase not in matched:
                matched.append(phrase)

    evasion_rate = len(matched) / max(total_turns, 1)
    score = min(100.0, evasion_rate * 200.0 + len(matched) * 8.0)
    return round(score, 2), matched


def _score_contradictions(user_turns: list) -> tuple:
    all_text = " ".join(_normalize(t) for t in user_turns)
    found = []
    for pos, neg in CONTRADICTION_MARKERS:
        if pos in all_text and neg in all_text:
            found.append(f"{pos} / {neg}")
    score = min(100.0, len(found) * 35.0)
    return round(score, 2), found


def _score_incompleteness(user_turns: list) -> tuple:
    matched = []
    short_turns = 0
    for turn in user_turns:
        norm = _normalize(turn)
        word_count = len(norm.split())
        if word_count < 4:
            short_turns += 1
        for marker in INCOMPLETENESS_MARKERS:
            if marker in norm and marker not in matched:
                matched.append(marker)

    short_ratio = short_turns / max(len(user_turns), 1)
    score = min(100.0, short_ratio * 60.0 + len(matched) * 10.0)
    return round(score, 2), matched


def _score_narrative_consistency(user_turns: list) -> float:
    """Higher score = more inconsistent (riskier)."""
    if len(user_turns) < 2:
        return 0.0

    direct_answers = 0
    for turn in user_turns:
        norm = _normalize(turn)
        for pattern in DIRECT_ANSWER_INDICATORS:
            if re.search(pattern, norm):
                direct_answers += 1
                break

    directness_ratio = direct_answers / len(user_turns)
    # Low directness = higher inconsistency risk
    consistency_score = max(0.0, (1.0 - directness_ratio) * 60.0)
    return round(consistency_score, 2)


def analyze_narrative_coherence(
    aura_transcript: Optional[list] = None,
    transcription: str = "",
) -> dict:
    """
    Analyze AURA conversation for narrative coherence signals.
    Returns coherence risk score and sub-component breakdown.
    """
    if not aura_transcript and not transcription.strip():
        return {
            "narrative_coherence_risk": 0.0,
            "narrative_consistency_score": 0.0,
            "contradiction_score": 0.0,
            "evasion_score": 0.0,
            "incompleteness_score": 0.0,
            "matched_signals": {
                "evasion": [], "contradictions": [], "incompleteness": []
            },
            "explanation": "Sin transcripción disponible para análisis narrativo.",
            "confidence": 0.1,
        }

    user_turns = []
    if aura_transcript and isinstance(aura_transcript, list):
        user_turns = _extract_user_turns(aura_transcript)

    if not user_turns and transcription.strip():
        user_turns = [transcription]

    evasion_score, evasion_matches = _score_evasion(user_turns)
    contradiction_score, contradiction_matches = _score_contradictions(user_turns)
    incompleteness_score, incompleteness_matches = _score_incompleteness(user_turns)
    consistency_score = _score_narrative_consistency(user_turns)

    narrative_risk = (
        evasion_score * 0.35
        + contradiction_score * 0.30
        + incompleteness_score * 0.20
        + consistency_score * 0.15
    )
    narrative_risk = round(min(100.0, narrative_risk), 2)

    parts = []
    if evasion_matches:
        parts.append(f"evasión ({len(evasion_matches)} señal/es)")
    if contradiction_matches:
        parts.append(f"contradicción ({len(contradiction_matches)} par/es)")
    if incompleteness_matches:
        parts.append(f"respuestas incompletas ({len(incompleteness_matches)} señal/es)")

    explanation = (
        f"Señales de incoherencia narrativa: {', '.join(parts)}."
        if parts else
        "Sin señales de incoherencia narrativa detectadas."
    )

    logger.info(
        f"[NARRATIVE] risk={narrative_risk:.1f} "
        f"evasion={evasion_score:.1f} contradiction={contradiction_score:.1f} "
        f"incompleteness={incompleteness_score:.1f} consistency={consistency_score:.1f}"
    )

    return {
        "narrative_coherence_risk": narrative_risk,
        "narrative_consistency_score": consistency_score,
        "contradiction_score": contradiction_score,
        "evasion_score": evasion_score,
        "incompleteness_score": incompleteness_score,
        "matched_signals": {
            "evasion": evasion_matches,
            "contradictions": contradiction_matches,
            "incompleteness": incompleteness_matches,
        },
        "explanation": explanation,
        "confidence": 0.75 if len(user_turns) >= 3 else 0.4,
    }
