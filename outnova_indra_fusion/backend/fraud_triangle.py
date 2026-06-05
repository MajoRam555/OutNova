"""
Fraud triangle analysis module.
Detects pressure, opportunity, and rationalization signals in text.
"""
import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Phrase banks ──────────────────────────────────────────────────────────────

PRESSURE_PHRASES = [
    "me dijeron que tenia que hacerlo", "me dijeron que tenía que hacerlo",
    "era urgente", "no tenia opcion", "no tenía opción",
    "me estan presionando", "me están presionando",
    "me lo pidio alguien", "me lo pidió alguien",
    "me obligaron", "me obligan", "me forzaron",
    "tengo miedo", "tenia miedo", "tenía miedo",
    "no puedo negarme",
    "no puedo hablar libremente",
    "hay alguien aqui", "hay alguien aquí",
    "me estan diciendo", "me están diciendo",
    "me lo ordenaron", "me dijeron que viniera",
    "no es mi decision", "no es mi decisión",
    "necesito el dinero urgente", "es por deuda", "debo mucho dinero",
    "me van a hacer dano", "me van a hacer daño",
    "me amenazaron", "me amenazan",
    "no tuve opcion", "no tuve opción", "no tengo alternativa",
    "es lo que me pidieron", "me lo mandaron hacer",
]

OPPORTUNITY_PHRASES = [
    "el sistema se equivoco", "el sistema se equivocó",
    "nadie se iba a dar cuenta",
    "tenia acceso", "tenía acceso",
    "use los datos", "usé los datos",
    "aproveche la oportunidad", "aproveché la oportunidad",
    "aproveche", "aproveché",
    "solo era una vez", "sólo era una vez",
    "el banco se equivoco", "el banco se equivocó",
    "no afectaba a nadie",
    "habia un hueco", "había un hueco",
    "era facil hacerlo", "era fácil hacerlo",
    "me dieron acceso sin querer", "encontre una forma", "encontré una forma",
    "vi la oportunidad",
    "nadie lo notaria", "nadie lo notaría",
    "el sistema lo permitia", "el sistema lo permitía",
    "podia entrar", "podía entrar",
    "tenia las claves", "tenía las claves",
    "tuve acceso a la cuenta",
    "era dinero que ya era mio", "era dinero que ya era mío",
]

RATIONALIZATION_PHRASES = [
    "todos lo hacen",
    "solo segui ordenes", "sólo seguí órdenes", "solo seguí órdenes",
    "era por necesidad",
    "no afectaba a nadie",
    "no fue tan grave", "no era para tanto",
    "me lo merecia", "me lo merecía",
    "era lo justo",
    "fue sin querer", "fue un error",
    "lo iba a devolver", "iba a devolver el dinero",
    "era temporal",
    "nadie salio perjudicado", "nadie salió perjudicado",
    "es una injusticia lo que me pagan",
    "la empresa me debe", "me deben mucho",
    "ellos tambien roban", "ellos también roban",
    "el banco tiene suficiente", "es una empresa grande",
    "es solo una vez", "fue la unica vez", "fue la única vez",
    "no iba a pasar nada", "era una emergencia",
    "tenia que sobrevivir", "tenía que sobrevivir",
]


def _normalize(text: str) -> str:
    text = text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _score_category(text_norm: str, phrases: list) -> tuple:
    matched = []
    for phrase in phrases:
        if _normalize(phrase) in text_norm:
            matched.append(phrase)
    score = min(100.0, len(matched) * 25.0)
    return round(score, 2), matched


def analyze_fraud_triangle(text: str, aura_transcript: Optional[list] = None) -> dict:
    """
    Analyze transcription + AURA user turns for fraud triangle signals.
    Returns pressure, opportunity, rationalization scores and combined risk.
    """
    full_text = text or ""
    if aura_transcript and isinstance(aura_transcript, list):
        for msg in aura_transcript:
            if msg.get("role") == "user":
                full_text += " " + msg.get("content", "")

    if not full_text.strip():
        return {
            "fraud_triangle_risk": 0.0,
            "pressure_score": 0.0,
            "opportunity_score": 0.0,
            "rationalization_score": 0.0,
            "matched_signals": {"pressure": [], "opportunity": [], "rationalization": []},
            "explanation": "Sin texto disponible para análisis del triángulo del fraude.",
            "confidence": 0.1,
        }

    norm = _normalize(full_text)
    pressure_score, pressure_matches = _score_category(norm, PRESSURE_PHRASES)
    opportunity_score, opportunity_matches = _score_category(norm, OPPORTUNITY_PHRASES)
    rationalization_score, rationalization_matches = _score_category(norm, RATIONALIZATION_PHRASES)

    fraud_risk = (
        pressure_score * 0.45
        + opportunity_score * 0.30
        + rationalization_score * 0.25
    )
    fraud_risk = round(min(100.0, fraud_risk), 2)

    all_matches = pressure_matches + opportunity_matches + rationalization_matches
    if all_matches:
        parts = []
        if pressure_matches:
            parts.append(f"presión ({len(pressure_matches)} señal/es)")
        if opportunity_matches:
            parts.append(f"oportunidad ({len(opportunity_matches)} señal/es)")
        if rationalization_matches:
            parts.append(f"racionalización ({len(rationalization_matches)} señal/es)")
        explanation = f"Señales del triángulo del fraude detectadas: {', '.join(parts)}."
    else:
        explanation = "Sin señales del triángulo del fraude en el texto analizado."

    logger.info(
        f"[FRAUD_TRI] risk={fraud_risk:.1f} "
        f"pressure={pressure_score:.1f} opportunity={opportunity_score:.1f} "
        f"rationalization={rationalization_score:.1f}"
    )

    return {
        "fraud_triangle_risk": fraud_risk,
        "pressure_score": pressure_score,
        "opportunity_score": opportunity_score,
        "rationalization_score": rationalization_score,
        "matched_signals": {
            "pressure": pressure_matches,
            "opportunity": opportunity_matches,
            "rationalization": rationalization_matches,
        },
        "explanation": explanation,
        "confidence": 0.8 if full_text.strip() else 0.1,
    }
