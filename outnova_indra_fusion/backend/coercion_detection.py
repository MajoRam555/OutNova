import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

COERCION_PHRASES = [
    "me estan obligando", "me están obligando",
    "no quiero hacer esto",
    "me dijeron que dijera", "me dijeron que respondiera",
    "estoy bajo amenaza",
    "alguien me esta viendo", "alguien me está viendo",
    "no puedo hablar libremente",
    "me estan forzando", "me están forzando",
    "me obligaron",
    "no puedo decirlo",
    "me estan vigilando", "me están vigilando",
    "estoy siendo presionado",
    "me dijeron que responder",
    "no es mi decision", "no es mi decisión",
    "hay alguien aqui", "hay alguien aquí",
    "me estan diciendo que hacer", "me están diciendo qué hacer",
    "no puedo negarme",
    "tengo miedo",
    "me amenazaron",
    "esto no es voluntario",
    "me estan apuntando", "me están apuntando",
    "me estan escuchando", "me están escuchando",
    "me estan controlando", "me están controlando",
    "no estoy solo",
    "ayuda",
    "auxilio",
    "socorro",
    "me forzaron",
    "me obligan",
]


def _normalize(text: str) -> str:
    text = text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_coercion(transcription: str) -> dict:
    if not transcription or not transcription.strip():
        return {
            "coercion_detected": False,
            "coercion_score": 0.0,
            "matched_phrases": [],
            "explanation": "Sin transcripción disponible.",
        }

    normalized = _normalize(transcription)
    matched = []

    for phrase in COERCION_PHRASES:
        norm_phrase = _normalize(phrase)
        if norm_phrase in normalized:
            matched.append(phrase)

    coercion_detected = len(matched) > 0
    coercion_score = min(1.0, len(matched) * 0.35) if matched else 0.0

    if coercion_detected:
        explanation = f"Frases de coerción detectadas: {', '.join(matched[:3])}"
    else:
        explanation = "Sin indicadores de coerción detectados."

    logger.info(f"Coercion check — detected={coercion_detected}, matches={len(matched)}")

    return {
        "coercion_detected": coercion_detected,
        "coercion_score": round(coercion_score, 4),
        "matched_phrases": matched,
        "explanation": explanation,
    }
