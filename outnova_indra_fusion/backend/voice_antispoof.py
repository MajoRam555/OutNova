import logging

logger = logging.getLogger(__name__)


def analyze_voice_antispoof(audio_path: str, pitch_cv: float = None, energy_cv: float = None,
                             duration_sec: float = None) -> dict:
    """
    Heuristic anti-spoof for synthetic/TTS voice detection.
    Uses pitch_cv and energy_cv as main indicators.
    """
    if duration_sec is not None and duration_sec < 3.0:
        return {
            "method": "librosa_heuristic",
            "synthetic_detected": False,
            "synthetic_probability": 0.0,
            "pitch_cv": pitch_cv,
            "available": False,
            "note": "Audio muy corto para análisis anti-spoof.",
        }

    synthetic_probability = 0.0
    reasons = []

    if pitch_cv is not None:
        if pitch_cv < 0.04:
            synthetic_probability += 0.45
            reasons.append("pitch extremadamente estable (posible TTS)")
        elif pitch_cv < 0.08:
            synthetic_probability += 0.20
            reasons.append("pitch muy bajo (voz posiblemente sintética)")

    if energy_cv is not None:
        if energy_cv < 0.05:
            synthetic_probability += 0.30
            reasons.append("energía completamente plana")
        elif energy_cv < 0.10:
            synthetic_probability += 0.15
            reasons.append("energía muy uniforme")

    synthetic_probability = min(1.0, synthetic_probability)
    synthetic_detected = synthetic_probability >= 0.55

    return {
        "method": "librosa_heuristic",
        "synthetic_detected": synthetic_detected,
        "synthetic_probability": round(synthetic_probability, 4),
        "pitch_cv": pitch_cv,
        "energy_cv": energy_cv,
        "available": True,
        "reasons": reasons,
    }
