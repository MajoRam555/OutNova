import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Official weights
WEIGHTS_WITH_DEEPFACE = {
    "voice_behavioral": 0.45,
    "liveness_spoof": 0.25,
    "audio_quality": 0.15,
    "multimodal": 0.10,
    "deepface_emotion": 0.05,
}

WEIGHTS_NO_DEEPFACE = {
    "voice_behavioral": 0.47,
    "liveness_spoof": 0.26,
    "audio_quality": 0.16,
    "multimodal": 0.11,
    "deepface_emotion": 0.0,
}


def compute_liveness_spoof_score(
    deepfake_flag: bool,
    total_blinks: int,
    blink_rate: float,
    blink_cv: float,
    face_ratio: float,
    eye_ratio: float,
    voice_antispoof: dict,
    duration_sec: float,
) -> dict:
    sub = {}

    # blink_liveness_score
    if deepfake_flag:
        sub["blink_liveness_score"] = 95.0
    elif duration_sec > 30 and total_blinks == 0:
        sub["blink_liveness_score"] = 90.0
    elif blink_rate < 3.0 and duration_sec > 20:
        sub["blink_liveness_score"] = 70.0
    elif blink_rate > 50.0:
        sub["blink_liveness_score"] = 65.0
    elif 8 <= blink_rate <= 25:
        sub["blink_liveness_score"] = 5.0
    else:
        sub["blink_liveness_score"] = 20.0

    # voice_spoof_score
    if voice_antispoof.get("synthetic_detected"):
        prob = voice_antispoof.get("synthetic_probability", 0.6)
        sub["voice_spoof_score"] = min(95.0, prob * 100.0)
    else:
        sub["voice_spoof_score"] = 5.0

    # face_video_integrity_score
    if face_ratio < 0.20:
        sub["face_video_integrity_score"] = 80.0
    elif face_ratio < 0.50:
        sub["face_video_integrity_score"] = 40.0
    elif eye_ratio < 0.30:
        sub["face_video_integrity_score"] = 35.0
    else:
        sub["face_video_integrity_score"] = 5.0

    # challenge_response_score (stub)
    sub["challenge_response_score"] = 0.0

    liveness_spoof = (
        sub["blink_liveness_score"] * 0.40
        + sub["voice_spoof_score"] * 0.25
        + sub["face_video_integrity_score"] * 0.25
        + sub["challenge_response_score"] * 0.10
    )
    sub["liveness_spoof_risk_score"] = round(liveness_spoof, 2)
    return sub


def compute_audio_quality_score(quality_dict: dict, silence_ratio: float,
                                  clipping_ratio: float, quality_confidence: float) -> dict:
    sub = {}
    sub["low_volume_score"] = quality_dict.get("low_volume_score", 50.0)
    sub["noise_score"] = quality_dict.get("noise_score", 50.0)
    sub["clipping_distortion_score"] = quality_dict.get("clipping_distortion_score", 0.0)
    sub["missing_audio_score"] = quality_dict.get("missing_audio_score", 50.0)

    aqr = quality_dict.get("audio_quality_risk_score")
    if aqr is not None:
        sub["audio_quality_risk_score"] = aqr
    else:
        sub["audio_quality_risk_score"] = (
            sub["low_volume_score"] * 0.30
            + sub["noise_score"] * 0.25
            + sub["clipping_distortion_score"] * 0.20
            + sub["missing_audio_score"] * 0.25
        )
    return sub


def compute_multimodal_score(
    transcription_words: int,
    aura_turn_count: int,
    face_ratio: float,
    silence_ratio: float,
    duration_sec: float,
) -> dict:
    sub = {}

    # audio_video_sync: if no face but there is audio, inconsistency
    if face_ratio < 0.20 and transcription_words > 10:
        sub["audio_video_sync_score"] = 65.0
    else:
        sub["audio_video_sync_score"] = 5.0

    # speaker_face_consistency (heuristic)
    sub["speaker_face_consistency_score"] = 10.0 if face_ratio > 0.5 else 45.0

    # response_timing_consistency
    if aura_turn_count > 0 and transcription_words == 0:
        sub["response_timing_consistency_score"] = 70.0
    elif aura_turn_count == 0 and transcription_words > 20:
        sub["response_timing_consistency_score"] = 40.0
    else:
        sub["response_timing_consistency_score"] = 10.0

    # environment_consistency (heuristic based on silence)
    if silence_ratio > 0.80:
        sub["environment_consistency_score"] = 50.0
    else:
        sub["environment_consistency_score"] = 5.0

    multimodal = (
        sub["audio_video_sync_score"] * 0.30
        + sub["speaker_face_consistency_score"] * 0.30
        + sub["response_timing_consistency_score"] * 0.25
        + sub["environment_consistency_score"] * 0.15
    )
    sub["multimodal_consistency_risk_score"] = round(multimodal, 2)
    return sub


def compute_deepface_emotion_score(stress_emotion_ratio: float, deepface_available: bool) -> float:
    if not deepface_available:
        return 0.0
    return min(100.0, stress_emotion_ratio * 120.0)


def calculate_final_score(
    voice_behavioral_risk_score: float,
    voice_behavioral_sub: dict,
    liveness_sub: dict,
    audio_quality_sub: dict,
    multimodal_sub: dict,
    deepface_emotion_score: float,
    deepface_available: bool,
    quality_confidence: float,
    coercion_detected: bool,
    deepfake_flag: bool,
) -> dict:
    """
    Calculates final risk_score (0-100) and decision.
    """
    weights = WEIGHTS_WITH_DEEPFACE if deepface_available else WEIGHTS_NO_DEEPFACE

    liveness_score = liveness_sub.get("liveness_spoof_risk_score", 50.0)
    audio_quality_score = audio_quality_sub.get("audio_quality_risk_score", 50.0)
    multimodal_score = multimodal_sub.get("multimodal_consistency_risk_score", 50.0)

    risk_score = (
        voice_behavioral_risk_score * weights["voice_behavioral"]
        + liveness_score * weights["liveness_spoof"]
        + audio_quality_score * weights["audio_quality"]
        + multimodal_score * weights["multimodal"]
        + deepface_emotion_score * weights["deepface_emotion"]
    )
    risk_score = round(min(100.0, max(0.0, risk_score)), 2)

    breakdown = {
        "weights_used": weights,
        "voice_behavioral_risk_score": round(voice_behavioral_risk_score, 2),
        "liveness_spoof_risk_score": round(liveness_score, 2),
        "audio_quality_risk_score": round(audio_quality_score, 2),
        "multimodal_consistency_risk_score": round(multimodal_score, 2),
        "deepface_emotion_risk_score": round(deepface_emotion_score, 2),
        "deepface_available": deepface_available,
        "voice_behavioral_sub": voice_behavioral_sub,
        "liveness_sub": liveness_sub,
        "audio_quality_sub": audio_quality_sub,
        "multimodal_sub": multimodal_sub,
    }

    # Decision logic
    status = "Pendiente"
    reason = ""

    if quality_confidence < 0.70:
        status = "Recaptura"
        reason = f"Calidad de audio insuficiente (confianza={quality_confidence:.0%}). Recaptura recomendada."
    elif coercion_detected:
        status = "Rechazado"
        reason = "Coerción detectada en la transcripción."
    elif deepfake_flag:
        status = "Rechazado"
        reason = "Indicadores de deepfake o falta de liveness detectados."
    elif risk_score < 30:
        status = "Aprobado"
        reason = f"Score de riesgo bajo ({risk_score:.1f}/100). Verificación satisfactoria."
    elif risk_score < 85:
        status = "Pendiente"
        reason = f"Score de riesgo moderado ({risk_score:.1f}/100). Requiere revisión manual."
    else:
        status = "Rechazado"
        reason = f"Score de riesgo alto ({risk_score:.1f}/100). Verificación rechazada."

    logger.info(f"[SCORE] risk_score={risk_score:.1f}, status={status}, coercion={coercion_detected}, deepfake={deepfake_flag}")

    return {
        "risk_score": risk_score,
        "risk_breakdown": breakdown,
        "status": status,
        "decision_reason": reason,
    }
