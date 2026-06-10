import logging
from typing import Optional, Dict, Any

from config import QUALITY_CONFIDENCE_THRESHOLD

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

    if quality_confidence < QUALITY_CONFIDENCE_THRESHOLD:
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


# ── Emotional-Contextual v2 scoring ──────────────────────────────────────────

CONTEXTUAL_WEIGHTS_NORMAL = {
    "emotional_ai": 0.30,
    "behavioral_baseline": 0.20,
    "fraud_triangle": 0.20,
    "narrative_coherence": 0.15,
    "identity_liveness": 0.10,
    "quality": 0.05,
}

CONTEXTUAL_WEIGHTS_ACCESSIBILITY = {
    "emotional_ai": 0.15,
    "behavioral_baseline": 0.15,
    "fraud_triangle": 0.20,
    "narrative_coherence": 0.20,
    "identity_liveness": 0.20,
    "quality": 0.10,
}


def compute_emotional_ai_risk(
    voice_emotion: Dict[str, Any],
    emotion_percentages: Dict[str, Any],
    stress_emotion_ratio: float,
    deepface_available: bool,
) -> dict:
    """Consolidate voice + face emotion signals into a single emotional AI risk score."""
    # Voice stress indicators
    stress_emotions = {"fear", "anger", "disgust", "sad", "miedo", "enojo", "tristeza", "asco"}
    voice_stress = 0.0
    if isinstance(voice_emotion, dict):
        for emotion, prob in voice_emotion.items():
            if any(s in emotion.lower() for s in stress_emotions):
                voice_stress = max(voice_stress, float(prob or 0))

    voice_stress_score = min(100.0, voice_stress * 120.0)

    # Face stress from DeepFace
    face_stress_score = min(100.0, stress_emotion_ratio * 120.0) if deepface_available else 0.0

    # Emotion shift (placeholder — would compare baseline vs sensitive phases)
    emotion_shift_score = 0.0

    # Multimodal consistency: voice and face disagree significantly
    if deepface_available and abs(voice_stress_score - face_stress_score) > 40:
        multimodal_consistency = min(100.0, abs(voice_stress_score - face_stress_score))
    else:
        multimodal_consistency = 0.0

    if deepface_available:
        emotional_ai_risk = (
            voice_stress_score * 0.40
            + face_stress_score * 0.35
            + emotion_shift_score * 0.15
            + multimodal_consistency * 0.10
        )
    else:
        emotional_ai_risk = voice_stress_score * 0.80 + multimodal_consistency * 0.20

    emotional_ai_risk = round(min(100.0, emotional_ai_risk), 2)

    return {
        "emotional_ai_risk": emotional_ai_risk,
        "voice_stress_score": round(voice_stress_score, 2),
        "face_stress_score": round(face_stress_score, 2),
        "emotion_shift_score": round(emotion_shift_score, 2),
        "multimodal_emotion_consistency": round(multimodal_consistency, 2),
    }


def compute_behavioral_baseline_risk(
    baseline_metrics: Dict[str, Any],
    audio_result: Dict[str, Any],
) -> dict:
    """
    Compare current session metrics vs neutral baseline captured in practice phase.
    Returns a behavioral deviation risk score.
    """
    if not baseline_metrics:
        return {"behavioral_baseline_risk": 20.0, "question_metrics": {}}

    baseline_pitch = baseline_metrics.get("avg_pitch_cv", None)
    baseline_speech_rate = baseline_metrics.get("avg_speech_rate_wpm", None)
    baseline_silence = baseline_metrics.get("avg_silence_ratio", None)

    current_pitch = audio_result.get("pitch_cv", None)
    current_speech_rate = audio_result.get("speech_rate_wpm", None)
    current_silence = audio_result.get("silence_ratio", None)

    deviations = []
    if baseline_pitch and current_pitch:
        pitch_dev = abs(current_pitch - baseline_pitch) / max(baseline_pitch, 0.01)
        deviations.append(min(100.0, pitch_dev * 80.0))
    if baseline_speech_rate and current_speech_rate and baseline_speech_rate > 0:
        rate_dev = abs(current_speech_rate - baseline_speech_rate) / baseline_speech_rate
        deviations.append(min(100.0, rate_dev * 100.0))
    if baseline_silence and current_silence:
        silence_dev = abs(current_silence - baseline_silence)
        deviations.append(min(100.0, silence_dev * 150.0))

    behavioral_risk = round(sum(deviations) / len(deviations), 2) if deviations else 20.0

    return {
        "behavioral_baseline_risk": behavioral_risk,
        "question_metrics": {
            "pitch_deviation": deviations[0] if len(deviations) > 0 else None,
            "speech_rate_deviation": deviations[1] if len(deviations) > 1 else None,
            "silence_deviation": deviations[2] if len(deviations) > 2 else None,
        },
    }


def compute_identity_liveness_risk(
    liveness_sub: dict,
    deepfake_flag: bool,
    coercion_detected: bool,
) -> float:
    """Consolidate liveness, deepfake and coercion into identity/liveness risk."""
    base = liveness_sub.get("liveness_spoof_risk_score", 30.0)
    if deepfake_flag:
        base = max(base, 85.0)
    if coercion_detected:
        base = max(base, 70.0)
    return round(min(100.0, base), 2)


def compute_quality_risk(
    audio_quality_sub: dict,
    quality_confidence: float,
) -> float:
    """Map audio quality score to quality risk."""
    aq = audio_quality_sub.get("audio_quality_risk_score", 50.0)
    if quality_confidence < QUALITY_CONFIDENCE_THRESHOLD:
        aq = max(aq, 60.0)
    return round(min(100.0, aq), 2)


def calculate_contextual_risk_score(
    emotional_ai_risk: float,
    behavioral_baseline_risk: float,
    fraud_triangle_risk: float,
    narrative_coherence_risk: float,
    identity_liveness_risk: float,
    quality_risk: float,
    accessibility_mode: bool = False,
    coercion_detected: bool = False,
    deepfake_flag: bool = False,
    quality_confidence: float = 1.0,
    sub_scores: Optional[dict] = None,
) -> dict:
    """
    Calculate final risk score using emotional-contextual v2 model.
    """
    weights = CONTEXTUAL_WEIGHTS_ACCESSIBILITY if accessibility_mode else CONTEXTUAL_WEIGHTS_NORMAL

    risk_score = (
        emotional_ai_risk * weights["emotional_ai"]
        + behavioral_baseline_risk * weights["behavioral_baseline"]
        + fraud_triangle_risk * weights["fraud_triangle"]
        + narrative_coherence_risk * weights["narrative_coherence"]
        + identity_liveness_risk * weights["identity_liveness"]
        + quality_risk * weights["quality"]
    )
    risk_score = round(min(100.0, max(0.0, risk_score)), 2)

    # Recaptura threshold
    if quality_confidence < QUALITY_CONFIDENCE_THRESHOLD:
        status = "Recaptura"
        reason = f"Calidad de audio insuficiente (confianza={quality_confidence:.0%}). Recaptura recomendada."
        recommended_action = "recapture"
    elif deepfake_flag:
        status = "Riesgo crítico"
        reason = "Indicadores de deepfake o falta de liveness detectados."
        recommended_action = "reject"
    elif coercion_detected:
        status = "Riesgo crítico"
        reason = "Coerción detectada en la transcripción."
        recommended_action = "reject"
    elif risk_score < 30:
        status = "Aprobado"
        reason = f"Riesgo bajo ({risk_score:.1f}/100). Verificación satisfactoria."
        recommended_action = "approve"
    elif risk_score < 60:
        status = "Revisión ligera"
        reason = f"Riesgo moderado ({risk_score:.1f}/100). Revisión recomendada."
        recommended_action = "light_review"
    elif risk_score < 80:
        status = "Revisión humana obligatoria"
        reason = f"Riesgo elevado ({risk_score:.1f}/100). Requiere revisión humana."
        recommended_action = "mandatory_review"
    else:
        # Critical: auto-reject only when multiple signals converge
        critical_signals = sum([
            emotional_ai_risk >= 70,
            fraud_triangle_risk >= 60,
            narrative_coherence_risk >= 60,
            identity_liveness_risk >= 60,
        ])
        if critical_signals >= 3:
            status = "Riesgo crítico"
            reason = f"Riesgo crítico ({risk_score:.1f}/100). Múltiples señales de alto riesgo convergentes."
            recommended_action = "reject"
        else:
            status = "Revisión humana obligatoria"
            reason = f"Riesgo alto ({risk_score:.1f}/100). Señales de riesgo requieren revisión experta."
            recommended_action = "mandatory_review"

    breakdown = {
        "model": "emotional_contextual_v2",
        "accessibility_mode": accessibility_mode,
        "weights_used": weights,
        "emotional_ai_risk": round(emotional_ai_risk, 2),
        "behavioral_baseline_risk": round(behavioral_baseline_risk, 2),
        "fraud_triangle_risk": round(fraud_triangle_risk, 2),
        "narrative_coherence_risk": round(narrative_coherence_risk, 2),
        "identity_liveness_risk": round(identity_liveness_risk, 2),
        "quality_risk": round(quality_risk, 2),
        **(sub_scores or {}),
    }

    logger.info(
        f"[SCORE_v2] risk={risk_score:.1f} status={status} "
        f"emotional={emotional_ai_risk:.1f} behavioral={behavioral_baseline_risk:.1f} "
        f"fraud={fraud_triangle_risk:.1f} narrative={narrative_coherence_risk:.1f} "
        f"liveness={identity_liveness_risk:.1f} quality={quality_risk:.1f} "
        f"accessibility={accessibility_mode}"
    )

    return {
        "risk_score": risk_score,
        "risk_breakdown": breakdown,
        "status": status,
        "decision_reason": reason,
        "recommended_action": recommended_action,
        "score_model_version": "emotional_contextual_v2",
    }
