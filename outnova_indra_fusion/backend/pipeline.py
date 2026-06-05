import json
import logging
from datetime import datetime
from pathlib import Path

from database import SessionLocal, AnalysisSession

logger = logging.getLogger(__name__)


def run_analysis(session_id: int, video_path: str, client_session_id: str):
    """
    Full analysis pipeline. Runs in ThreadPoolExecutor (_ml_executor).
    Tolerant of individual module failures.
    Saves partial results to SQLite even if some modules fail.
    """
    db = SessionLocal()
    try:
        _run_analysis_internal(db, session_id, video_path, client_session_id)
    except Exception as e:
        logger.error(f"[PIPELINE] Error fatal en sesión {session_id}: {e}", exc_info=True)
        try:
            record = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
            if record:
                record.status = "Pendiente"
                record.status_reason = f"Error en pipeline: {str(e)[:200]}"
                record.updated_at = datetime.utcnow().isoformat()
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _run_analysis_internal(db, session_id: int, video_path: str, client_session_id: str):
    logger.info(f"[PIPELINE] Iniciando análisis — session_id={session_id}, video={video_path}")

    record = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if not record:
        logger.error(f"[PIPELINE] Sesión {session_id} no encontrada en DB.")
        return

    # Mark as processing
    record.status = "Procesando"
    record.updated_at = datetime.utcnow().isoformat()
    db.commit()

    video_result = {}
    audio_result = {}
    coercion_result = {"coercion_detected": False, "coercion_score": 0.0, "matched_phrases": []}

    # ── STEP 1: Video Analysis ──────────────────────────────────────────────
    logger.info("[PIPELINE] Paso 1: Análisis de video...")
    try:
        from video_analysis import analyze_video
        video_result = analyze_video(video_path)
        logger.info(f"[PIPELINE] Video OK — blinks={video_result.get('total_blinks')}, deepfake={video_result.get('deepfake_flag')}")
    except Exception as e:
        logger.error(f"[PIPELINE] Video analysis error: {e}")
        video_result = {"errors": [str(e)]}

    # ── STEP 2: Audio Pipeline ─────────────────────────────────────────────
    logger.info("[PIPELINE] Paso 2: Pipeline de audio...")
    try:
        from audio_pipeline import run_audio_pipeline
        duration_sec = video_result.get("video_duration_sec", 0.0)
        audio_result = run_audio_pipeline(video_path, client_session_id, duration_sec=duration_sec)
        logger.info(f"[PIPELINE] Audio OK — words={audio_result.get('transcription_words')}, quality={audio_result.get('quality_confidence', 0):.2f}")
    except Exception as e:
        logger.error(f"[PIPELINE] Audio pipeline error: {e}")
        audio_result = {"errors": [str(e)], "quality_confidence": 0.1, "voice_behavioral_risk_score": 50.0}

    # ── STEP 3: Coercion ───────────────────────────────────────────────────
    logger.info("[PIPELINE] Paso 3: Detección de coerción...")
    try:
        coercion_result = audio_result.get("coercion", {})
        if not coercion_result:
            from coercion_detection import detect_coercion
            coercion_result = detect_coercion(audio_result.get("transcription", ""))
    except Exception as e:
        logger.warning(f"[PIPELINE] Coerción error: {e}")
        coercion_result = {"coercion_detected": False, "coercion_score": 0.0, "matched_phrases": []}

    # ── STEP 3b: Fraud Triangle ────────────────────────────────────────────
    logger.info("[PIPELINE] Paso 3b: Triángulo del fraude...")
    fraud_result = {}
    try:
        from fraud_triangle import analyze_fraud_triangle
        aura_transcript = json.loads(record.aura_transcript or "[]")
        fraud_result = analyze_fraud_triangle(
            text=audio_result.get("transcription", ""),
            aura_transcript=aura_transcript,
        )
        logger.info(f"[PIPELINE] Fraude risk={fraud_result.get('fraud_triangle_risk', 0):.1f}")
    except Exception as e:
        logger.warning(f"[PIPELINE] Fraud triangle error: {e}")
        fraud_result = {"fraud_triangle_risk": 0.0, "pressure_score": 0.0,
                        "opportunity_score": 0.0, "rationalization_score": 0.0,
                        "matched_signals": {}, "explanation": "", "confidence": 0.1}

    # ── STEP 3c: Narrative Coherence ───────────────────────────────────────
    logger.info("[PIPELINE] Paso 3c: Coherencia narrativa...")
    narrative_result = {}
    try:
        from narrative_analysis import analyze_narrative_coherence
        aura_transcript = json.loads(record.aura_transcript or "[]")
        narrative_result = analyze_narrative_coherence(
            aura_transcript=aura_transcript,
            transcription=audio_result.get("transcription", ""),
        )
        logger.info(f"[PIPELINE] Narrativa risk={narrative_result.get('narrative_coherence_risk', 0):.1f}")
    except Exception as e:
        logger.warning(f"[PIPELINE] Narrative analysis error: {e}")
        narrative_result = {"narrative_coherence_risk": 0.0, "narrative_consistency_score": 0.0,
                            "contradiction_score": 0.0, "evasion_score": 0.0,
                            "incompleteness_score": 0.0, "matched_signals": {},
                            "explanation": "", "confidence": 0.1}

    # ── STEP 4: Scoring (emotional-contextual v2) ──────────────────────────
    logger.info("[PIPELINE] Paso 4: Calculando score emocional-contextual v2...")
    liveness_sub = {}
    audio_quality_sub = {}
    multimodal_sub = {}
    voice_behavioral_sub = {}
    try:
        from scoring import (
            compute_liveness_spoof_score,
            compute_audio_quality_score,
            compute_multimodal_score,
            compute_emotional_ai_risk,
            compute_behavioral_baseline_risk,
            compute_identity_liveness_risk,
            compute_quality_risk,
            calculate_contextual_risk_score,
        )

        voice_behavioral_sub = audio_result.get("voice_behavioral_sub_scores", {})
        voice_behavioral_risk_score = audio_result.get("voice_behavioral_risk_score", 50.0)

        liveness_sub = compute_liveness_spoof_score(
            deepfake_flag=video_result.get("deepfake_flag", False),
            total_blinks=video_result.get("total_blinks", 0),
            blink_rate=video_result.get("blink_rate_per_min", 0.0),
            blink_cv=video_result.get("blink_cv", 0.0),
            face_ratio=video_result.get("face_detected_ratio", 0.0),
            eye_ratio=video_result.get("eye_detected_ratio", 0.0),
            voice_antispoof=audio_result.get("voice_antispoof", {}),
            duration_sec=video_result.get("video_duration_sec", 0.0),
        )

        audio_quality_sub = compute_audio_quality_score(
            quality_dict=audio_result.get("audio_quality", {}),
            silence_ratio=audio_result.get("silence_ratio", 1.0),
            clipping_ratio=audio_result.get("clipping_ratio", 0.0),
            quality_confidence=audio_result.get("quality_confidence", 0.1),
        )

        multimodal_sub = compute_multimodal_score(
            transcription_words=audio_result.get("transcription_words", 0),
            aura_turn_count=record.aura_turn_count or 0,
            face_ratio=video_result.get("face_detected_ratio", 0.0),
            silence_ratio=audio_result.get("silence_ratio", 1.0),
            duration_sec=video_result.get("video_duration_sec", 0.0),
        )

        # Emotional AI risk
        try:
            import json as _json
            voice_emotion = _json.loads(audio_result.get("voice_emotion") or "{}")
        except Exception:
            voice_emotion = audio_result.get("voice_emotion", {})
        emotion_ai_result = compute_emotional_ai_risk(
            voice_emotion=voice_emotion,
            emotion_percentages=video_result.get("emotion_percentages", {}),
            stress_emotion_ratio=video_result.get("stress_emotion_ratio", 0.0),
            deepface_available=video_result.get("deepface_available", False),
        )

        # Behavioral baseline risk
        behavioral_result = compute_behavioral_baseline_risk(
            baseline_metrics=audio_result.get("baseline_metrics", {}),
            audio_result=audio_result,
        )

        # Identity/liveness risk (consolidated)
        identity_liveness_risk = compute_identity_liveness_risk(
            liveness_sub=liveness_sub,
            deepfake_flag=video_result.get("deepfake_flag", False),
            coercion_detected=coercion_result.get("coercion_detected", False),
        )

        # Quality risk (consolidated)
        quality_risk_val = compute_quality_risk(
            audio_quality_sub=audio_quality_sub,
            quality_confidence=audio_result.get("quality_confidence", 0.1),
        )

        accessibility_mode = getattr(record, "accessibility_mode", False) or False

        score_result = calculate_contextual_risk_score(
            emotional_ai_risk=emotion_ai_result["emotional_ai_risk"],
            behavioral_baseline_risk=behavioral_result["behavioral_baseline_risk"],
            fraud_triangle_risk=fraud_result.get("fraud_triangle_risk", 0.0),
            narrative_coherence_risk=narrative_result.get("narrative_coherence_risk", 0.0),
            identity_liveness_risk=identity_liveness_risk,
            quality_risk=quality_risk_val,
            accessibility_mode=accessibility_mode,
            coercion_detected=coercion_result.get("coercion_detected", False),
            deepfake_flag=video_result.get("deepfake_flag", False),
            quality_confidence=audio_result.get("quality_confidence", 0.1),
            sub_scores={
                "voice_behavioral_sub": voice_behavioral_sub,
                "liveness_sub": liveness_sub,
                "audio_quality_sub": audio_quality_sub,
                "multimodal_sub": multimodal_sub,
                "emotion_ai_detail": emotion_ai_result,
                "behavioral_detail": behavioral_result,
            },
        )
        # Attach extra fields needed for DB save
        score_result["emotion_ai_result"] = emotion_ai_result
        score_result["behavioral_result"] = behavioral_result
        score_result["identity_liveness_risk"] = identity_liveness_risk
        score_result["quality_risk"] = quality_risk_val

        logger.info(f"[PIPELINE] Score={score_result['risk_score']:.1f}, status={score_result['status']}")

    except Exception as e:
        logger.error(f"[PIPELINE] Scoring error: {e}", exc_info=True)
        score_result = {
            "risk_score": 50.0,
            "risk_breakdown": {"error": str(e)},
            "status": "Pendiente",
            "decision_reason": f"Error en scoring: {str(e)[:100]}",
            "recommended_action": "pending",
            "score_model_version": "emotional_contextual_v2",
            "emotion_ai_result": {},
            "behavioral_result": {},
            "identity_liveness_risk": 30.0,
            "quality_risk": 50.0,
        }
        liveness_sub = {}
        audio_quality_sub = {}
        multimodal_sub = {}
        voice_behavioral_sub = {}

    # ── STEP 5: Save to DB ─────────────────────────────────────────────────
    logger.info("[PIPELINE] Paso 5: Guardando resultados en DB...")
    try:
        _save_results(
            db, record, video_result, audio_result, coercion_result,
            liveness_sub, audio_quality_sub, multimodal_sub, voice_behavioral_sub,
            score_result, fraud_result, narrative_result,
        )
        logger.info(f"[PIPELINE] Resultados guardados — sesión {session_id} => {score_result['status']}")
    except Exception as e:
        logger.error(f"[PIPELINE] DB save error: {e}", exc_info=True)


def _save_results(db, record: AnalysisSession, video: dict, audio: dict,
                   coercion: dict, liveness_sub: dict, audio_quality_sub: dict,
                   multimodal_sub: dict, voice_behavioral_sub: dict, score: dict,
                   fraud: dict = None, narrative: dict = None):
    now = datetime.utcnow().isoformat()
    fraud = fraud or {}
    narrative = narrative or {}

    # Video
    record.video_duration_sec = video.get("video_duration_sec")
    record.total_blinks = video.get("total_blinks")
    record.blink_rate_per_min = video.get("blink_rate_per_min")
    record.blink_cv = video.get("blink_cv")
    record.face_detected_ratio = video.get("face_detected_ratio")
    record.eye_detected_ratio = video.get("eye_detected_ratio")
    record.deepfake_flag = video.get("deepfake_flag", False)
    record.deepfake_reason = video.get("deepfake_reason")
    record.emotion_percentages = json.dumps(video.get("emotion_percentages", {}))
    record.dominant_emotion = video.get("dominant_emotion")
    record.stress_emotion_ratio = video.get("stress_emotion_ratio")
    record.deepface_available = video.get("deepface_available", False)

    # Audio
    record.audio_path = audio.get("audio_path")
    record.transcription = audio.get("transcription", "")
    record.transcription_words = audio.get("transcription_words", 0)
    record.detected_language = audio.get("detected_language")
    record.audio_analysis_json = json.dumps(audio.get("voice_behavioral_sub_scores", {}))
    record.audio_quality = json.dumps(audio.get("audio_quality", {}))
    record.quality_confidence = audio.get("quality_confidence")
    record.silence_ratio = audio.get("silence_ratio")
    record.clipping_ratio = audio.get("clipping_ratio")
    record.noise_level = audio.get("noise_level")
    record.avg_volume = audio.get("avg_volume")
    record.first_voice_latency = audio.get("first_voice_latency")
    record.speech_rate_wpm = audio.get("speech_rate_wpm")
    record.pitch_cv = audio.get("pitch_cv")
    record.energy_cv = audio.get("energy_cv")
    record.voice_emotion = json.dumps(audio.get("voice_emotion", {}))
    record.voice_antispoof = json.dumps(audio.get("voice_antispoof", {}))
    record.speaker_verification = json.dumps(audio.get("speaker_verification", {}))

    # Coercion
    record.coercion_detected = coercion.get("coercion_detected", False)
    record.coercion_score = coercion.get("coercion_score", 0.0)
    record.coercion_matches = json.dumps(coercion.get("matched_phrases", []))

    # Score
    record.risk_score = score.get("risk_score")
    record.risk_breakdown = json.dumps(score.get("risk_breakdown", {}))

    # Voice behavioral sub-scores
    record.voice_behavioral_risk_score = voice_behavioral_sub.get("voice_behavioral_risk_score")
    record.coercion_text_score = voice_behavioral_sub.get("coercion_text_score")
    record.response_latency_score = voice_behavioral_sub.get("response_latency_score")
    record.pause_silence_score = voice_behavioral_sub.get("pause_silence_score")
    record.speech_rate_score = voice_behavioral_sub.get("speech_rate_score")
    record.pitch_variability_score = voice_behavioral_sub.get("pitch_variability_score")
    record.energy_prosody_score = voice_behavioral_sub.get("energy_prosody_score")
    record.short_answer_score = voice_behavioral_sub.get("short_answer_score")

    # Liveness
    record.liveness_spoof_risk_score = liveness_sub.get("liveness_spoof_risk_score")
    record.blink_liveness_score = liveness_sub.get("blink_liveness_score")
    record.voice_spoof_score = liveness_sub.get("voice_spoof_score")
    record.face_video_integrity_score = liveness_sub.get("face_video_integrity_score")
    record.challenge_response_score = liveness_sub.get("challenge_response_score")

    # Audio quality sub-scores
    record.audio_quality_risk_score = audio_quality_sub.get("audio_quality_risk_score")
    record.low_volume_score = audio_quality_sub.get("low_volume_score")
    record.noise_score = audio_quality_sub.get("noise_score")
    record.clipping_distortion_score = audio_quality_sub.get("clipping_distortion_score")
    record.missing_audio_score = audio_quality_sub.get("missing_audio_score")

    # Multimodal
    record.multimodal_consistency_risk_score = multimodal_sub.get("multimodal_consistency_risk_score")
    record.audio_video_sync_score = multimodal_sub.get("audio_video_sync_score")
    record.speaker_face_consistency_score = multimodal_sub.get("speaker_face_consistency_score")
    record.response_timing_consistency_score = multimodal_sub.get("response_timing_consistency_score")
    record.environment_consistency_score = multimodal_sub.get("environment_consistency_score")

    # Final status
    record.status = score.get("status", "Pendiente")
    record.decision_reason = score.get("decision_reason", "")
    record.updated_at = now

    # ── Emotional-Contextual v2 fields ─────────────────────────────────────
    record.score_model_version = score.get("score_model_version", "emotional_contextual_v2")
    record.recommended_action = score.get("recommended_action", "")
    record.risk_explanation = score.get("decision_reason", "")

    # Emotional AI
    emotion_ai = score.get("emotion_ai_result", {})
    record.emotional_ai_risk = emotion_ai.get("emotional_ai_risk")
    record.emotion_shift_score = emotion_ai.get("emotion_shift_score")
    record.multimodal_emotion_consistency = emotion_ai.get("multimodal_emotion_consistency")

    # Behavioral baseline
    behavioral = score.get("behavioral_result", {})
    record.behavioral_baseline_risk = behavioral.get("behavioral_baseline_risk")
    record.baseline_metrics = json.dumps(audio.get("baseline_metrics", {}))
    record.question_metrics = json.dumps(behavioral.get("question_metrics", {}))

    # Fraud triangle
    record.fraud_triangle_risk = fraud.get("fraud_triangle_risk")
    record.fraud_triangle_scores = json.dumps({
        "pressure_score": fraud.get("pressure_score"),
        "opportunity_score": fraud.get("opportunity_score"),
        "rationalization_score": fraud.get("rationalization_score"),
        "matched_signals": fraud.get("matched_signals", {}),
        "explanation": fraud.get("explanation", ""),
    })
    record.pressure_score = fraud.get("pressure_score")
    record.opportunity_score = fraud.get("opportunity_score")
    record.rationalization_score = fraud.get("rationalization_score")

    # Narrative coherence
    record.narrative_coherence_risk = narrative.get("narrative_coherence_risk")
    record.narrative_consistency_score = narrative.get("narrative_consistency_score")
    record.contradiction_score = narrative.get("contradiction_score")
    record.evasion_score = narrative.get("evasion_score")
    record.incompleteness_score = narrative.get("incompleteness_score")

    # Identity/liveness v2
    record.identity_liveness_risk = score.get("identity_liveness_risk")

    # Quality v2
    record.quality_risk = score.get("quality_risk")

    db.commit()
