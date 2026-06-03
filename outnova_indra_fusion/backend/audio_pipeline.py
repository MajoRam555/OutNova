import json
import logging
import os
import subprocess
from pathlib import Path

from config import AUDIO_DIR, AUDIO_SAMPLE_RATE
from audio_quality import analyze_audio_quality
from speech_to_text import transcribe_audio_file
from voice_emotion import analyze_voice_emotion, extract_acoustic_features
from voice_antispoof import analyze_voice_antispoof
from speaker_verification import verify_speaker
from coercion_detection import detect_coercion

logger = logging.getLogger(__name__)


def extract_audio_ffmpeg(video_path: str, session_id: str) -> str | None:
    """Extracts mono 16kHz WAV from video using ffmpeg."""
    out_path = str(AUDIO_DIR / f"{session_id}.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1",
        out_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and Path(out_path).exists():
            logger.info(f"Audio extraído con ffmpeg: {out_path}")
            return out_path
        else:
            logger.warning(f"ffmpeg audio error: {result.stderr.decode()[:300]}")
            return None
    except Exception as e:
        logger.warning(f"extract_audio_ffmpeg excepción: {e}")
        return None


def extract_audio_moviepy(video_path: str, session_id: str) -> str | None:
    """Fallback: extract audio using moviepy."""
    out_path = str(AUDIO_DIR / f"{session_id}.wav")
    try:
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip

        clip = VideoFileClip(video_path)
        if clip.audio is None:
            logger.warning("moviepy: video sin pista de audio")
            return None
        clip.audio.write_audiofile(out_path, fps=AUDIO_SAMPLE_RATE, nbytes=2, ffmpeg_params=["-ac", "1"])
        clip.close()
        if Path(out_path).exists():
            logger.info(f"Audio extraído con moviepy: {out_path}")
            return out_path
        return None
    except Exception as e:
        logger.warning(f"extract_audio_moviepy error: {e}")
        return None


def extract_audio(video_path: str, session_id: str) -> str | None:
    """Try ffmpeg first, then moviepy."""
    audio_path = extract_audio_ffmpeg(video_path, session_id)
    if audio_path:
        return audio_path
    logger.info("Intentando extracción de audio con moviepy...")
    return extract_audio_moviepy(video_path, session_id)


def _compute_voice_behavioral_subscores(
    coercion_result: dict,
    acoustic: dict,
    quality: dict,
    transcription_words: int,
    duration_sec: float,
) -> dict:
    """
    Calculates 7 voice behavioral sub-scores (0-100, higher = more risk).
    """
    # coercion_text_score
    if coercion_result.get("coercion_detected"):
        coercion_text = min(100.0, 50.0 + coercion_result.get("coercion_score", 0.5) * 100.0)
    else:
        coercion_text = 0.0

    # response_latency_score
    latency = acoustic.get("first_speech_sec", 0.0)
    if latency > 8.0:
        latency_score = 80.0
    elif latency > 5.0:
        latency_score = 50.0
    elif latency > 3.0:
        latency_score = 25.0
    else:
        latency_score = 5.0

    # pause_silence_score
    silence_ratio = quality.get("silence_ratio", 0.0)
    if silence_ratio > 0.85:
        pause_score = 90.0
    elif silence_ratio > 0.60:
        pause_score = 55.0
    elif silence_ratio > 0.40:
        pause_score = 25.0
    else:
        pause_score = 5.0

    # speech_rate_score
    wpm = acoustic.get("speech_rate_wpm", 0.0)
    if wpm == 0:
        rate_score = 70.0
    elif wpm < 50:
        rate_score = 65.0
    elif wpm > 300:
        rate_score = 60.0
    elif 80 <= wpm <= 220:
        rate_score = 5.0
    else:
        rate_score = 25.0

    # pitch_variability_score
    pitch_cv = acoustic.get("pitch_cv", 0.0)
    if pitch_cv < 0.05:
        pitch_var_score = 70.0
    elif pitch_cv > 0.55:
        pitch_var_score = 65.0
    elif 0.10 <= pitch_cv <= 0.40:
        pitch_var_score = 5.0
    else:
        pitch_var_score = 25.0

    # energy_prosody_score
    energy_mean = acoustic.get("energy_mean", 0.0)
    energy_cv = acoustic.get("energy_cv", 0.0)
    if energy_mean < 0.01:
        energy_score = 75.0
    elif energy_cv < 0.05:
        energy_score = 55.0
    elif 0.10 <= energy_cv <= 0.50:
        energy_score = 5.0
    else:
        energy_score = 20.0

    # short_answer_score
    if duration_sec > 0 and transcription_words > 0:
        words_per_sec = transcription_words / duration_sec
        if words_per_sec < 0.5:
            short_score = 60.0
        elif words_per_sec > 3.5:
            short_score = 30.0
        else:
            short_score = 5.0
    elif transcription_words == 0:
        short_score = 75.0
    else:
        short_score = 30.0

    weights = [0.25, 0.10, 0.15, 0.15, 0.15, 0.10, 0.10]
    scores_list = [
        coercion_text, latency_score, pause_score, rate_score,
        pitch_var_score, energy_score, short_score
    ]
    voice_behavioral = sum(w * s for w, s in zip(weights, scores_list))

    return {
        "coercion_text_score": round(coercion_text, 2),
        "response_latency_score": round(latency_score, 2),
        "pause_silence_score": round(pause_score, 2),
        "speech_rate_score": round(rate_score, 2),
        "pitch_variability_score": round(pitch_var_score, 2),
        "energy_prosody_score": round(energy_score, 2),
        "short_answer_score": round(short_score, 2),
        "voice_behavioral_risk_score": round(voice_behavioral, 2),
    }


def run_audio_pipeline(video_path: str, session_id: str, duration_sec: float = 0.0) -> dict:
    """
    Full audio pipeline. Returns comprehensive audio analysis dict.
    Tolerant of individual module failures.
    """
    result = {
        "audio_extracted": False,
        "audio_path": None,
        "transcription": "",
        "transcription_words": 0,
        "detected_language": None,
        "audio_quality": {},
        "quality_confidence": 0.1,
        "silence_ratio": 1.0,
        "clipping_ratio": 0.0,
        "noise_level": 0.0,
        "avg_volume": 0.0,
        "first_voice_latency": 0.0,
        "speech_rate_wpm": 0.0,
        "pitch_cv": 0.0,
        "energy_cv": 0.0,
        "voice_emotion": {},
        "voice_antispoof": {},
        "speaker_verification": {},
        "coercion": {},
        "voice_behavioral_sub_scores": {},
        "voice_behavioral_risk_score": 50.0,
        "errors": [],
    }

    # 1. Extract audio
    logger.info(f"[AUDIO] Extrayendo audio de {video_path}")
    audio_path = extract_audio(video_path, session_id)
    if not audio_path:
        result["errors"].append("No se pudo extraer audio del video.")
        logger.error("[AUDIO] Extracción fallida — saltando pipeline de audio.")
        return result

    result["audio_extracted"] = True
    result["audio_path"] = audio_path

    # 2. Quality analysis
    logger.info("[AUDIO] Analizando calidad de audio...")
    try:
        quality = analyze_audio_quality(audio_path)
        result["audio_quality"] = quality
        result["quality_confidence"] = quality.get("quality_confidence", 0.1)
        result["silence_ratio"] = quality.get("silence_ratio", 1.0)
        result["clipping_ratio"] = quality.get("clipping_ratio", 0.0)
        result["noise_level"] = quality.get("snr_db", 0.0)
        result["avg_volume"] = quality.get("rms_mean", 0.0)
        if not duration_sec:
            duration_sec = quality.get("duration_sec", 0.0)
    except Exception as e:
        result["errors"].append(f"audio_quality: {e}")
        logger.warning(f"[AUDIO] quality error: {e}")

    # 3. Transcription
    logger.info("[AUDIO] Transcribiendo con Whisper...")
    try:
        stt = transcribe_audio_file(audio_path)
        result["transcription"] = stt.get("transcription", "")
        result["transcription_words"] = stt.get("transcription_words", 0)
        result["detected_language"] = stt.get("detected_language")
        if stt.get("error"):
            result["errors"].append(f"whisper: {stt['error']}")
    except Exception as e:
        result["errors"].append(f"whisper_exception: {e}")
        logger.warning(f"[AUDIO] Whisper error: {e}")

    # 4. Acoustic features
    logger.info("[AUDIO] Extrayendo features acústicas...")
    acoustic = {}
    try:
        acoustic = extract_acoustic_features(
            audio_path,
            transcription_words=result["transcription_words"],
            duration_sec=duration_sec,
        )
        result["first_voice_latency"] = acoustic.get("first_speech_sec", 0.0)
        result["speech_rate_wpm"] = acoustic.get("speech_rate_wpm", 0.0)
        result["pitch_cv"] = acoustic.get("pitch_cv", 0.0)
        result["energy_cv"] = acoustic.get("energy_cv", 0.0)
        if acoustic.get("error"):
            result["errors"].append(f"acoustic_features: {acoustic['error']}")
    except Exception as e:
        result["errors"].append(f"acoustic_features_exception: {e}")
        logger.warning(f"[AUDIO] Acoustic features error: {e}")

    # 5. Voice emotion
    logger.info("[AUDIO] Analizando emoción de voz...")
    try:
        emotion = analyze_voice_emotion(audio_path)
        result["voice_emotion"] = emotion
    except Exception as e:
        result["voice_emotion"] = {"error": str(e)}
        result["errors"].append(f"voice_emotion: {e}")
        logger.warning(f"[AUDIO] Voice emotion error: {e}")

    # 6. Anti-spoof
    try:
        antispoof = analyze_voice_antispoof(
            audio_path,
            pitch_cv=result.get("pitch_cv"),
            energy_cv=result.get("energy_cv"),
            duration_sec=duration_sec,
        )
        result["voice_antispoof"] = antispoof
    except Exception as e:
        result["voice_antispoof"] = {"error": str(e)}
        result["errors"].append(f"voice_antispoof: {e}")

    # 7. Speaker verification (stub)
    try:
        result["speaker_verification"] = verify_speaker(audio_path)
    except Exception as e:
        result["speaker_verification"] = {"error": str(e)}

    # 8. Coercion detection
    logger.info("[AUDIO] Detectando coerción...")
    try:
        coercion = detect_coercion(result["transcription"])
        result["coercion"] = coercion
    except Exception as e:
        result["coercion"] = {"coercion_detected": False, "coercion_score": 0.0, "error": str(e)}
        result["errors"].append(f"coercion: {e}")

    # 9. Voice behavioral sub-scores
    try:
        behavioral = _compute_voice_behavioral_subscores(
            coercion_result=result["coercion"],
            acoustic=acoustic,
            quality=result["audio_quality"],
            transcription_words=result["transcription_words"],
            duration_sec=duration_sec,
        )
        result["voice_behavioral_sub_scores"] = behavioral
        result["voice_behavioral_risk_score"] = behavioral.get("voice_behavioral_risk_score", 50.0)
    except Exception as e:
        result["errors"].append(f"voice_behavioral: {e}")
        logger.warning(f"[AUDIO] Voice behavioral error: {e}")

    logger.info(
        f"[AUDIO] Pipeline completo — words={result['transcription_words']}, "
        f"quality_conf={result['quality_confidence']:.2f}, "
        f"behavioral_risk={result['voice_behavioral_risk_score']:.1f}"
    )
    return result
