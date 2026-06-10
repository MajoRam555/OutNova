import logging
import numpy as np

from config import QUALITY_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


def analyze_audio_quality(audio_path: str) -> dict:
    """
    Analyzes WAV file for quality metrics.
    Returns risk scores and confidence.
    """
    base_result = {
        "duration_sec": 0.0,
        "silence_ratio": 1.0,
        "rms_mean": 0.0,
        "rms_max": 0.0,
        "clipping_ratio": 0.0,
        "snr_db": 0.0,
        "sample_rate": 16000,
        "quality_ok": False,
        "quality_confidence": 0.1,
        "low_volume_score": 80.0,
        "noise_score": 50.0,
        "clipping_distortion_score": 0.0,
        "missing_audio_score": 90.0,
        "audio_quality_risk_score": 80.0,
        "error": None,
    }

    try:
        import soundfile as sf
        import librosa

        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration_sec = len(audio) / sr

        if duration_sec < 1.0:
            base_result["duration_sec"] = duration_sec
            base_result["error"] = "Audio demasiado corto"
            return base_result

        rms_frame = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms_mean = float(np.mean(rms_frame))
        rms_max = float(np.max(rms_frame))

        silence_threshold = 0.01
        silent_frames = np.sum(rms_frame < silence_threshold)
        silence_ratio = float(silent_frames / len(rms_frame))

        clipping_threshold = 0.98
        clipping_ratio = float(np.mean(np.abs(audio) > clipping_threshold))

        noise_floor = np.percentile(np.abs(audio), 10)
        signal_peak = np.percentile(np.abs(audio), 90)
        snr_db = float(20 * np.log10((signal_peak + 1e-10) / (noise_floor + 1e-10)))

        # Sub-scores (0=best, 100=worst)
        low_volume_score = max(0.0, min(100.0, (0.05 - rms_mean) / 0.05 * 100)) if rms_mean < 0.05 else 0.0
        noise_score = max(0.0, min(100.0, max(0.0, (20.0 - snr_db) / 20.0 * 80.0)))
        clipping_distortion_score = min(100.0, clipping_ratio * 1000.0)
        missing_audio_score = min(100.0, silence_ratio * 100.0)

        audio_quality_risk_score = (
            low_volume_score * 0.30
            + noise_score * 0.25
            + clipping_distortion_score * 0.20
            + missing_audio_score * 0.25
        )

        quality_confidence = max(0.0, min(1.0,
            1.0
            - (silence_ratio * 0.5)
            - (clipping_ratio * 3.0)
            - max(0.0, (0.02 - rms_mean) / 0.02 * 0.3)
        ))

        return {
            "duration_sec": round(duration_sec, 2),
            "silence_ratio": round(silence_ratio, 4),
            "rms_mean": round(rms_mean, 6),
            "rms_max": round(rms_max, 6),
            "clipping_ratio": round(clipping_ratio, 6),
            "snr_db": round(snr_db, 2),
            "sample_rate": sr,
            "quality_ok": quality_confidence >= QUALITY_CONFIDENCE_THRESHOLD,
            "quality_confidence": round(quality_confidence, 4),
            "low_volume_score": round(low_volume_score, 2),
            "noise_score": round(noise_score, 2),
            "clipping_distortion_score": round(clipping_distortion_score, 2),
            "missing_audio_score": round(missing_audio_score, 2),
            "audio_quality_risk_score": round(audio_quality_risk_score, 2),
            "error": None,
        }

    except Exception as e:
        logger.warning(f"audio_quality error: {e}")
        base_result["error"] = str(e)
        return base_result
