import logging
import numpy as np

logger = logging.getLogger(__name__)


def analyze_voice_emotion(audio_path: str) -> dict:
    """
    Tries Wav2Vec2 emotion classifier first, falls back to librosa heuristic.
    """
    try:
        return _wav2vec2_emotion(audio_path)
    except Exception as e:
        logger.warning(f"Wav2Vec2 emotion fallback a librosa: {e}")
        return _librosa_emotion_fallback(audio_path)


def _wav2vec2_emotion(audio_path: str) -> dict:
    from transformers import pipeline
    import torch

    classifier = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=-1,
    )
    results = classifier(audio_path)
    scores = {r["label"]: round(r["score"], 4) for r in results}
    dominant = max(scores, key=scores.get) if scores else "unknown"

    return {
        "dominant_emotion": dominant,
        "emotion_scores": scores,
        "confidence": scores.get(dominant, 0.0),
        "primary_engine": "wav2vec2",
        "error": None,
    }


def _librosa_emotion_fallback(audio_path: str) -> dict:
    try:
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(audio) < 1600:
            return _empty_emotion("Audio muy corto para análisis emocional.")

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        energy = float(np.mean(librosa.feature.rms(y=audio)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))

        emotion = "neutral"
        if energy > 0.08 and zcr > 0.12:
            emotion = "stressed"
        elif energy < 0.02:
            emotion = "low_energy"
        elif zcr > 0.18:
            emotion = "nervous"

        return {
            "dominant_emotion": emotion,
            "emotion_scores": {emotion: 0.6, "neutral": 0.4},
            "confidence": 0.6,
            "primary_engine": "librosa_heuristic",
            "error": None,
        }
    except Exception as e:
        return _empty_emotion(str(e))


def _empty_emotion(error: str) -> dict:
    return {
        "dominant_emotion": "unknown",
        "emotion_scores": {},
        "confidence": 0.0,
        "primary_engine": "none",
        "error": error,
    }


def extract_acoustic_features(audio_path: str, transcription_words: int = 0,
                               duration_sec: float = 0.0) -> dict:
    """
    Extracts pitch, energy, speech rate and timing features from WAV.
    """
    defaults = {
        "pitch_mean_hz": 0.0,
        "pitch_std_hz": 0.0,
        "pitch_cv": 0.0,
        "energy_mean": 0.0,
        "energy_std": 0.0,
        "energy_cv": 0.0,
        "speech_rate_wpm": 0.0,
        "first_speech_sec": 0.0,
        "error": None,
    }

    try:
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(audio) < 3200:
            defaults["error"] = "Audio muy corto"
            return defaults

        # Pitch
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=2048
        )
        voiced_f0 = f0[voiced_flag] if f0 is not None and voiced_flag is not None else np.array([])
        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0
        pitch_cv = pitch_std / (pitch_mean + 1e-10) if pitch_mean > 0 else 0.0

        # Energy
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
        energy_cv = energy_std / (energy_mean + 1e-10)

        # Speech rate
        speech_rate_wpm = 0.0
        if duration_sec > 0 and transcription_words > 0:
            speech_rate_wpm = round(transcription_words / (duration_sec / 60.0), 1)

        # First speech latency
        silence_thresh = np.mean(rms) * 0.3
        first_speech_sec = 0.0
        hop_dur = 512 / sr
        for i, r in enumerate(rms):
            if r > silence_thresh:
                first_speech_sec = i * hop_dur
                break

        return {
            "pitch_mean_hz": round(pitch_mean, 2),
            "pitch_std_hz": round(pitch_std, 2),
            "pitch_cv": round(pitch_cv, 4),
            "energy_mean": round(energy_mean, 6),
            "energy_std": round(energy_std, 6),
            "energy_cv": round(energy_cv, 4),
            "speech_rate_wpm": speech_rate_wpm,
            "first_speech_sec": round(first_speech_sec, 2),
            "error": None,
        }

    except Exception as e:
        logger.warning(f"extract_acoustic_features error: {e}")
        defaults["error"] = str(e)
        return defaults
