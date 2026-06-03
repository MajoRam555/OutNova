import logging

logger = logging.getLogger(__name__)


def verify_speaker(audio_path: str, reference_path: str = None) -> dict:
    """Stub for future speaker verification implementation."""
    logger.debug("speaker_verification: stub activo, retornando resultado neutro.")
    return {
        "available": False,
        "score": None,
        "note": "speaker verification stub — pendiente implementación con resemblyzer o SpeechBrain",
    }
