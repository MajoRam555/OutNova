import logging
from pathlib import Path

from config import WHISPER_MODEL

logger = logging.getLogger(__name__)

_whisper_model = None


def init_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        logger.info(f"Cargando modelo Whisper '{WHISPER_MODEL}'...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        logger.info("Whisper listo.")
        return _whisper_model
    except Exception as e:
        logger.error(f"No se pudo cargar Whisper: {e}")
        return None


def transcribe_audio_file(audio_path: str, language: str = "es") -> dict:
    if not Path(audio_path).exists():
        return {
            "transcription": "",
            "transcription_words": 0,
            "detected_language": None,
            "error": f"Archivo no encontrado: {audio_path}",
        }

    model = init_whisper_model()
    if model is None:
        return {
            "transcription": "",
            "transcription_words": 0,
            "detected_language": None,
            "error": "Modelo Whisper no disponible.",
        }

    try:
        result = model.transcribe(audio_path, language=language, fp16=False)
        text = result.get("text", "").strip()
        detected_lang = result.get("language", language)
        words = len(text.split()) if text else 0

        logger.info(f"Whisper transcribió {words} palabras en idioma '{detected_lang}'.")
        return {
            "transcription": text,
            "transcription_words": words,
            "detected_language": detected_lang,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error en transcripción Whisper: {e}")
        return {
            "transcription": "",
            "transcription_words": 0,
            "detected_language": None,
            "error": str(e),
        }
