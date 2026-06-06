import logging
import subprocess
from pathlib import Path

from config import WHISPER_MODEL
from device_manager import get_device

logger = logging.getLogger(__name__)

_whisper_model = None
_whisper_device: str = "cpu"


def init_whisper_model():
    global _whisper_model, _whisper_device
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        import torch
        device = get_device()
        logger.info(f"Cargando modelo Whisper '{WHISPER_MODEL}' en {device}...")
        try:
            _whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
            _whisper_device = device
        except RuntimeError as oom:
            if "out of memory" in str(oom).lower():
                logger.warning("[WHISPER] CUDA OOM — fallback a CPU.")
                torch.cuda.empty_cache()
                _whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
                _whisper_device = "cpu"
            else:
                raise
        logger.info(f"Whisper listo en {_whisper_device}.")
        return _whisper_model
    except Exception as e:
        logger.error(f"No se pudo cargar Whisper: {e}")
        return None


def convert_audio_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio file to 16kHz mono WAV using ffmpeg. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-f", "wav",
                output_path,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg convert error: {result.stderr.decode(errors='replace')}")
            return False
        return True
    except Exception as e:
        logger.error(f"convert_audio_to_wav error: {e}")
        return False


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
        result = model.transcribe(audio_path, language=language, fp16=(_whisper_device == "cuda"))
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
