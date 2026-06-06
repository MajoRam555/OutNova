import asyncio
import base64
import io
import logging

logger = logging.getLogger(__name__)


async def synthesize(text: str, voice: str) -> bytes | None:
    """
    Synthesize text to MP3 bytes using edge-tts (Microsoft Neural TTS, free).
    Returns None if synthesis fails — caller should fall back to browser TTS.
    """
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        data = buf.getvalue()
        return data if data else None
    except ImportError:
        logger.warning("[TTS] edge-tts no instalado. Ejecuta: pip install edge-tts")
        return None
    except Exception as e:
        logger.warning(f"[TTS] Síntesis fallida: {e}")
        return None


async def synthesize_b64(text: str, voice: str, timeout: float = 8.0) -> str | None:
    """
    Returns base64-encoded MP3 string for embedding in WebSocket messages.
    Returns None on failure or timeout — frontend falls back to Web Speech API.
    """
    try:
        data = await asyncio.wait_for(synthesize(text, voice), timeout=timeout)
        if not data:
            return None
        return base64.b64encode(data).decode()
    except asyncio.TimeoutError:
        logger.warning(f"[TTS] Timeout ({timeout}s) al sintetizar.")
        return None
    except Exception as e:
        logger.warning(f"[TTS] synthesize_b64 error: {e}")
        return None
