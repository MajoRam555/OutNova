import os
from pathlib import Path

APP_NAME = "OutNova Indra Fusion"
APP_VERSION = "2.4.1"

BASE_DIR = Path(__file__).parent.resolve()
DATABASE_URL = f"sqlite:///{BASE_DIR}/indra_sessions.db"

VIDEO_DIR = BASE_DIR / "videos_recibidos"
AUDIO_DIR = BASE_DIR / "uploads" / "audio"
PTT_DIR = BASE_DIR / "uploads" / "ptt"
STATIC_DIR = BASE_DIR / "static"

# Workers
INDRA_WORKERS = int(os.getenv("INDRA_WORKERS", "2"))

# Session
MAX_SESSION_SECONDS = int(os.getenv("MAX_SESSION_SECONDS", "300"))

# AURA / LLM
AURA_LOAD_ON_START = bool(int(os.getenv("AURA_LOAD_ON_START", "0")))
AURA_LLM_MODEL = os.getenv("AURA_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
AURA_LLM_FALLBACK = os.getenv("AURA_LLM_FALLBACK", "microsoft/Phi-3-mini-4k-instruct")

# ML Models
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
USE_DEEPFACE = bool(int(os.getenv("USE_DEEPFACE", "0")))
USE_WAV2VEC2 = bool(int(os.getenv("USE_WAV2VEC2", "1")))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Audio
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

# Thresholds
MIN_LIGHT_THRESHOLD = float(os.getenv("MIN_LIGHT_THRESHOLD", "40"))
MAX_NOISE_THRESHOLD = float(os.getenv("MAX_NOISE_THRESHOLD", "40"))

# Upload
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

# FFmpeg
FFMPEG_REQUIRED = bool(int(os.getenv("FFMPEG_REQUIRED", "1")))
