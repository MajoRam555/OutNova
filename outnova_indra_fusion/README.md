# OutNova Indra Fusion

Sistema de verificación biométrica conversacional para KYC/AML.

**Python 3.11 · FastAPI · Streamlit · Whisper · OpenCV · AURA LLM**

---

## Arquitectura

```
Frontend HTML → WebSocket AURA → FastAPI Backend → Pipeline ML → SQLite → Dashboard
```

El sistema combina:
- **Majito**: Score biométrico oficial (45/25/15/10/5), pipeline audio/video, coerción, dashboard
- **JP**: AURA conversacional, WebSocket, LLM local (Qwen2.5-0.5B), transcripts

---

## Requisitos previos

- **Python 3.11** (obligatorio)
- **FFmpeg** en PATH del sistema ([descarga](https://ffmpeg.org/download.html))
- Windows 10/11 (scripts .bat) — Linux compatible con comandos equivalentes

---

## Instalación rápida

```bat
# 1. Dependencias base
setup.bat

# 2. PyTorch + Whisper (pipeline ML)
setup_ml.bat

# 3. DeepFace (opcional, puede tardar)
setup_optional_deepface.bat
```

---

## Uso

```bat
# Iniciar servidor (terminal 1)
start_server.bat
# Abrir: http://localhost:8000

# Iniciar dashboard analista (terminal 2)
start_dashboard.bat
# Abrir: http://localhost:8501
```

---

## Flujo del sistema

1. **Bienvenida** — El usuario ve el sistema y confirma que está listo
2. **Verificación de entorno** — Cámara, micrófono, luminosidad y ruido validados
3. **Sesión conversacional** — AURA guía la conversación por WebSocket mientras se graba video+audio
4. **Upload** — El video WebM se sube al backend (POST /upload)
5. **Pipeline biométrico** — Ejecutado en background (ThreadPoolExecutor)
6. **Dashboard** — El analista revisa score, transcripción, liveness, coerción, AURA

---

## Score oficial

| Componente | Peso (sin DeepFace) | Peso (con DeepFace) |
|---|---|---|
| Voice Behavioral Risk | 47% | 45% |
| Liveness & Spoof Risk | 26% | 25% |
| Audio Quality Risk | 16% | 15% |
| Multimodal Consistency | 11% | 10% |
| DeepFace Emotion Risk | — | 5% |

**Decisión automática:**
- Score < 30 + sin deepfake → Aprobado
- Score 30–84 → Pendiente
- Score ≥ 85 → Rechazado
- Quality confidence < 70% → Recaptura
- Coerción detectada → Rechazado

---

## Estructura del proyecto

```
outnova_indra_fusion/
├── backend/
│   ├── main.py              # FastAPI + WebSocket + endpoints
│   ├── database.py          # SQLite con SQLAlchemy
│   ├── config.py            # Configuración centralizada
│   ├── aura_agent.py        # AURA LLM local + fallback por reglas
│   ├── pipeline.py          # Orquestador del pipeline biométrico
│   ├── audio_pipeline.py    # Pipeline completo de audio
│   ├── video_analysis.py    # Análisis de video + liveness
│   ├── speech_to_text.py    # Whisper
│   ├── audio_quality.py     # Calidad de audio
│   ├── voice_emotion.py     # Wav2Vec2 + librosa fallback
│   ├── voice_antispoof.py   # Heurística anti-spoof
│   ├── speaker_verification.py  # Stub (futuro)
│   ├── coercion_detection.py    # Detección server-side
│   ├── scoring.py           # Score oficial 45/25/15/10/5
│   ├── dashboard.py         # Streamlit para analistas
│   ├── start_server.py      # Entrypoint uvicorn
│   ├── static/outnova.html  # Frontend completo
│   ├── requirements.txt
│   ├── requirements-ml.txt
│   ├── requirements-optional.txt
│   ├── videos_recibidos/    # Videos subidos
│   └── uploads/audio/       # Audio extraído
├── setup.bat
├── setup_ml.bat
├── setup_optional_deepface.bat
├── start_server.bat
├── start_dashboard.bat
└── README.md
```

---

## Variables de entorno

Crear `backend/.env` o exportar antes de iniciar:

```env
INDRA_WORKERS=2
WHISPER_MODEL=base
USE_DEEPFACE=0
USE_WAV2VEC2=1
AURA_LOAD_ON_START=0
AURA_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
MAX_UPLOAD_SIZE_MB=200
LOG_LEVEL=INFO
```

---

## API Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Sirve el frontend HTML |
| WS | `/ws/conversacion?session_id=X` | WebSocket AURA |
| GET | `/health` | Estado del backend |
| GET | `/aura/status` | Estado del LLM AURA |
| POST | `/aura/load` | Cargar LLM en background |
| POST | `/upload` | Subir video de sesión |
| GET | `/sessions` | Listar sesiones (con filtros) |
| GET | `/sessions/{id}` | Detalle de sesión |
| PATCH | `/sessions/{id}/status` | Cambiar estado manualmente |

---

## Troubleshooting

**FFmpeg no encontrado:**
- Descarga desde https://ffmpeg.org/download.html
- Agrega `C:\ffmpeg\bin` a PATH del sistema y reinicia terminal

**Error al instalar torch:**
- Usa `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**DeepFace conflictos:**
- Instala en entorno separado si hay conflictos con TensorFlow
- El sistema funciona sin DeepFace (pesos redistribuidos automáticamente)

**AURA no responde:**
- Verifica `/aura/status` — si `llm_ready=false`, usa fallback por reglas
- Cargar LLM toma varios minutos la primera vez (descarga del modelo)
- El sistema funciona inmediatamente con fallback por reglas

**WebSocket no conecta:**
- Verifica que el backend esté corriendo en puerto 8000
- Abre http://localhost:8000/health para verificar estado

---

## Seguridad (pendiente para producción)

- [ ] Autenticación en API y dashboard
- [ ] Restringir CORS (quitar `allow_origins=["*"]`)
- [ ] HTTPS / TLS en producción
- [ ] Rate limiting en endpoints
- [ ] No exponer videos públicamente

---

## Pendiente futuro

- Speaker verification (resemblyzer / SpeechBrain)
- Faster-Whisper para mayor velocidad
- MediaPipe para liveness más robusto
- Challenge-response para score de liveness
- Notificaciones cuando pipeline termina

---

*OutNova Security v2.4.1 · Sistema biométrico conversacional KYC/AML*
