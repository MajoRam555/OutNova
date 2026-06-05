"""
OutNova Indra Fusion — Backend principal FastAPI
"""
import json
import logging
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (Depends, FastAPI, File, Form, HTTPException,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from config import (AUDIO_DIR, INDRA_WORKERS, MAX_SESSION_SECONDS,
                    MAX_UPLOAD_SIZE_MB, PTT_DIR, STATIC_DIR, VIDEO_DIR, AURA_LOAD_ON_START)
from database import (AnalysisSession, SessionLocal, create_session_record,
                       get_db, get_session_by_client_id, init_db,
                       update_session_status)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Thread Pool Executors ──────────────────────────────────────────────────
_ml_executor: ThreadPoolExecutor = None
_aura_executor: ThreadPoolExecutor = None
_llm_executor: ThreadPoolExecutor = None


def _check_ffmpeg() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ml_executor, _aura_executor, _llm_executor

    logger.info("=== OutNova Indra Fusion — Iniciando backend ===")

    # Create folders
    for d in [VIDEO_DIR, AUDIO_DIR, PTT_DIR, STATIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Carpetas de trabajo verificadas.")

    # Init DB
    init_db()

    # Executors
    _ml_executor = ThreadPoolExecutor(max_workers=INDRA_WORKERS, thread_name_prefix="ml")
    _aura_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura")
    _llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
    logger.info(f"ThreadPoolExecutors listos (ml_workers={INDRA_WORKERS}).")

    # FFmpeg check
    if _check_ffmpeg():
        logger.info("FFmpeg disponible.")
    else:
        logger.warning("FFmpeg NO encontrado en PATH. El pipeline de audio/video puede fallar.")

    # Optional: load AURA LLM on start
    if AURA_LOAD_ON_START:
        from aura_agent import get_engine
        logger.info("[AURA] Cargando LLM en background...")
        _llm_executor.submit(get_engine().load_llm_only)

    logger.info("Backend listo. Accede en http://localhost:8000/")
    yield

    # Shutdown
    logger.info("Cerrando executors...")
    _ml_executor.shutdown(wait=False)
    _aura_executor.shutdown(wait=False)
    _llm_executor.shutdown(wait=False)


app = FastAPI(
    title="OutNova Indra Fusion",
    version="2.4.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static files ──────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    index = STATIC_DIR / "outnova.html"
    if not index.exists():
        return JSONResponse({"error": "Frontend no encontrado. Verifica static/outnova.html"}, status_code=404)
    return FileResponse(str(index))


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    from aura_agent import get_engine
    engine = get_engine()
    status = engine.status()

    deepface_available = None
    try:
        import deepface
        deepface_available = True
    except ImportError:
        deepface_available = False

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "database_ready": True,
        "video_dir_ready": VIDEO_DIR.exists(),
        "ffmpeg_available": _check_ffmpeg(),
        "aura_ready": status["llm_ready"] or status["used_fallback"],
        "llm_ready": status["llm_ready"],
        "llm_model": status["loaded_model"],
        "deepface_available": deepface_available,
    }


# ── AURA Status ────────────────────────────────────────────────────────────
@app.get("/aura/status")
async def aura_status():
    from aura_agent import get_engine
    return get_engine().status()


# ── Load AURA LLM on-demand ────────────────────────────────────────────────
@app.post("/aura/load")
async def aura_load():
    from aura_agent import get_engine
    engine = get_engine()
    if engine.llm_ready:
        return {"message": "LLM ya está cargado.", "model": engine.loaded_model}
    _llm_executor.submit(engine.load_llm_only)
    return {"message": "Carga de LLM iniciada en background."}


# ── PTT: transcribe short audio clip ──────────────────────────────────────
@app.post("/aura/transcribe-turn")
async def transcribe_ptt_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
):
    import asyncio
    from speech_to_text import convert_audio_to_wav, transcribe_audio_file

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio vacío.")

    # Detect extension from filename or default to webm
    ext = Path(audio.filename or "ptt.webm").suffix.lower() or ".webm"
    raw_path = PTT_DIR / f"{session_id}_{uuid.uuid4().hex}{ext}"
    wav_path = raw_path.with_suffix(".wav")

    try:
        with open(raw_path, "wb") as f:
            f.write(content)

        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(
            _ml_executor,
            lambda: convert_audio_to_wav(str(raw_path), str(wav_path)),
        )
        if not ok:
            raise HTTPException(status_code=422, detail="No se pudo convertir el audio PTT.")

        result = await loop.run_in_executor(
            _ml_executor,
            lambda: transcribe_audio_file(str(wav_path), language="es"),
        )
    finally:
        for p in [raw_path, wav_path]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    if result.get("error") and not result.get("transcription"):
        raise HTTPException(status_code=500, detail=f"Transcripción fallida: {result['error']}")

    text = result.get("transcription", "").strip()
    return {
        "transcription": text,
        "words": result.get("transcription_words", 0),
        "language": result.get("detected_language"),
        "session_id": session_id,
    }


# ── WebSocket Conversación AURA ────────────────────────────────────────────
@app.websocket("/ws/conversacion")
async def ws_conversacion(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[WS] Conexión aceptada: session_id={session_id}")

    from aura_agent import get_engine, get_or_create_session
    import asyncio

    engine = get_engine()
    session = get_or_create_session(session_id)
    loop = asyncio.get_event_loop()

    async def send_json(data: dict):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.debug(f"[WS] send error: {e}")

    # Send session info
    st = engine.status()
    await send_json({
        "type": "session",
        "session_id": session_id,
        "llm_ready": st["llm_ready"],
        "tts_ready": st["tts_ready"],
        "used_fallback": st["used_fallback"] or not st["llm_ready"],
    })

    # AURA greeting
    greeting = engine.fallback_reply(session, 0)
    if engine.llm_ready:
        try:
            greeting = await loop.run_in_executor(
                _llm_executor,
                lambda: engine.generate_reply(session, "")
            )
        except Exception as e:
            logger.warning(f"[WS] Greeting LLM error: {e}")
            greeting = engine.fallback_reply(session, 0)

    session.add_message("aura", greeting)
    session.turn_count += 1
    await send_json({"type": "aura_turn", "text": greeting, "turn": session.turn_count})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_json({"type": "error", "message": "JSON inválido."})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await send_json({"type": "pong"})
                continue

            if msg_type == "conversation_end":
                await send_json({"type": "status", "message": "Conversación finalizada. Procesando upload..."})
                break

            if msg_type == "user_text":
                user_text = msg.get("text", "").strip()
                if not user_text:
                    continue

                source = msg.get("source", "typed")
                session.add_message("user", user_text, source=source)
                await send_json({"type": "status", "message": "AURA está procesando..."})

                # Generate reply in LLM executor
                try:
                    reply = await loop.run_in_executor(
                        _llm_executor,
                        lambda ut=user_text: engine.generate_reply(session, ut)
                    )
                except Exception as e:
                    logger.warning(f"[WS] Reply generation error: {e}")
                    reply = engine.fallback_reply(session, session.turn_count, user_text)

                session.add_message("aura", reply)
                session.turn_count += 1
                await send_json({"type": "aura_turn", "text": reply, "turn": session.turn_count})

    except WebSocketDisconnect:
        logger.info(f"[WS] Desconectado: {session_id}")
    except Exception as e:
        logger.error(f"[WS] Error inesperado: {e}")
        try:
            await send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── Upload ─────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    session_id: str = Form(...),
    timestamp: Optional[str] = Form(None),
    accessibility_mode: Optional[bool] = Form(False),
    db: Session = Depends(get_db),
):
    # Validate size
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content = await video.read()
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Archivo demasiado grande (max {MAX_UPLOAD_SIZE_MB}MB).")

    # Validate extension
    ext = Path(video.filename or "video.webm").suffix.lower()
    if ext not in {".webm", ".mp4", ".mov", ".avi"}:
        ext = ".webm"

    # Save file
    filename = f"{session_id}{ext}"
    video_path = VIDEO_DIR / filename
    with open(video_path, "wb") as f:
        f.write(content)
    logger.info(f"[UPLOAD] Video guardado: {video_path} ({len(content)/1024:.1f} KB)")

    # Get AURA transcript
    from aura_agent import get_session, pop_session
    aura_session = get_session(session_id)
    aura_transcript = None
    aura_turns = 0
    aura_model_status = "fallback"
    aura_used_fallback = True

    if aura_session:
        aura_transcript = aura_session.get_transcript_json()
        aura_turns = aura_session.turn_count
        aura_used_fallback = aura_session.used_fallback
        aura_model_status = "llm" if not aura_used_fallback else "fallback"

    # Create or update DB record
    existing = get_session_by_client_id(db, session_id)
    if existing:
        existing.video_path = str(video_path)
        existing.aura_transcript = aura_transcript
        existing.aura_turn_count = aura_turns
        existing.aura_model_status = aura_model_status
        existing.aura_used_fallback = aura_used_fallback
        existing.accessibility_mode = bool(accessibility_mode)
        existing.status = "Pendiente"
        existing.updated_at = datetime.utcnow().isoformat()
        db.commit()
        db.refresh(existing)
        record = existing
    else:
        record = create_session_record(db, session_id, str(video_path))
        record.aura_transcript = aura_transcript
        record.aura_turn_count = aura_turns
        record.aura_model_status = aura_model_status
        record.aura_used_fallback = aura_used_fallback
        record.accessibility_mode = bool(accessibility_mode)
        db.commit()
        db.refresh(record)

    record_id = record.id

    # Launch pipeline in background
    _ml_executor.submit(
        _run_pipeline_safe,
        record_id,
        str(video_path),
        session_id,
    )

    logger.info(f"[UPLOAD] Pipeline lanzado en background — DB id={record_id}")

    return JSONResponse(
        status_code=202,
        content={
            "id": str(record_id),
            "status": "processing",
            "message": "El video está siendo analizado.",
            "session_id": session_id,
        },
    )


def _run_pipeline_safe(session_id: int, video_path: str, client_session_id: str):
    try:
        from pipeline import run_analysis
        run_analysis(session_id, video_path, client_session_id)
    except Exception as e:
        logger.error(f"[PIPELINE] Error en _run_pipeline_safe: {e}", exc_info=True)


# ── Sessions list ──────────────────────────────────────────────────────────
@app.get("/sessions")
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(AnalysisSession)
    if status:
        query = query.filter(AnalysisSession.status == status)
    total = query.count()
    records = query.order_by(AnalysisSession.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "sessions": [r.to_dict() for r in records],
    }


# ── Session detail ─────────────────────────────────────────────────────────
@app.get("/sessions/{session_id}")
async def get_session_detail(session_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    return record.to_dict()


# ── Update session status ──────────────────────────────────────────────────
ALLOWED_STATUSES = {"Pendiente", "Procesando", "Aprobado", "Rechazado", "Recaptura"}


@app.patch("/sessions/{session_id}/status")
async def update_status(
    session_id: int,
    body: dict,
    db: Session = Depends(get_db),
):
    new_status = body.get("status")
    reason = body.get("reason", "")

    if new_status not in ALLOWED_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Estado inválido. Opciones: {sorted(ALLOWED_STATUSES)}"
        )

    record = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")

    record.status = new_status
    if reason:
        record.status_reason = reason
    record.updated_at = datetime.utcnow().isoformat()
    db.commit()
    db.refresh(record)

    return {"id": session_id, "status": new_status, "updated_at": record.updated_at}
