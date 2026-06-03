import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (Boolean, Column, Float, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    # General
    id = Column(Integer, primary_key=True, index=True)
    client_session_id = Column(String, unique=True, index=True, nullable=False)
    video_path = Column(String, nullable=True)
    video_duration_sec = Column(Float, nullable=True)
    status = Column(String, default="Pendiente")
    status_reason = Column(Text, nullable=True)
    decision_reason = Column(Text, nullable=True)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at = Column(String, default=lambda: datetime.utcnow().isoformat())

    # AURA
    aura_transcript = Column(Text, nullable=True)
    aura_turn_count = Column(Integer, default=0)
    aura_model_status = Column(String, nullable=True)
    aura_used_fallback = Column(Boolean, default=False)

    # Video/Liveness
    total_blinks = Column(Integer, nullable=True)
    blink_rate_per_min = Column(Float, nullable=True)
    blink_cv = Column(Float, nullable=True)
    face_detected_ratio = Column(Float, nullable=True)
    eye_detected_ratio = Column(Float, nullable=True)
    avg_ear = Column(Float, nullable=True)
    deepfake_flag = Column(Boolean, default=False)
    deepfake_reason = Column(Text, nullable=True)
    emotion_percentages = Column(Text, nullable=True)
    dominant_emotion = Column(String, nullable=True)
    stress_emotion_ratio = Column(Float, nullable=True)
    deepface_available = Column(Boolean, default=False)

    # Audio
    audio_path = Column(String, nullable=True)
    transcription = Column(Text, nullable=True)
    transcription_words = Column(Integer, default=0)
    detected_language = Column(String, nullable=True)
    audio_analysis_json = Column(Text, nullable=True)
    audio_quality = Column(Text, nullable=True)
    quality_confidence = Column(Float, nullable=True)
    silence_ratio = Column(Float, nullable=True)
    clipping_ratio = Column(Float, nullable=True)
    noise_level = Column(Float, nullable=True)
    avg_volume = Column(Float, nullable=True)
    first_voice_latency = Column(Float, nullable=True)
    speech_rate_wpm = Column(Float, nullable=True)
    pitch_cv = Column(Float, nullable=True)
    energy_cv = Column(Float, nullable=True)
    voice_emotion = Column(Text, nullable=True)
    voice_antispoof = Column(Text, nullable=True)
    speaker_verification = Column(Text, nullable=True)

    # Coercion
    coercion_detected = Column(Boolean, default=False)
    coercion_score = Column(Float, default=0.0)
    coercion_matches = Column(Text, nullable=True)

    # Score
    risk_score = Column(Float, nullable=True)
    risk_breakdown = Column(Text, nullable=True)

    # Voice Behavioral sub-scores
    voice_behavioral_risk_score = Column(Float, nullable=True)
    coercion_text_score = Column(Float, nullable=True)
    response_latency_score = Column(Float, nullable=True)
    pause_silence_score = Column(Float, nullable=True)
    speech_rate_score = Column(Float, nullable=True)
    pitch_variability_score = Column(Float, nullable=True)
    energy_prosody_score = Column(Float, nullable=True)
    short_answer_score = Column(Float, nullable=True)

    # Liveness/Spoof sub-scores
    liveness_spoof_risk_score = Column(Float, nullable=True)
    blink_liveness_score = Column(Float, nullable=True)
    voice_spoof_score = Column(Float, nullable=True)
    face_video_integrity_score = Column(Float, nullable=True)
    challenge_response_score = Column(Float, nullable=True)

    # Audio Quality sub-scores
    audio_quality_risk_score = Column(Float, nullable=True)
    low_volume_score = Column(Float, nullable=True)
    noise_score = Column(Float, nullable=True)
    clipping_distortion_score = Column(Float, nullable=True)
    missing_audio_score = Column(Float, nullable=True)

    # Multimodal sub-scores
    multimodal_consistency_risk_score = Column(Float, nullable=True)
    audio_video_sync_score = Column(Float, nullable=True)
    speaker_face_consistency_score = Column(Float, nullable=True)
    response_timing_consistency_score = Column(Float, nullable=True)
    environment_consistency_score = Column(Float, nullable=True)

    # DeepFace emotion risk
    deepface_emotion_risk_score = Column(Float, nullable=True)

    def to_dict(self) -> dict:
        d = {}
        for col in self.__table__.columns:
            val = getattr(self, col.name)
            if isinstance(val, str) and col.name in (
                "aura_transcript", "emotion_percentages", "audio_analysis_json",
                "audio_quality", "voice_emotion", "voice_antispoof",
                "speaker_verification", "coercion_matches", "risk_breakdown"
            ):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            d[col.name] = val
        return d


def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Base de datos SQLite inicializada.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_session_record(db: Session, client_session_id: str, video_path: str) -> AnalysisSession:
    record = AnalysisSession(
        client_session_id=client_session_id,
        video_path=video_path,
        status="Pendiente",
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_session_by_client_id(db: Session, client_session_id: str) -> Optional[AnalysisSession]:
    return db.query(AnalysisSession).filter(
        AnalysisSession.client_session_id == client_session_id
    ).first()


def update_session_status(db: Session, session_id: int, status: str, reason: str = None):
    record = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if record:
        record.status = status
        if reason:
            record.status_reason = reason
        record.updated_at = datetime.utcnow().isoformat()
        db.commit()
    return record
