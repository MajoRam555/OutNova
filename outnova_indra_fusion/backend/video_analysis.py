import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HAAR_EYE = None
HAAR_FACE = None


def _load_cascades():
    global HAAR_FACE, HAAR_EYE
    try:
        import cv2
        if HAAR_FACE is None:
            HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if HAAR_EYE is None:
            HAAR_EYE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    except Exception as e:
        logger.warning(f"OpenCV cascade error: {e}")


def convert_webm_to_mp4(webm_path: str) -> str | None:
    """Converts WebM to MP4. Tries ffmpeg, then moviepy, then imageio_ffmpeg."""
    mp4_path = str(Path(webm_path).with_suffix(".mp4"))

    # Method 1: ffmpeg system
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-c:v", "libx264", "-c:a", "aac", mp4_path],
            capture_output=True, timeout=180
        )
        if result.returncode == 0 and Path(mp4_path).exists():
            logger.info(f"WebM→MP4 ffmpeg OK: {mp4_path}")
            return mp4_path
        logger.warning(f"ffmpeg convert error: {result.stderr.decode()[:200]}")
    except Exception as e:
        logger.warning(f"ffmpeg convert exception: {e}")

    # Method 2: moviepy
    try:
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip
        clip = VideoFileClip(webm_path)
        clip.write_videofile(mp4_path, codec="libx264", audio_codec="aac", logger=None)
        clip.close()
        if Path(mp4_path).exists():
            logger.info(f"WebM→MP4 moviepy OK: {mp4_path}")
            return mp4_path
    except Exception as e:
        logger.warning(f"moviepy convert error: {e}")

    # Method 3: imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [ff, "-y", "-i", webm_path, "-c:v", "libx264", "-c:a", "aac", mp4_path],
            capture_output=True, timeout=180
        )
        if result.returncode == 0 and Path(mp4_path).exists():
            logger.info(f"WebM→MP4 imageio_ffmpeg OK: {mp4_path}")
            return mp4_path
    except Exception as e:
        logger.warning(f"imageio_ffmpeg convert error: {e}")

    logger.error("No se pudo convertir WebM a MP4.")
    return None


def detect_blinks(video_path: str) -> dict:
    """
    Uses OpenCV Haar cascades to detect blinks and compute liveness metrics.
    """
    _load_cascades()
    result = {
        "total_blinks": 0,
        "blink_rate_per_min": 0.0,
        "blink_cv": 0.0,
        "face_detected_ratio": 0.0,
        "eye_detected_ratio": 0.0,
        "avg_ear": 0.0,
        "video_duration_sec": 0.0,
        "deepfake_flag": False,
        "deepfake_reason": "Análisis de liveness normal.",
        "error": None,
    }

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result["error"] = "No se pudo abrir video con OpenCV."
            return result

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        result["video_duration_sec"] = round(duration_sec, 2)

        face_count = 0
        eye_count = 0
        frames_processed = 0
        eye_states = []  # True=eyes detected, False=no eyes

        # Sample every 3rd frame for performance
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 3 != 0:
                continue

            frames_processed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if HAAR_FACE is None:
                continue

            faces = HAAR_FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                face_count += 1
                # Check eyes in first face
                (x, y, w, h) = faces[0]
                roi = gray[y:y + h, x:x + w]
                eyes = HAAR_EYE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
                if len(eyes) >= 2:
                    eye_count += 1
                    eye_states.append(True)
                else:
                    eye_states.append(False)
            else:
                eye_states.append(False)

        cap.release()

        if frames_processed == 0:
            result["error"] = "Sin frames procesados"
            return result

        result["face_detected_ratio"] = round(face_count / frames_processed, 4)
        result["eye_detected_ratio"] = round(eye_count / frames_processed, 4)

        # Blink detection: eye state transitions True→False→True
        blinks = 0
        blink_intervals = []
        in_blink = False
        last_blink_frame = None

        for i in range(1, len(eye_states)):
            if eye_states[i - 1] and not eye_states[i]:
                in_blink = True
            elif in_blink and eye_states[i]:
                blinks += 1
                if last_blink_frame is not None:
                    blink_intervals.append(i - last_blink_frame)
                last_blink_frame = i
                in_blink = False

        result["total_blinks"] = blinks
        if duration_sec > 0:
            result["blink_rate_per_min"] = round(blinks / (duration_sec / 60.0), 2)

        if len(blink_intervals) > 1:
            intervals_arr = np.array(blink_intervals, dtype=float)
            result["blink_cv"] = round(float(np.std(intervals_arr) / (np.mean(intervals_arr) + 1e-10)), 4)

        # Deepfake heuristic
        rate = result["blink_rate_per_min"]
        face_ratio = result["face_detected_ratio"]
        eye_ratio = result["eye_detected_ratio"]
        blink_cv = result["blink_cv"]

        deepfake_reasons = []
        if duration_sec > 30 and blinks == 0:
            deepfake_reasons.append("0 parpadeos en video largo")
        if rate > 0 and rate < 3.0:
            deepfake_reasons.append(f"blink rate muy bajo ({rate:.1f}/min)")
        if rate > 60.0:
            deepfake_reasons.append(f"blink rate extremadamente alto ({rate:.1f}/min)")
        if blink_cv < 0.08 and blinks > 5:
            deepfake_reasons.append("parpadeo mecánico (CV demasiado bajo)")
        if face_ratio < 0.30 and duration_sec > 10:
            deepfake_reasons.append(f"cara casi nunca detectada ({face_ratio:.0%})")
        if eye_ratio < 0.20 and duration_sec > 10:
            deepfake_reasons.append(f"ojos casi nunca detectados ({eye_ratio:.0%})")

        if deepfake_reasons:
            result["deepfake_flag"] = True
            result["deepfake_reason"] = "Posible deepfake/no-liveness: " + "; ".join(deepfake_reasons)
        else:
            result["deepfake_reason"] = "Sin indicadores de deepfake."

        logger.info(
            f"[VIDEO] blinks={blinks}, rate={rate:.1f}/min, "
            f"face={face_ratio:.0%}, deepfake={result['deepfake_flag']}"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[VIDEO] detect_blinks error: {e}")

    return result


def optional_deepface_analysis(video_path: str, sample_every_n_sec: float = 5.0) -> dict:
    """
    Optional DeepFace emotion analysis. Returns empty result if not available.
    """
    try:
        from deepface import DeepFace
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_interval = int(fps * sample_every_n_sec)

        emotion_accumulator = {}
        samples = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_interval != 0:
                continue

            try:
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)
                if isinstance(result, list):
                    result = result[0]
                emotions = result.get("emotion", {})
                for emo, val in emotions.items():
                    emotion_accumulator[emo] = emotion_accumulator.get(emo, 0.0) + val
                samples += 1
            except Exception:
                pass

        cap.release()

        if samples == 0:
            return {"deepface_available": True, "emotion_percentages": {}, "dominant_emotion": "unknown",
                    "stress_emotion_ratio": 0.0}

        avg_emotions = {k: round(v / samples, 2) for k, v in emotion_accumulator.items()}
        dominant = max(avg_emotions, key=avg_emotions.get) if avg_emotions else "unknown"

        stress_emotions = {"fear", "angry", "disgust", "sad"}
        stress_sum = sum(avg_emotions.get(e, 0) for e in stress_emotions)
        total_sum = sum(avg_emotions.values()) or 1
        stress_ratio = stress_sum / total_sum

        return {
            "deepface_available": True,
            "emotion_percentages": avg_emotions,
            "dominant_emotion": dominant,
            "stress_emotion_ratio": round(stress_ratio, 4),
        }

    except ImportError:
        return {"deepface_available": False, "emotion_percentages": {}, "dominant_emotion": None,
                "stress_emotion_ratio": 0.0}
    except Exception as e:
        logger.warning(f"DeepFace error: {e}")
        return {"deepface_available": False, "emotion_percentages": {}, "dominant_emotion": None,
                "stress_emotion_ratio": 0.0, "error": str(e)}


def analyze_video(video_path: str) -> dict:
    """
    Full video analysis pipeline. Converts WebM→MP4 if needed, runs blink/liveness,
    optionally runs DeepFace.
    """
    from config import USE_DEEPFACE

    result = {
        "video_duration_sec": 0.0,
        "total_blinks": 0,
        "blink_rate_per_min": 0.0,
        "blink_cv": 0.0,
        "face_detected_ratio": 0.0,
        "eye_detected_ratio": 0.0,
        "avg_ear": 0.0,
        "deepfake_flag": False,
        "deepfake_reason": "No analizado.",
        "emotion_percentages": {},
        "dominant_emotion": None,
        "stress_emotion_ratio": 0.0,
        "deepface_available": False,
        "errors": [],
    }

    # Convert if WebM
    analysis_path = video_path
    if video_path.lower().endswith(".webm"):
        logger.info(f"[VIDEO] Convirtiendo WebM→MP4: {video_path}")
        mp4_path = convert_webm_to_mp4(video_path)
        if mp4_path:
            analysis_path = mp4_path
        else:
            result["errors"].append("Conversión WebM→MP4 fallida. Intentando con WebM directamente.")

    # Blink/liveness
    logger.info(f"[VIDEO] Detectando parpadeos y liveness: {analysis_path}")
    try:
        blink_data = detect_blinks(analysis_path)
        result.update({
            "video_duration_sec": blink_data.get("video_duration_sec", 0.0),
            "total_blinks": blink_data.get("total_blinks", 0),
            "blink_rate_per_min": blink_data.get("blink_rate_per_min", 0.0),
            "blink_cv": blink_data.get("blink_cv", 0.0),
            "face_detected_ratio": blink_data.get("face_detected_ratio", 0.0),
            "eye_detected_ratio": blink_data.get("eye_detected_ratio", 0.0),
            "deepfake_flag": blink_data.get("deepfake_flag", False),
            "deepfake_reason": blink_data.get("deepfake_reason", ""),
        })
        if blink_data.get("error"):
            result["errors"].append(f"blink_detection: {blink_data['error']}")
    except Exception as e:
        result["errors"].append(f"detect_blinks: {e}")
        logger.error(f"[VIDEO] detect_blinks exception: {e}")

    # DeepFace (optional)
    if USE_DEEPFACE:
        logger.info("[VIDEO] Ejecutando DeepFace...")
        try:
            df_data = optional_deepface_analysis(analysis_path)
            result.update({
                "deepface_available": df_data.get("deepface_available", False),
                "emotion_percentages": df_data.get("emotion_percentages", {}),
                "dominant_emotion": df_data.get("dominant_emotion"),
                "stress_emotion_ratio": df_data.get("stress_emotion_ratio", 0.0),
            })
        except Exception as e:
            result["errors"].append(f"deepface: {e}")
            logger.warning(f"[VIDEO] DeepFace error: {e}")
    else:
        # Try import to detect availability
        try:
            import deepface
            result["deepface_available"] = True
        except ImportError:
            result["deepface_available"] = False

    logger.info(
        f"[VIDEO] Análisis completo — duration={result['video_duration_sec']:.1f}s, "
        f"blinks={result['total_blinks']}, deepfake={result['deepfake_flag']}"
    )
    return result
