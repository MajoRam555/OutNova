"""
AURA — Agente conversacional de verificación biométrica.
Usa LLM local (Qwen2.5-0.5B) con fallback por reglas.
"""
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import AURA_LLM_FALLBACK, AURA_LLM_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Eres AURA, agente conversacional de verificación biométrica. "
    "Habla en español con un tono cálido, profesional y natural — como un asesor de confianza, no como un sistema automático. "
    "Tu misión: guiar al usuario por una entrevista de verificación de identidad, generando respuestas habladas espontáneas "
    "y detectando señales de coerción, evasión o incoherencia. "
    "Estructura de cada turno: (1) reconoce brevemente lo que dijo el usuario con una frase natural y empática, "
    "(2) haz exactamente UNA pregunta relevante. Máximo 2-3 oraciones en total. "
    "Varía el tono: a veces curioso, a veces reflexivo, siempre humano. "
    "No uses frases robóticas ni repetitivas. Adapta el contenido según las respuestas anteriores. "
    "No repitas preguntas ya hechas. Alterna entre estos temas: "
    "entorno físico actual, consentimiento libre, atención cognitiva, identidad personal y bienestar emocional."
)

# Acknowledgment phrases — picked by turn index (deterministic, no random)
_ACKS = [
    "Gracias, lo tomo en cuenta.",
    "Entendido, perfecto.",
    "Bien, te escucho.",
    "De acuerdo, gracias.",
    "Muy bien.",
    "Interesante, gracias por compartirlo.",
    "Perfecto, anotado.",
    "Claro, lo entiendo.",
]

# Fallback questions — phrased naturally, with soft connectors
FALLBACK_TURNS = [
    "Hola, soy AURA. Voy a acompañarte en este proceso de verificación. ¿Puedes confirmarme que estás haciendo esto de forma voluntaria?",
    "¿Puedes describirme brevemente el lugar donde te encuentras ahora mismo?",
    "Vamos a hacer un pequeño ejercicio de atención: ¿podrías deletrear la palabra MESA al revés?",
    "¿Qué objeto tienes más cerca de ti en este momento?",
    "¿Qué estabas haciendo justo antes de comenzar esta verificación?",
    "¿Hay alguien más contigo en la habitación en este momento?",
    "¿Puedes mover ligeramente la cabeza hacia los lados y decirme cómo te sientes hoy?",
    "¿Puedes confirmarme que esta decisión la estás tomando de forma completamente personal, sin ninguna presión externa?",
    "¿Puedes decirme tu nombre completo?",
    "¿Puedes decirme cuál es la fecha de hoy con tus propias palabras?",
    "¿Te encuentras en un lugar tranquilo y privado en este momento?",
    "Estamos llegando al final del proceso. ¿Tienes alguna pregunta sobre esta verificación?",
]


@dataclass
class AuraChatSession:
    client_session_id: str
    messages: list = field(default_factory=list)
    turn_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    used_fallback: bool = False
    model_status: str = "not_loaded"

    def add_message(self, role: str, content: str, source: Optional[str] = None):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if source:
            entry["source"] = source
        self.messages.append(entry)
        self.updated_at = datetime.utcnow().isoformat()

    def get_transcript_json(self) -> str:
        return json.dumps(self.messages, ensure_ascii=False)

    def build_llm_history(self) -> list:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in self.messages[-10:]:  # last 10 turns for context window
            role = "assistant" if msg["role"] == "aura" else "user"
            history.append({"role": role, "content": msg["content"]})
        return history


class AuraEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self.llm = None
        self.tokenizer = None
        self.llm_ready = False
        self.tts_ready = False
        self.used_fallback = False
        self.load_error = None
        self.loaded_model = None

    def status(self) -> dict:
        return {
            "llm_ready": self.llm_ready,
            "tts_ready": self.tts_ready,
            "used_fallback": self.used_fallback,
            "load_error": str(self.load_error) if self.load_error else None,
            "loaded_model": self.loaded_model,
        }

    def load(self):
        """Load LLM model. Called from _aura_executor to avoid blocking startup."""
        self.load_llm_only()

    def load_llm_only(self):
        """Try to load Qwen2.5, fallback to Phi-3, fallback to rules."""
        models_to_try = [AURA_LLM_MODEL, AURA_LLM_FALLBACK]
        for model_id in models_to_try:
            try:
                logger.info(f"[AURA] Intentando cargar LLM: {model_id}")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                model.eval()

                with self._lock:
                    self.tokenizer = tokenizer
                    self.llm = model
                    self.llm_ready = True
                    self.loaded_model = model_id
                    self.load_error = None

                logger.info(f"[AURA] LLM listo: {model_id}")
                return

            except Exception as e:
                logger.warning(f"[AURA] No se pudo cargar {model_id}: {e}")
                self.load_error = e

        logger.warning("[AURA] Ningún LLM cargó. Usando fallback por reglas.")
        with self._lock:
            self.llm_ready = False
            self.used_fallback = True

    def _generate_llm(self, session: AuraChatSession, user_text: str) -> str:
        """Generate response using loaded LLM."""
        try:
            import torch

            history = session.build_llm_history()
            if user_text:
                history.append({"role": "user", "content": user_text})

            # Try chat template
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    history,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except Exception:
                # Fallback: concat as plain text
                text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
                text += "\nassistant:"
                input_ids = self.tokenizer.encode(text, return_tensors="pt")

            with torch.no_grad():
                output = self.llm.generate(
                    input_ids,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.75,
                    top_p=0.92,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = output[0][input_ids.shape[-1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Cap at 50 words (allows acknowledgment + one question naturally)
            words = text.split()
            if len(words) > 50:
                text = " ".join(words[:50]) + "."

            return text if text else self.fallback_reply(session, session.turn_count, user_text)

        except Exception as e:
            logger.warning(f"[AURA] LLM generate error: {e}")
            return self.fallback_reply(session, session.turn_count, user_text)

    def fallback_reply(self, session: AuraChatSession, turn: int, user_text: str = "") -> str:
        """Rule-based fallback. Adds acknowledgment for turns > 0. Never raises."""
        idx = turn % len(FALLBACK_TURNS)
        question = FALLBACK_TURNS[idx]
        if turn > 0 and user_text and user_text.strip():
            ack = _ACKS[turn % len(_ACKS)]
            return f"{ack} {question}"
        return question

    def generate_reply(self, session: AuraChatSession, user_text: str) -> str:
        """
        Main entry point. Uses LLM if ready, else fallback.
        Never raises — always returns a string.
        """
        with self._lock:
            llm_ready = self.llm_ready

        if llm_ready:
            reply = self._generate_llm(session, user_text)
        else:
            reply = self.fallback_reply(session, session.turn_count, user_text)
            session.used_fallback = True

        return reply


# Singleton
_engine = AuraEngine()

# In-memory sessions: {client_session_id: AuraChatSession}
_sessions: dict[str, AuraChatSession] = {}
_sessions_lock = threading.Lock()


def get_engine() -> AuraEngine:
    return _engine


def get_or_create_session(client_session_id: str) -> AuraChatSession:
    with _sessions_lock:
        if client_session_id not in _sessions:
            _sessions[client_session_id] = AuraChatSession(
                client_session_id=client_session_id
            )
        return _sessions[client_session_id]


def get_session(client_session_id: str) -> Optional[AuraChatSession]:
    with _sessions_lock:
        return _sessions.get(client_session_id)


def pop_session(client_session_id: str) -> Optional[AuraChatSession]:
    with _sessions_lock:
        return _sessions.pop(client_session_id, None)
