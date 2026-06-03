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
    "Eres AURA, agente conversacional de verificación biométrica y riesgo financiero. "
    "Habla en español, de forma profesional, breve y empática. "
    "Tu objetivo es guiar una validación de identidad, provocar respuestas habladas naturales, "
    "observar coherencia y detectar señales de coerción o evasión. "
    "No excedas 20 palabras por turno. Haz una sola pregunta por turno. "
    "No repitas preguntas. Alterna preguntas de entorno, atención, consentimiento y contexto."
)

# Fallback question bank
FALLBACK_TURNS = [
    "Hola, soy AURA. ¿Puedes confirmar que realizas esta verificación de forma voluntaria?",
    "¿Puedes describirme brevemente qué hay detrás de ti en este momento?",
    "Para confirmar tu atención, ¿podrías deletrear la palabra MESA al revés?",
    "¿Qué objeto tienes más cerca de ti en este momento?",
    "¿Qué hiciste antes de comenzar esta verificación?",
    "¿Hay alguien en la habitación indicándote qué responder?",
    "¿Puedes mover ligeramente la cabeza y decirme cómo te sientes hoy?",
    "¿Confirmas que la operación que estás realizando es tu decisión personal?",
    "¿Cuál es tu nombre completo para confirmar identidad?",
    "¿Puedes decirme la fecha de hoy con tus propias palabras?",
    "¿Estás en un lugar seguro y privado para continuar?",
    "Gracias por tu tiempo. ¿Tienes alguna duda sobre este proceso?",
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

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
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
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = output[0][input_ids.shape[-1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Truncate to 20 words
            words = text.split()
            if len(words) > 20:
                text = " ".join(words[:20]) + "..."

            return text if text else self.fallback_reply(session, session.turn_count, user_text)

        except Exception as e:
            logger.warning(f"[AURA] LLM generate error: {e}")
            return self.fallback_reply(session, session.turn_count, user_text)

    def fallback_reply(self, session: AuraChatSession, turn: int, user_text: str = "") -> str:
        """Rule-based fallback. Never raises."""
        idx = turn % len(FALLBACK_TURNS)
        return FALLBACK_TURNS[idx]

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
