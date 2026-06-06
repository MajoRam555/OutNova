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

SYSTEM_PROMPT = """Eres AURA, agente de verificación biométrica conversacional. Hablas en español.

PERSONALIDAD: Cálida, perceptiva y directa — como un profesional de salud mental haciendo una entrevista clínica, \
no como un bot siguiendo un script. Escuchas de verdad y reaccionas a lo que se te dice.

REGLA FUNDAMENTAL: Cada respuesta tuya DEBE referenciar algo específico de lo que el usuario acaba de decir. \
Nunca respondas de forma genérica ignorando el contenido de su mensaje.

ESTRUCTURA DE CADA TURNO:
1. Reconocimiento específico (1 oración): menciona algo concreto de lo que dijo. \
   Ejemplos: "Interesante que menciones que estás en la cocina." / "Noto que dudaste un momento antes de responder." \
   / "Tiene sentido que estés un poco nervioso, es normal al inicio."
2. Una sola pregunta de seguimiento (1 oración): avanza el tema o profundiza si algo llamó tu atención.

LÍMITE: Máximo 2-3 oraciones en total. Nada más.

REACCIONA ESPECIALMENTE A:
- Respuestas muy cortas (< 5 palabras): invita amablemente a elaborar antes de pasar al siguiente tema.
- Menciones de otras personas presentes: pregunta directamente si está solo/a.
- Hesitación o "no sé" / "tal vez": reconócelo con empatía, no lo ignores.
- Respuestas que describen estrés, miedo o presión: responde con calma y pregunta si se siente cómodo continuando.
- Respuestas detalladas y fluidas: valóralas genuinamente antes de seguir.

PROHIBIDO:
- Frases robóticas: "Entendido.", "Perfecto.", "Gracias por compartir.", "De acuerdo."
- Repetir la misma estructura de reconocimiento dos veces seguidas.
- Hacer más de una pregunta por turno.
- Respuestas de más de 3 oraciones.
- Cambiar de tema abruptamente si la respuesta anterior fue incompleta o confusa."""

# ── Contextual acknowledgment banks (por tipo de señal) ──────────────────────

_ACK_SHORT = [
    "Entiendo, aunque me gustaría escucharte un poco más sobre eso.",
    "Es una respuesta muy breve — ¿puedes contarme algo más al respecto?",
    "¿Podrías desarrollar un poco esa idea?",
    "Me quedé con ganas de saber más sobre lo que dijiste.",
]

_ACK_STRESS = [
    "Noto que puede ser un momento incómodo — eso es completamente normal.",
    "Está bien si sientes algo de tensión, tomemos el tiempo que necesites.",
    "Entiendo que esto puede sentirse un poco intimidante al principio.",
    "Lo que describes suena como algo que te genera presión. Quiero que sepas que puedes hablar con libertad.",
]

_ACK_HESITATION = [
    "Veo que no tienes total certeza sobre eso, y está bien.",
    "La duda es completamente válida — no hay respuestas incorrectas aquí.",
    "Entiendo que no sea algo fácil de precisar.",
    "No te preocupes si no recuerdas exactamente — lo importante es tu perspectiva.",
]

_ACK_POSITIVE = [
    "Me alegra escuchar eso.",
    "Qué bueno que lo confirmes.",
    "Eso es justo lo que necesitaba saber.",
    "Perfecto, eso me ayuda mucho.",
]

_ACK_DETAILED = [
    "Gracias por ese nivel de detalle, me es muy útil.",
    "Aprecio que hayas sido tan claro/a con eso.",
    "Esa descripción me da una imagen muy completa.",
    "Me ayuda mucho que lo hayas explicado así.",
]

_ACK_OTHER_PERSON = [
    "Mencionaste que hay alguien contigo — quiero asegurarme de algo.",
    "Antes de continuar, quiero confirmar una cosa sobre esa persona que mencionaste.",
]

_ACK_DEFAULT = [
    "Entiendo lo que describes.",
    "Tiene sentido lo que dices.",
    "Me queda claro.",
    "Lo que describes me ayuda a entender mejor el contexto.",
    "Bien, lo tomo en cuenta.",
]


def _analyze_user_text(text: str) -> str:
    """
    Classify the user's response into a signal type for acknowledgment selection.
    Returns: 'short' | 'stress' | 'hesitation' | 'positive' | 'detailed' | 'other_person' | 'default'
    """
    t = text.lower().strip()
    words = t.split()

    if len(words) <= 4:
        return "short"

    if any(kw in t for kw in ["alguien", "acompañado", "acompañada", "hay alguien", "no estoy solo", "no estoy sola"]):
        return "other_person"

    if any(kw in t for kw in ["nervioso", "nerviosa", "miedo", "asustado", "asustada",
                               "presión", "presion", "obligado", "obligada", "me dijeron", "me pidieron"]):
        return "stress"

    if any(kw in t for kw in ["no sé", "no se", "no estoy seguro", "no estoy segura",
                               "tal vez", "quizás", "quizas", "no recuerdo", "no me acuerdo"]):
        return "hesitation"

    if any(kw in t for kw in ["sí", "claro", "por supuesto", "voluntario", "voluntaria",
                               "con gusto", "sin problema", "completamente"]):
        return "positive"

    if len(words) > 25:
        return "detailed"

    return "default"


def _contextual_ack(user_text: str, turn: int) -> str:
    """Return an acknowledgment phrase matched to the user's response signal."""
    signal = _analyze_user_text(user_text)
    bank_map = {
        "short": _ACK_SHORT,
        "stress": _ACK_STRESS,
        "hesitation": _ACK_HESITATION,
        "positive": _ACK_POSITIVE,
        "detailed": _ACK_DETAILED,
        "other_person": _ACK_OTHER_PERSON,
        "default": _ACK_DEFAULT,
    }
    bank = bank_map.get(signal, _ACK_DEFAULT)
    return bank[turn % len(bank)]


def _needs_followup(user_text: str, phase: str) -> Optional[str]:
    """
    If the user's response warrants a follow-up instead of advancing,
    return the follow-up question. Otherwise return None.
    """
    t = user_text.lower().strip()
    words = t.split()

    # Very short answer in a scoring phase — ask for more
    if len(words) <= 3 and phase in ("sensitive_dilemma", "identity_context"):
        return "¿Podrías contarme un poco más sobre eso?"

    # Someone else is present — always probe this
    if any(kw in t for kw in ["alguien", "acompañado", "acompañada", "hay alguien", "no estoy solo", "no estoy sola"]):
        return "¿Esa persona está cerca de ti en este momento, o en otro cuarto?"

    # Stress signal — check if they want to continue
    if any(kw in t for kw in ["me dijeron", "me pidieron", "me mandaron", "me obligaron",
                               "me presionaron", "no quería", "no queria"]):
        return "Escucho que hay algo detrás de eso. ¿Estás participando en esta verificación por decisión propia?"

    return None

# ── Phase-aware conversation structure ───────────────────────────────────────
#
# Each turn is a dict with:
#   text            — the question AURA asks
#   phase           — conversation phase (practice / baseline / identity_context /
#                     sensitive_dilemma / closing)
#   question_type   — warm_up / environment / cognitive / consent / identity /
#                     behavioral / dilemma / closing
#   counts_for_score— whether this turn's response contributes to scoring
#   expected_signal — what behavioral signal we're probing (or None)

PHASE_TURNS = [
    # ── PHASE 1: Practice (~30-45s, no scoring) ─────────────────────────────
    {
        "text": "Hola, soy AURA. Voy a acompañarte en esta verificación. Antes de comenzar, hagamos una pequeña prueba. ¿Puedes decirme cómo está el tiempo hoy donde tú estás?",
        "phase": "practice",
        "question_type": "warm_up",
        "counts_for_score": False,
        "expected_signal": None,
    },
    {
        "text": "Perfecto, ya vi que el audio funciona bien. ¿Puedes decirme en voz alta el nombre del mes en que naciste?",
        "phase": "practice",
        "question_type": "warm_up",
        "counts_for_score": False,
        "expected_signal": None,
    },
    # ── PHASE 2: Baseline (~60s, only baseline metrics) ──────────────────────
    {
        "text": "Muy bien. Ahora sí comenzamos formalmente. ¿Puedes describirme brevemente el lugar donde te encuentras en este momento?",
        "phase": "baseline",
        "question_type": "environment",
        "counts_for_score": True,
        "expected_signal": "baseline_speech_pattern",
    },
    {
        "text": "¿Qué objeto tienes más cerca de ti ahora mismo?",
        "phase": "baseline",
        "question_type": "environment",
        "counts_for_score": True,
        "expected_signal": "baseline_speech_pattern",
    },
    {
        "text": "Vamos a hacer un pequeño ejercicio. ¿Puedes deletrear la palabra CASA al revés?",
        "phase": "baseline",
        "question_type": "cognitive",
        "counts_for_score": True,
        "expected_signal": "cognitive_baseline",
    },
    # ── PHASE 3: Identity context (~45s, low weight) ─────────────────────────
    {
        "text": "Gracias. Ahora pasamos a la parte de identidad. ¿Puedes confirmarme tu nombre completo?",
        "phase": "identity_context",
        "question_type": "identity",
        "counts_for_score": True,
        "expected_signal": "identity_consistency",
    },
    {
        "text": "¿Cuál es la fecha de hoy con tus propias palabras?",
        "phase": "identity_context",
        "question_type": "cognitive",
        "counts_for_score": True,
        "expected_signal": "temporal_orientation",
    },
    {
        "text": "¿Hay alguien más contigo en este momento, o estás solo/a?",
        "phase": "identity_context",
        "question_type": "consent",
        "counts_for_score": True,
        "expected_signal": "coercion_context",
    },
    # ── PHASE 4: Sensitive dilemma (2-3 min, main scoring) ───────────────────
    {
        "text": "Quiero asegurarme de algo importante. ¿Estás realizando esta verificación por tu propia voluntad, sin que nadie te haya pedido o presionado que lo hagas?",
        "phase": "sensitive_dilemma",
        "question_type": "consent",
        "counts_for_score": True,
        "expected_signal": "coercion_pressure",
    },
    {
        "text": "Te voy a hacer una pregunta un poco más personal. Si en algún momento durante esta verificación sintieras que no puedes hablar con libertad, ¿me lo harías saber de alguna manera?",
        "phase": "sensitive_dilemma",
        "question_type": "behavioral",
        "counts_for_score": True,
        "expected_signal": "coercion_pressure",
    },
    {
        "text": "¿Qué estabas haciendo justo antes de comenzar este proceso hoy?",
        "phase": "sensitive_dilemma",
        "question_type": "behavioral",
        "counts_for_score": True,
        "expected_signal": "narrative_consistency",
    },
    {
        "text": "¿Puedes contarme, con tus propias palabras, por qué estás realizando esta verificación hoy?",
        "phase": "sensitive_dilemma",
        "question_type": "dilemma",
        "counts_for_score": True,
        "expected_signal": "narrative_coherence",
    },
    {
        "text": "Si en algún momento descubrieras que hay un error en tu cuenta o en algún proceso financiero que te involucra, ¿qué harías?",
        "phase": "sensitive_dilemma",
        "question_type": "dilemma",
        "counts_for_score": True,
        "expected_signal": "rationalization_opportunity",
    },
    # ── PHASE 5: Closing (~15s, no scoring) ──────────────────────────────────
    {
        "text": "Gracias, estamos terminando. ¿Confirmas que todo lo que compartiste hoy es verdadero y que participaste de forma libre?",
        "phase": "closing",
        "question_type": "closing",
        "counts_for_score": False,
        "expected_signal": None,
    },
    {
        "text": "Perfecto. La verificación ha concluido. Muchas gracias por tu tiempo y cooperación.",
        "phase": "closing",
        "question_type": "closing",
        "counts_for_score": False,
        "expected_signal": None,
    },
]

# Flat list of texts for backward-compatible fallback
FALLBACK_TURNS = [t["text"] for t in PHASE_TURNS]


@dataclass
class AuraChatSession:
    client_session_id: str
    messages: list = field(default_factory=list)
    turn_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    used_fallback: bool = False
    model_status: str = "not_loaded"
    current_phase: str = "practice"
    turn_index: int = 0

    def _current_turn_meta(self) -> dict:
        """Return phase metadata for the current turn index."""
        idx = self.turn_index % len(PHASE_TURNS)
        t = PHASE_TURNS[idx]
        return {
            "phase": t["phase"],
            "question_type": t["question_type"],
            "counts_for_score": t["counts_for_score"],
            "expected_signal": t["expected_signal"],
        }

    def add_message(self, role: str, content: str, source: Optional[str] = None):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if source:
            entry["source"] = source
        # Attach phase metadata to user responses (for scoring)
        if role == "user" and self.turn_index > 0:
            meta = self._current_turn_meta()
            entry.update(meta)
            self.current_phase = meta["phase"]
        # Advance turn index when AURA asks a question
        if role == "aura":
            next_idx = self.turn_index + 1
            if next_idx < len(PHASE_TURNS):
                self.current_phase = PHASE_TURNS[next_idx]["phase"]
            self.turn_index = next_idx
        self.messages.append(entry)
        self.updated_at = datetime.utcnow().isoformat()

    def get_transcript_json(self) -> str:
        return json.dumps(self.messages, ensure_ascii=False)

    def build_llm_history(self, user_text: str = "") -> list:
        # Inject current phase context into the system prompt
        phase_ctx = (
            f"\n\nFASE ACTUAL: {self.current_phase}. "
            + {
                "practice": "Fase de calentamiento — tono muy relajado, sin presión.",
                "baseline": "Estableciendo baseline — preguntas neutrales y conversacionales.",
                "identity_context": "Contexto de identidad — tono profesional, verificar datos.",
                "sensitive_dilemma": "Fase crítica de evaluación — pon atención a incoherencias, evasión o señales de coerción.",
                "closing": "Cierre — tono cálido y tranquilizador.",
            }.get(self.current_phase, "")
        )
        if user_text:
            followup = _needs_followup(user_text, self.current_phase)
            signal = _analyze_user_text(user_text)
            phase_ctx += (
                f"\n\nRESPUESTA ACTUAL DEL USUARIO: \"{user_text}\""
                f"\nSEÑAL DETECTADA: {signal}"
            )
            if followup:
                phase_ctx += f"\nSUGERENCIA DE SEGUIMIENTO: considera preguntar: «{followup}»"

        history = [{"role": "system", "content": SYSTEM_PROMPT + phase_ctx}]
        for msg in self.messages[-12:]:
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

            history = session.build_llm_history(user_text=user_text)

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
        """
        Context-aware rule-based fallback.
        - Checks if a follow-up is warranted before advancing to the next scripted question.
        - Uses signal-matched acknowledgment phrases instead of generic ones.
        Never raises.
        """
        is_first_turn = turn == 0 or not user_text.strip()

        if is_first_turn:
            return FALLBACK_TURNS[0]

        # Check if the response warrants staying on the same topic
        followup = _needs_followup(user_text, session.current_phase)
        if followup:
            ack = _contextual_ack(user_text, turn)
            return f"{ack} {followup}"

        # Advance to next scripted question
        idx = min(turn, len(FALLBACK_TURNS) - 1)
        question = FALLBACK_TURNS[idx]
        ack = _contextual_ack(user_text, turn)
        return f"{ack} {question}"

    def generate_reply(self, session: AuraChatSession, user_text: str) -> str:
        """
        Main entry point. Uses LLM if ready, else context-aware fallback.
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
