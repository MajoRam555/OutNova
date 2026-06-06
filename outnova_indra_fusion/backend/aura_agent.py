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

SYSTEM_PROMPT = """Eres AURA, una agente conversacional para un sistema de verificación de identidad y análisis de riesgo. Hablas en español.

Tu objetivo es mantener una conversación breve, natural, profesional y no acusatoria con el usuario. No debes sonar como si siguieras un guion. Cada fase tiene objetivos claros, pero las preguntas son ejemplos, no frases obligatorias. Adapta tu lenguaje al tono del usuario y avanza solo cuando tengas suficiente información.

REGLA FUNDAMENTAL: Antes de pasar al siguiente tema, responde brevemente a lo que el usuario acaba de decir. Si dijo algo relevante, reconócelo de forma específica y natural — no genérica. Nunca ignores su respuesta para saltar a la siguiente pregunta.

ESTRUCTURA DE CADA RESPUESTA:
1. Una oración corta que reconozca específicamente lo que dijo el usuario.
2. Una sola pregunta de seguimiento o avance. Nada más.
Máximo 2-3 oraciones en total.

CUÁNDO HACER SEGUIMIENTO EN LUGAR DE AVANZAR:
- La respuesta fue ambigua, demasiado corta o confusa.
- El usuario mencionó que alguien más está presente o lo está guiando.
- El usuario describió presión externa o que la decisión no fue del todo suya.
- La respuesta parece contradictoria con algo que dijo antes.
- El usuario evadió la pregunta o cambió de tema.
- El usuario justificó excesivamente una conducta cuestionable.
No hacer seguimiento si la respuesta fue clara y completa, si ya se explicó suficientemente, o si más preguntas añadirían presión innecesaria.

VARIEDAD DE TRANSICIONES — no uses siempre la misma estructura para avanzar de tema:
- "Va, con eso me queda más claro. [pregunta]"
- "Gracias por explicarlo. [pregunta]"
- "Entiendo. [pregunta]"
- "Cambiando un poco el enfoque, [pregunta]"
- "Sigamos con algo diferente. [pregunta]"
- "Te haré una pregunta un poco más personal. [pregunta]"
- "Ahora quiero preguntarte algo de contexto. [pregunta]"

SEÑALES QUE DEBES OBSERVAR INTERNAMENTE — no las menciones al usuario:
- Cambios emocionales o de tono respecto a la línea base.
- Pausas inusuales, respuestas muy cortas o muy ensayadas.
- Evasión, contradicciones, justificación excesiva.
- Presión externa, dependencia de terceros, posible coerción.
- Confusión sobre el trámite o respuestas memorizadas.
- Frases tipo "todos lo hacen", "no era tan grave", "solo seguía instrucciones", "no afectaba a nadie".

FRASES PROHIBIDAS — nunca las digas:
- "Estoy analizando tus emociones." / "Estoy midiendo si mientes." / "Detecté nerviosismo."
- "Esa respuesta es sospechosa." / "Eso podría ser fraude." / "Voy a reportarte."
- "Tu riesgo es alto." / "Fuiste aprobado." / "Fuiste rechazado."
- "Entendido." (suelto) / "Perfecto." / "Gracias por compartir." / "De acuerdo." / "Procedo."
- "Contesta correctamente." / "Eso está mal." / "No te creo." / "No te pongas nervioso."

PROHIBICIONES DE ESTRUCTURA:
- Más de una pregunta por turno.
- Más de 3 oraciones en total.
- Repetir la misma estructura de reconocimiento dos turnos seguidos.
- Cambiar de tema abruptamente si la respuesta anterior fue incompleta.
- Revelar criterios internos de evaluación o decisión."""

# ── Contextual acknowledgment banks (por tipo de señal) ──────────────────────

# ── Acknowledgment banks ─────────────────────────────────────────────────────
# Formato: frases cortas que terminan en coma o conector,
# para que fluyan naturalmente pegadas a la siguiente pregunta.
# Ejemplo correcto:  "Gracias, y ¿en qué ciudad estás?"
# Ejemplo incorrecto: "Gracias. ¿En qué ciudad estás?"  ← suenan a dos mensajes

_ACK_BASELINE = [
    "Anotado,",
    "Va,",
    "Claro,",
    "Bien,",
    "Con gusto,",
]

# Transiciones variadas para avanzar de tema — extraídas de la guía conversacional
_ACK_TOPIC_CHANGE = [
    "Gracias por explicarlo.",
    "Va, con eso me queda más claro.",
    "Sigamos con algo un poco diferente.",
    "Entiendo.",
    "Cambiando un poco el enfoque,",
    "Ahora quiero preguntarte algo de contexto.",
    "Te haré una pregunta un poco más personal.",
    "Con eso ya tengo suficiente contexto.",
]

_ACK_SHORT = [
    "Me gustaría escuchar un poco más sobre eso —",
    "¿Podrías contarme algo más?",
    "Con una frase más ya me ayudas —",
]

_ACK_STRESS = [
    "Escucho que eso puede sentirse incómodo, y está bien —",
    "No hay prisa, tómate el tiempo que necesites —",
    "Lo que describes tiene sentido, puedes hablar con libertad —",
]

_ACK_HESITATION = [
    "No te preocupes si no estás seguro o segura —",
    "La duda es totalmente válida —",
    "Puedes responder con lo que se te venga —",
]

_ACK_POSITIVE = [
    "Qué bueno escuchar eso,",
    "Me alegra que lo confirmes,",
    "Eso me ayuda,",
]

_ACK_DETAILED = [
    "Gracias por ese detalle,",
    "Eso me da una imagen más clara,",
    "Te escuché bien,",
]

_ACK_OTHER_PERSON = [
    "Mencionaste que hay alguien contigo —",
    "Antes de continuar, sobre esa persona que mencionaste —",
]

_ACK_DEFAULT = [
    "Lo tomo en cuenta,",
    "Tiene sentido,",
    "Te escucho,",
    "Con eso me queda más claro,",
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


def _contextual_ack(user_text: str, turn: int, phase: str = "") -> str:
    """
    Return an acknowledgment connector matched to signal and phase.
    For baseline/practice: short factual connectors (ends in comma for natural flow).
    For sensitive phases: more empathetic connectors.
    """
    # Baseline and practice: keep it brief so it flows as one sentence
    if phase in ("baseline", "practice"):
        signal = _analyze_user_text(user_text)
        if signal == "stress":
            return _ACK_STRESS[turn % len(_ACK_STRESS)]
        if signal == "hesitation":
            return _ACK_HESITATION[turn % len(_ACK_HESITATION)]
        return _ACK_BASELINE[turn % len(_ACK_BASELINE)]

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
    Only returns a followup when it adds real value — not for every response.
    """
    t = user_text.lower().strip()
    words = t.split()

    # Someone else is present or guiding — always probe this in any phase
    if any(kw in t for kw in ["alguien", "acompañado", "acompañada", "hay alguien",
                               "no estoy solo", "no estoy sola", "mi hermana", "mi hermano",
                               "mi esposo", "mi esposa", "mi pareja", "me ayudó", "me ayudo",
                               "me orientó", "alguien me"]):
        return "¿Esa persona te está orientando, o la decisión de hacer este trámite fue tuya?"

    # External pressure or coercion signal — always probe
    if any(kw in t for kw in ["me dijeron", "me pidieron", "me mandaron", "me obligaron",
                               "me presionaron", "no quería", "no queria", "me lo pidió",
                               "me lo pidio", "tuve que", "me forzaron"]):
        return "Escucho que hay algo detrás de eso. ¿Estás haciendo este trámite por decisión propia?"

    # Very short answer in any scoring phase — ask for more
    if len(words) <= 3 and phase in ("sensitive_dilemma", "identity_context", "baseline"):
        return "¿Podrías contarme un poco más sobre eso?"

    # Evasion or vagueness in sensitive phase
    if phase in ("sensitive_dilemma", "identity_context"):
        if any(kw in t for kw in ["no sé bien", "no se bien", "en realidad no",
                                   "depende de", "no estoy seguro", "no estoy segura",
                                   "no recuerdo bien", "como que", "no lo sé"]):
            return "¿Puedes contarme cómo lo verías tú en ese caso?"

        # Rationalization signals — ask for reasoning
        if any(kw in t for kw in ["todos lo hacen", "no afecta", "no era tan", "no era para tanto",
                                   "solo seguía", "solo seguia", "era necesario",
                                   "no le hacía daño", "no le hacia daño", "tampoco es para tanto"]):
            return "¿Qué te haría decidir que era la opción correcta en ese momento?"

    return None

# ── Phase-aware conversation structure ───────────────────────────────────────
#
# Each turn is a dict with:
#   text            — the question AURA asks
#   phase           — practice / baseline / identity_context / sensitive_dilemma / closing
#   question_type   — warm_up / environment / cognitive / consent / identity /
#                     behavioral / dilemma / closing
#   counts_for_score— whether this turn's response contributes to scoring
#   expected_signal — what behavioral signal we're probing (or None)

PHASE_TURNS = [
    # ── PHASE 1: Modo práctica (30-45s, sin score) ───────────────────────────
    # Objetivo: bajar ansiedad, probar micrófono y cámara, estabilizar al usuario.
    # Tono: muy relajado, amigable, nada formal.
    {
        "text": "Hola, soy Aura. Antes de empezar, quiero asegurarme de que el micrófono y la cámara funcionen bien. Dime algo breve cuando estés listo o lista.",
        "phase": "practice",
        "question_type": "warm_up",
        "counts_for_score": False,
        "expected_signal": None,
    },
    {
        "text": "Gracias, te escucho bien. Antes de continuar, ¿cómo te sientes en este momento?",
        "phase": "practice",
        "question_type": "warm_up",
        "counts_for_score": False,
        "expected_signal": None,
    },

    # ── PHASE 2: Calibración neutral (60s, solo baseline) ────────────────────
    # Objetivo: medir voz, ritmo, pausas, emoción neutral de la persona.
    # Preguntas fáciles y concretas — no hay respuesta incorrecta.
    {
        "text": "Listo, podemos comenzar. ¿Cómo te llamas?",
        "phase": "baseline",
        "question_type": "identity",
        "counts_for_score": True,
        "expected_signal": "baseline_speech_pattern",
    },
    {
        "text": "¿En qué ciudad estás en este momento?",
        "phase": "baseline",
        "question_type": "environment",
        "counts_for_score": True,
        "expected_signal": "baseline_speech_pattern",
    },
    {
        "text": "¿Qué fecha es hoy?",
        "phase": "baseline",
        "question_type": "cognitive",
        "counts_for_score": True,
        "expected_signal": "cognitive_baseline",
    },
    {
        "text": "¿Puedes contarme brevemente en qué consiste el trámite que estás haciendo?",
        "phase": "baseline",
        "question_type": "identity",
        "counts_for_score": True,
        "expected_signal": "baseline_speech_pattern",
    },
    {
        "text": "¿Qué estabas haciendo justo antes de entrar a esta sesión?",
        "phase": "baseline",
        "question_type": "behavioral",
        "counts_for_score": True,
        "expected_signal": "baseline_speech_pattern",
    },

    # ── PHASE 3: Identidad y contexto (45s, peso bajo) ───────────────────────
    {
        "text": "¿Estás en un lugar tranquilo ahora mismo, o hay alguien contigo?",
        "phase": "identity_context",
        "question_type": "consent",
        "counts_for_score": True,
        "expected_signal": "coercion_context",
    },
    {
        "text": "¿La decisión de hacer este trámite hoy fue tuya?",
        "phase": "identity_context",
        "question_type": "consent",
        "counts_for_score": True,
        "expected_signal": "coercion_pressure",
    },

    # ── PHASE 4: Dilemas éticos / triángulo del fraude (2-3 min, peso principal) ──
    {
        "text": "Quiero preguntarte algo un poco más personal. ¿Alguna vez alguien, en el trabajo o en otro contexto, te pidió hacer algo que sentiste que no estaba del todo bien?",
        "phase": "sensitive_dilemma",
        "question_type": "dilemma",
        "counts_for_score": True,
        "expected_signal": "rationalization_opportunity",
    },
    {
        "text": "¿Crees que hay situaciones donde romper una regla podría estar justificado?",
        "phase": "sensitive_dilemma",
        "question_type": "dilemma",
        "counts_for_score": True,
        "expected_signal": "rationalization_opportunity",
    },
    {
        "text": "Imagina que un sistema bancario te acredita dinero por error. ¿Qué harías?",
        "phase": "sensitive_dilemma",
        "question_type": "dilemma",
        "counts_for_score": True,
        "expected_signal": "rationalization_opportunity",
    },
    {
        "text": "¿Puedes contarme con tus propias palabras por qué estás haciendo este trámite hoy?",
        "phase": "sensitive_dilemma",
        "question_type": "behavioral",
        "counts_for_score": True,
        "expected_signal": "narrative_coherence",
    },

    # ── PHASE 5: Cierre (15s, sin score) ─────────────────────────────────────
    {
        "text": "Casi terminamos. ¿Confirmas que lo que compartiste hoy es información real y que participaste de forma voluntaria?",
        "phase": "closing",
        "question_type": "closing",
        "counts_for_score": False,
        "expected_signal": None,
    },
    {
        "text": "Gracias por tu tiempo. Con eso cerramos la sesión y tu información será procesada.",
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
        _PHASE_CONTEXT = {
            "practice": (
                "OBJETIVO: Reducir nervios iniciales. Confirmar que cámara y micrófono funcionan. "
                "Tono muy relajado, sin presión. Esta fase no cuenta para el score. "
                "SEÑALES A OBSERVAR: Confusión digital, dificultad técnica, nerviosismo extremo, "
                "necesidad de asistencia."
            ),
            "baseline": (
                "OBJETIVO: Establecer la línea base individual del usuario respondiendo preguntas "
                "simples y de baja carga emocional. No presionar. "
                "SEÑALES A OBSERVAR: Tono habitual, ritmo de habla, pausas normales, "
                "volumen, latencia antes de responder, claridad del audio."
            ),
            "identity_context": (
                "OBJETIVO: Entender el contexto del trámite y detectar posible presión externa. "
                "Tono profesional pero cálido. "
                "SEÑALES A OBSERVAR: Coherencia narrativa, presión de terceros, dependencia de alguien "
                "fuera de cámara, confusión sobre el propósito del trámite, respuestas memorizadas."
            ),
            "sensitive_dilemma": (
                "OBJETIVO: Observar cambios emocionales frente a preguntas de mayor carga ética. "
                "No acusar. Las preguntas son conversacionales, no de interrogatorio. "
                "SEÑALES A OBSERVAR: Evasión, contradicción, justificación excesiva, pausas largas, "
                "cambio emocional vs línea base, respuestas tipo 'todos lo hacen' o 'no afectaba a nadie', "
                "menciones de presión o coerción, incomodidad desproporcionada."
            ),
            "closing": (
                "OBJETIVO: Cerrar la conversación de forma tranquila. No revelar evaluación ni resultado. "
                "SEÑALES A OBSERVAR: Ansiedad al cierre, preguntas sobre resultado, mención tardía "
                "de terceros, problemas técnicos no reportados antes."
            ),
        }

        phase_desc = _PHASE_CONTEXT.get(self.current_phase, "")
        phase_ctx = f"\n\nFASE ACTUAL: {self.current_phase}.\n{phase_desc}"

        if user_text:
            followup = _needs_followup(user_text, self.current_phase)
            signal = _analyze_user_text(user_text)
            phase_ctx += (
                f"\n\nRESPUESTA ACTUAL DEL USUARIO: \"{user_text}\""
                f"\nSEÑAL DETECTADA: {signal}"
            )
            if followup:
                phase_ctx += (
                    f"\nSUGERENCIA DE SEGUIMIENTO (si aplica): «{followup}» — "
                    f"úsala solo si el seguimiento agrega valor real, no como rutina."
                )
            else:
                phase_ctx += (
                    "\nLa respuesta fue suficientemente clara — avanza al siguiente tema con una "
                    "transición natural y variada."
                )

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
        from device_manager import get_device
        from config import AURA_USE_GPU

        models_to_try = [AURA_LLM_MODEL, AURA_LLM_FALLBACK]
        for model_id in models_to_try:
            try:
                logger.info(f"[AURA] Intentando cargar LLM: {model_id}")
                from transformers import AutoTokenizer
                import torch

                device = get_device() if AURA_USE_GPU else "cpu"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32

                tokenizer = AutoTokenizer.from_pretrained(model_id)

                # Use Qwen2ForCausalLM directly — bypasses the auto_map registry
                # lookup that fails on some transformers versions.
                try:
                    from transformers import Qwen2ForCausalLM
                    _cls = Qwen2ForCausalLM
                except ImportError:
                    from transformers import AutoModelForCausalLM
                    _cls = AutoModelForCausalLM

                try:
                    model = _cls.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        device_map=device,
                        low_cpu_mem_usage=True,
                    )
                except RuntimeError as oom:
                    if "out of memory" in str(oom).lower():
                        logger.warning(f"[AURA] CUDA OOM al cargar {model_id} — fallback a CPU.")
                        torch.cuda.empty_cache()
                        model = _cls.from_pretrained(
                            model_id,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                        )
                        device = "cpu"
                    else:
                        raise

                model.eval()
                logger.info(f"[AURA] LLM listo en {device}: {model_id}")

                with self._lock:
                    self.tokenizer = tokenizer
                    self.llm = model
                    self.llm_ready = True
                    self.loaded_model = model_id
                    self.load_error = None

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

            input_ids = input_ids.to(self.llm.device)

            with torch.no_grad():
                output = self.llm.generate(
                    input_ids,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=0.65,
                    top_p=0.90,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = output[0][input_ids.shape[-1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Cap at 60 words — enough for ack + specific reference + one question
            words = text.split()
            if len(words) > 60:
                text = " ".join(words[:60]) + "."

            return text if text else self.fallback_reply(session, session.turn_count, user_text)

        except Exception as e:
            logger.warning(f"[AURA] LLM generate error: {e}")
            return self.fallback_reply(session, session.turn_count, user_text)

    def fallback_reply(self, session: AuraChatSession, turn: int, user_text: str = "") -> str:
        """
        Context-aware rule-based fallback. Never raises.
        Produces ONE natural sentence (transition + question), not two.
        Uses varied transitions from the conversational guide.
        """
        if not user_text.strip():
            return FALLBACK_TURNS[0]

        phase = session.current_phase

        # Follow-up takes priority over advancing
        followup = _needs_followup(user_text, phase)
        if followup:
            ack = _contextual_ack(user_text, session.turn_index, phase)
            return f"{ack} {followup}".strip()

        # Advance to next scripted question using turn_index (properly tracked)
        idx = min(session.turn_index, len(FALLBACK_TURNS) - 1)
        question = FALLBACK_TURNS[idx]

        # Detect phase change: use topic-change transition instead of same-phase ack
        next_phase = PHASE_TURNS[idx]["phase"] if idx < len(PHASE_TURNS) else phase
        if next_phase != phase and next_phase not in ("practice",):
            # Varied topic-change transition from the guide
            transition = _ACK_TOPIC_CHANGE[session.turn_index % len(_ACK_TOPIC_CHANGE)]
            # If transition ends with period, join with space; if comma/dash, join directly
            sep = " " if transition.endswith(".") else " "
            return f"{transition} {question}".strip()

        # Same phase — use contextual ack
        ack = _contextual_ack(user_text, session.turn_index, phase)
        return f"{ack} {question}".strip()

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
