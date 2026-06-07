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

from config import AURA_LLM_FALLBACK, AURA_LLM_MODEL, AURA_CLAUDE_API_KEY, AURA_CLAUDE_MODEL

logger = logging.getLogger(__name__)

# ── System prompt para Claude API ────────────────────────────────────────────
# Basado en la Sección 18 de la Guía conversacional para AURA (versión canónica)
# + reglas de formato de las secciones 10, 11 y 12.
SYSTEM_PROMPT_FULL = """Eres AURA, una agente conversacional para INDRA, un sistema de verificación de identidad y análisis de riesgo diseñado para instituciones financieras en procesos de onboarding digital, KYC, prevención de fraude y cumplimiento AML.

Tu objetivo es mantener una conversación breve, natural, profesional y no acusatoria con el usuario. Debes ayudar a recopilar señales útiles sobre identidad, comprensión del trámite, comportamiento emocional, coherencia narrativa y posibles señales de presión, oportunidad o racionalización.

No eres juez, analista humano, policía, terapeuta ni detector de mentiras. No debes aprobar, rechazar, acusar, diagnosticar ni decir que alguien está cometiendo fraude.

No debes sonar como si siguieras un guion. Cada fase tiene objetivos, pero las preguntas son ejemplos, no frases obligatorias. Debes adaptar tu lenguaje al usuario, responder brevemente a lo que dice y avanzar solo cuando tengas suficiente información.

Nunca digas que estás detectando emociones, mentira, fraude o riesgo. Nunca acuses al usuario. Nunca muestres juicio moral. Nunca digas que el banco sospecha del usuario. Tu función visible es acompañar la verificación digital de forma clara y segura.

FASES DE LA CONVERSACIÓN:
1. Modo práctica: probar cámara/micrófono y reducir nervios. No cuenta para score.
2. Calibración neutral: hacer preguntas fáciles para establecer una línea base.
3. Contexto del trámite: entender por qué el usuario realiza el proceso y si lo hace por decisión propia.
4. Dilemas éticos: explorar presión, oportunidad y racionalización de forma conversacional e hipotética.
5. Cierre: terminar de forma tranquila, neutral y profesional.

REGLAS DE RESPUESTA:
- Máximo 2 oraciones: primera reconoce lo que dijo el usuario (específico, no genérico), segunda hace UNA sola pregunta.
- Sé breve. Sé natural. Adapta el tono al usuario.
- Responde siempre a lo que el usuario acaba de decir antes de avanzar.
- Varía las transiciones de tema. Nunca repitas "Ahora pasaré a la siguiente pregunta."
- Ejemplos de transiciones válidas: "Va, con eso me queda más claro." / "Sigamos con algo diferente." / "Entiendo." / "Cambiando un poco el enfoque," / "Te haré una pregunta un poco más personal."
- Haz seguimiento SOLO si hay ambigüedad, contradicción, presión externa, evasión, confusión, respuesta demasiado corta o posible coerción.
- No reveles criterios internos de evaluación. No menciones que calculas emociones, fraude o riesgo.

FRASES PROHIBIDAS — nunca uses:
"Entendido." / "Perfecto." / "Gracias por compartir." / "De acuerdo." / "Procedo." / "Claro que sí."
"Estoy analizando tus emociones." / "Detecté nerviosismo." / "Eso podría ser fraude."
"Tu riesgo es alto." / "Fuiste aprobado." / "Fuiste rechazado." / "Tus respuestas son inconsistentes."
Más de una pregunta por turno. Juicios morales directos.

Cada turno incluirá instrucciones internas con FASE, OBJETIVO y SEÑAL detectada. Síguelas como guía — no como guion."""

# Prompt compacto para el LLM local (Qwen 1.5B) — versión corta que el modelo pequeño sí puede seguir
SYSTEM_PROMPT = """Eres AURA, agente conversacional de verificación. Hablas en español.

REGLAS (síguelas siempre):
1. Máximo 2 oraciones por respuesta. Nunca más.
2. Primera oración: reconoce brevemente lo que dijo el usuario — de forma específica, no genérica.
3. Segunda oración: haz UNA sola pregunta. Solo una.
4. Nunca reveles que analizas emociones, fraude o riesgo.
5. Frases prohibidas: "Entendido.", "Perfecto.", "Gracias por compartir.", "De acuerdo.", "Procedo.", "Claro que sí."
6. Si el usuario menciona a otra persona o presión externa, haz la TAREA indicada.

En cada turno recibirás instrucciones específicas sobre qué pregunta hacer. Síguelas."""

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
        # Per-turn task: tell the model exactly what to do this turn.
        # Small models (1.5B) follow explicit tasks far better than abstract guidelines.
        task_lines = [f"FASE: {self.current_phase}"]

        if user_text:
            signal = _analyze_user_text(user_text)
            followup = _needs_followup(user_text, self.current_phase)
            task_lines.append(f'USUARIO DIJO: "{user_text}"')
            task_lines.append(f"SEÑAL: {signal}")

            if followup:
                task_lines.append(
                    f"TAREA: Primero reconoce lo que dijo en una oración. "
                    f"Luego haz esta pregunta (reformulada de forma natural): «{followup}»"
                )
            else:
                next_idx = min(self.turn_index, len(PHASE_TURNS) - 1)
                next_q = PHASE_TURNS[next_idx]["text"]
                task_lines.append(
                    f"TAREA: Primero reconoce lo que dijo en una oración. "
                    f"Luego haz esta pregunta reformulada de forma natural "
                    f"(NO la copies literalmente): «{next_q}»"
                )
        else:
            # First greeting or no user text yet — just use next scripted question
            next_idx = min(self.turn_index, len(PHASE_TURNS) - 1)
            next_q = PHASE_TURNS[next_idx]["text"]
            task_lines.append(f"TAREA: Di: «{next_q}»")

        task_ctx = "\n\n" + "\n".join(task_lines)

        history = [{"role": "system", "content": SYSTEM_PROMPT + task_ctx}]
        # Last 6 messages — small models handle short context better
        for msg in self.messages[-6:]:
            role = "assistant" if msg["role"] == "aura" else "user"
            history.append({"role": role, "content": msg["content"]})
        return history

    def build_claude_system(self, user_text: str = "") -> str:
        """System prompt for Claude API: full guide + per-turn objective (NOT a scripted question)."""
        # Per-turn meta from PHASE_TURNS
        idx = min(self.turn_index, len(PHASE_TURNS) - 1)
        turn_meta = PHASE_TURNS[idx]

        # Map question_type → objective description
        _TURN_OBJECTIVE = {
            "warm_up": "Saluda, preséntate brevemente y pide algo corto para probar el micrófono. Crea un ambiente cómodo y sin presión.",
            "environment": "Establece el entorno del usuario con una pregunta simple sobre ubicación o contexto.",
            "cognitive": "Verifica orientación temporal básica con una pregunta simple.",
            "identity": "Obtén información básica de identidad o pide que describa el trámite que está realizando.",
            "behavioral": "Explora el contexto conductual: qué estaba haciendo, cómo llegó aquí, cuál es su situación.",
            "consent": "Verifica autonomía: si hay terceros presentes y si la participación es por decisión propia.",
            "dilemma": "Explora presión, oportunidad o racionalización con una pregunta hipotética, conversacional y no acusatoria.",
            "closing": "Cierra la conversación con calma. Confirma participación voluntaria y da un cierre neutral.",
        }

        # Map expected_signal → what to observe
        _SIGNAL_OBJ = {
            "baseline_speech_pattern": "Establece patrón base de habla y respuesta.",
            "cognitive_baseline": "Verifica orientación y claridad básica.",
            "coercion_context": "Observa si hay terceros o señales de contexto de presión.",
            "coercion_pressure": "Verifica que la participación sea voluntaria y autónoma.",
            "rationalization_opportunity": "Observa racionalización, minimización del daño o justificación de conductas riesgosas.",
            "narrative_coherence": "Evalúa coherencia del relato sobre el trámite.",
        }

        objective = _TURN_OBJECTIVE.get(turn_meta["question_type"], "Avanza la conversación hacia el objetivo de la fase.")
        signal_obj = _SIGNAL_OBJ.get(turn_meta["expected_signal"] or "", "")
        example_q = turn_meta["text"]

        task_lines = [
            f"FASE: {self.current_phase}",
            f"OBJETIVO DEL TURNO: {objective}",
        ]
        if signal_obj:
            task_lines.append(f"SEÑAL A OBSERVAR: {signal_obj}")

        task_lines.append(f"PREGUNTA DE EJEMPLO (orienta el tema, NO la copies textualmente): «{example_q}»")

        if user_text:
            signal = _analyze_user_text(user_text)
            followup = _needs_followup(user_text, self.current_phase)
            task_lines.append(f'RESPUESTA DEL USUARIO: "{user_text[:300]}"')
            task_lines.append(f"SEÑAL DETECTADA: {signal}")

            if followup:
                task_lines.append(
                    f"ACCIÓN: Hay una señal que requiere seguimiento. "
                    f"Reconoce en una oración lo que dijo y haz una pregunta de seguimiento natural. "
                    f"Sugerencia (no obligatoria): «{followup}»"
                )
            else:
                task_lines.append(
                    "ACCIÓN: La respuesta fue suficiente. Reconoce brevemente lo que dijo "
                    "y avanza al objetivo de este turno. Formula tu propia pregunta natural — "
                    "no copies el ejemplo."
                )
        else:
            task_lines.append(
                "ACCIÓN: Inicia o continúa la conversación hacia el objetivo de este turno. "
                "Formula tu propia pregunta natural — el ejemplo es solo orientación."
            )

        return SYSTEM_PROMPT_FULL + "\n\n---\nINSTRUCCIONES PARA ESTE TURNO:\n" + "\n".join(task_lines)

    def build_claude_messages(self) -> list:
        """Message list for Claude API (no system role, alternating user/assistant)."""
        msgs = []
        for m in self.messages[-10:]:
            role = "assistant" if m["role"] == "aura" else "user"
            msgs.append({"role": role, "content": m["content"]})
        # Claude API requires first message to be user
        while msgs and msgs[0]["role"] == "assistant":
            msgs.pop(0)
        return msgs or [{"role": "user", "content": "Hola"}]


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
            "claude_api_enabled": bool(AURA_CLAUDE_API_KEY),
            "claude_model": AURA_CLAUDE_MODEL if AURA_CLAUDE_API_KEY else None,
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

    def _generate_claude(self, session: AuraChatSession, user_text: str) -> str:
        """Generate response via Claude API. Runs in thread executor — sync client is fine."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=AURA_CLAUDE_API_KEY)

            system = session.build_claude_system(user_text)
            messages = session.build_claude_messages()

            response = client.messages.create(
                model=AURA_CLAUDE_MODEL,
                max_tokens=150,
                system=system,
                messages=messages,
            )
            text = response.content[0].text.strip()

            # Enforce word cap (Claude is concise but just in case)
            words = text.split()
            if len(words) > 70:
                text = " ".join(words[:70]) + "."

            logger.debug(f"[AURA/Claude] {text[:80]}")
            return text if text else self.fallback_reply(session, session.turn_count, user_text)

        except Exception as e:
            logger.warning(f"[AURA] Claude API error: {e} — fallback a LLM local")
            return ""  # Signal caller to try local LLM next

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
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                output = self.llm.generate(
                    input_ids,
                    attention_mask=attention_mask,
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
        Main entry point. Priority: Claude API → local LLM → rule-based fallback.
        Never raises — always returns a string.
        """
        # 1. Claude API (best quality, requires AURA_CLAUDE_API_KEY)
        if AURA_CLAUDE_API_KEY:
            reply = self._generate_claude(session, user_text)
            if reply:
                return reply
            # Empty string means API failed — fall through to local LLM

        # 2. Local LLM (Qwen2.5)
        with self._lock:
            llm_ready = self.llm_ready
        if llm_ready:
            return self._generate_llm(session, user_text)

        # 3. Rule-based fallback
        session.used_fallback = True
        return self.fallback_reply(session, session.turn_count, user_text)


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
