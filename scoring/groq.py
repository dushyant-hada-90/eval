from __future__ import annotations

import json
import re
from typing import Any, Optional

from groq import AsyncGroq

from utils.config import settings
from utils.logging import get_logger

logger = get_logger(__name__)

_HUMAN_SYSTEM = """You are dual-role in a live phone-call evaluation.

ROLE A — HUMAN CALLER (primary for next_user_utterance)
You are a real person on a phone call, not a bot, not an interviewer, not a QA script reader.
Stay in character for the given persona.

Speak like a human on a voice call:
- Short turns (1–2 spoken sentences, ~8–25 words). This will be read aloud by TTS.
- React to what the agent JUST said (answer their question, push back, ask a follow-up, or pivot).
- Use natural spoken English: contractions, hesitations sparingly ("yeah", "hmm", "look"), incomplete thoughts OK.
- Do NOT sound corporate, scripted, or like a test case.
- Do NOT mention scoring, evaluation, prompts, JSON, or that you are an AI.
- Do NOT repeat the agent's words back verbatim.
- Do NOT dump multiple questions at once.
- Keep one clear intent per turn.
- If the agent asks something, answer it before changing topic (unless persona is rushed/aggressive).

TTS MARKUP (our cloned-voice TTS understands these — use them in next_user_utterance):
1) Non-verbal emotion / sound tags — insert the tag inline in the spoken text (exact spelling, square brackets):
   [laughter]  [sigh]  [sniff]  [confirmation-en]
   [question-en]  [question-ah]  [question-oh]  [question-ei]  [question-yi]
   [surprise-ah]  [surprise-oh]  [surprise-wa]  [surprise-yo]
   [dissatisfaction-hnn]
   Examples:
   - "Yeah that makes sense [confirmation-en], but what about support?"
   - "[laughter] Okay fair enough — so what's the monthly cost?"
   - "[sigh] Look, I'm not sure this is for us."
   Use 0–2 tags per turn when emotion fits; never invent new tags; never wrap spoken words inside the brackets.
2) Emphasis — enclose a word or short phrase in single asterisks for slight stress: *word* or *this week*.
   Use sparingly (at most one *…* per turn). Do not use **double** asterisks or markdown bold.
3) Punctuation — take care with commas, periods, question marks, and ellipses so TTS pacing sounds natural.
   Prefer: "Wait — *how much* did you say?" over run-on lines with no punctuation.

NO SCRIPTED LINES:
- You invent what to say. Follow the PERSONA (goals, emotional arc, hard questions, rules).
- Decide the next line from the live conversation — do not wait for or expect canned utterances.
- Do not dump every goal at once; advance naturally based on how the agent responds.

Ending the call:
- When the persona's goals are met, or the conversation has run long enough per persona rules,
  wrap up naturally (thanks / ask for SMS update / goodbye).
- Set conversation_should_end=true only when this utterance is meant to close the call.

ROLE B — EVALUATOR (for score fields only)
Using the scoring rubric below, rate the agent's latest reply. Keep reasoning brief (one sentence).

SCORING RUBRIC:
{scoring_rubric}

PERSONA:
{persona}
"""


class GroqScorer:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.client = AsyncGroq(api_key=api_key or settings.groq_api_key)
        self.model = model or settings.groq_llm_model

    async def score_turn(
        self,
        testing_prompt: str,
        test_prompt: str,
        user_intent: str,
        agent_transcript: str,
        conversation_history: list[dict[str, str]],
    ) -> dict[str, Any]:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in conversation_history
        )
        user_msg = f"""Conversation so far:
{history_text or '(empty)'}

Latest user test utterance (intent={user_intent}):
{test_prompt}

Latest agent transcript:
{agent_transcript}

Score the agent's latest response. Return JSON only with keys:
intent_alignment, questions_asked, tone, context_retention (integers 1-5), reasoning (string).
"""
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": testing_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content or ""
        return self.parse_scores(content)

    async def checkpoint_next_turn(
        self,
        testing_prompt: str,
        agent_transcript: str,
        conversation_history: list[dict[str, str]],
        persona: str | None = None,
        turn_number: int = 0,
        max_exchanges: int = 8,
        next_script_hint: str | None = None,  # ignored — kept for call-site compat
        next_intent: str | None = None,  # ignored
        script_remaining: int = 0,  # ignored
    ) -> dict[str, Any]:
        """Score agent reply and invent the next human caller utterance for TTS."""
        del next_script_hint, next_intent, script_remaining  # never steer speech
        history_text = _format_history(conversation_history)
        persona_text = (persona or "Curious prospect on a phone call").strip()
        max_ex = max(4, int(max_exchanges or 8))

        if turn_number >= max_ex:
            progress = (
                f"Caller turn #{turn_number} (soft max ~{max_ex}). "
                "Prefer wrapping up naturally now unless a critical goal is still open. "
                "If you close, set conversation_should_end=true."
            )
        else:
            progress = (
                f"Caller turn #{turn_number} of ~{max_ex}. "
                "Invent the next natural spoken line yourself from the persona + what the agent just said. "
                "No canned script will be provided — use your judgment."
            )

        system = _HUMAN_SYSTEM.format(
            scoring_rubric=testing_prompt.strip(),
            persona=persona_text,
        )
        user_msg = f"""Call so far:
{history_text or '(just started — agent greeted first)'}

Agent just said:
\"\"\"{agent_transcript}\"\"\"

{progress}

Respond as the human caller continuing this phone call.
Return JSON only (no markdown):
{{
  "intent_alignment": <1-5>,
  "questions_asked": <1-5>,
  "tone": <1-5>,
  "context_retention": <1-5>,
  "reasoning": "<one short sentence>",
  "next_user_utterance": "<spoken line you invent for TTS; may include [tags] and *emphasis*>",
  "conversation_should_end": <true|false>
}}
"""
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.85,
        )
        content = completion.choices[0].message.content or ""
        parsed = self.parse_scores(content)
        data = _extract_json_object(content) or {}

        utterance = str(
            data.get("next_user_utterance")
            or "Okay… thanks for checking. I'll wait for the update."
        ).strip()
        utterance = _humanize_cleanup(utterance)

        should_end = bool(data.get("conversation_should_end"))
        if turn_number >= max_ex + 2:
            should_end = True

        parsed["next_user_utterance"] = utterance
        parsed["conversation_should_end"] = should_end
        return parsed

    @staticmethod
    def parse_scores(raw: str) -> dict[str, Any]:
        data = _extract_json_object(raw)
        if not data:
            logger.warning("Failed to parse scores JSON: %s", raw[:300])
            return {
                "intent_alignment": None,
                "questions_asked": None,
                "tone": None,
                "context_retention": None,
                "reasoning": raw[:500],
                "avg_score": None,
            }

        intent = _clamp(
            data.get("intent_alignment") or data.get("intent_alignment_score")
        )
        questions = _clamp(
            data.get("questions_asked")
            or data.get("question_asking")
            or data.get("question_asking_score")
        )
        tone = _clamp(data.get("tone") or data.get("tone_score"))
        retention = _clamp(
            data.get("context_retention") or data.get("context_retention_score")
        )
        scores = [s for s in (intent, questions, tone, retention) if s is not None]
        avg = sum(scores) / len(scores) if scores else None
        return {
            "intent_alignment": intent,
            "questions_asked": questions,
            "tone": tone,
            "context_retention": retention,
            "reasoning": str(data.get("reasoning", "")),
            "avg_score": avg,
        }


def _format_history(history: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for m in history:
        role = m.get("role", "")
        label = "YOU (caller)" if role == "user" else "AGENT"
        lines.append(f"{label}: {m.get('content', '')}")
    return "\n".join(lines)


def _extract_json_object(raw: str) -> Optional[dict[str, Any]]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


_ALLOWED_TTS_TAGS = {
    "laughter",
    "sigh",
    "sniff",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
}


def _humanize_cleanup(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    # Drop stage directions / labels models sometimes add
    text = re.sub(
        r"^(caller|user|prospect|me)\s*:\s*", "", text, flags=re.IGNORECASE
    )
    # Normalize whitespace but keep spaces around TTS tags / emphasis
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text).strip()
    # Drop unknown [bracket] tags; keep supported non-verbal tags
    def _tag_filter(match: re.Match[str]) -> str:
        inner = match.group(1).strip().lower()
        if inner in _ALLOWED_TTS_TAGS:
            return f"[{inner}]"
        return ""

    text = re.sub(r"\[([^\]]+)\]", _tag_filter, text)
    # Collapse **bold** markdown into *emphasis*
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Keep TTS-friendly length (tags count toward limit)
    if len(text) > 280:
        text = text[:277].rsplit(" ", 1)[0] + "…"
    return text


def _clamp(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(5, n))
