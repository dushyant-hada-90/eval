"""Backward-compatible shims — prefer `stt`, `tts`, `audio`, `scoring` packages. """

from audio.pcm_converter import PCMConverter
from scoring.groq import GroqScorer
from stt.groq import GroqSTTAdapter as GroqSTT
from tts.groq import GroqTTSAdapter as GroqTTS

__all__ = ["GroqSTT", "GroqTTS", "GroqScorer", "PCMConverter"]
