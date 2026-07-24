from __future__ import annotations

import io
import wave
from typing import Tuple

import numpy as np


class PCMConverter:
    @staticmethod
    def wav_to_pcm(wav_bytes: bytes, target_rate: int = 24000) -> Tuple[bytes, int]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if sampwidth == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) << 8
        elif sampwidth == 2:
            audio = np.frombuffer(frames, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported WAV sampwidth={sampwidth}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

        if rate != target_rate:
            audio = PCMConverter._resample(audio, rate, target_rate)
            rate = target_rate

        return audio.tobytes(), rate

    @staticmethod
    def pcm_to_wav(
        pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1
    ) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    @staticmethod
    def duration_ms(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> float:
        if not pcm_bytes or sample_rate <= 0:
            return 0.0
        samples = len(pcm_bytes) // (2 * channels)
        return (samples / sample_rate) * 1000.0

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or len(audio) == 0:
            return audio
        duration = len(audio) / float(src_rate)
        dst_len = max(1, int(round(duration * dst_rate)))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        resampled = np.interp(x_new, x_old, audio.astype(np.float32))
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    @staticmethod
    def chunk_pcm(pcm_bytes: bytes, chunk_ms: int, sample_rate: int) -> list[bytes]:
        bytes_per_ms = max((sample_rate * 2) // 1000, 1)
        chunk_size = max(bytes_per_ms * chunk_ms, 2)
        if chunk_size % 2:
            chunk_size += 1
        return [
            pcm_bytes[i : i + chunk_size] for i in range(0, len(pcm_bytes), chunk_size)
        ]

    @staticmethod
    def ensure_wav(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """If bytes look like WAV keep them; otherwise wrap raw PCM16."""
        if audio_bytes[:4] == b"RIFF":
            return audio_bytes
        return PCMConverter.pcm_to_wav(audio_bytes, sample_rate=sample_rate)
