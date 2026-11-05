"""Tests for the faster-whisper engine and its whisper.cpp fallback."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from openocto.config import STTConfig
from openocto.stt.base import TranscriptionResult
from openocto.stt.fallback import FallbackSTTEngine
from openocto.stt.faster_whisper import FasterWhisperEngine


class _FakeWhisperModel:
    calls: list[tuple[str, dict]] = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.transcribe_kwargs = None
        self.audio = None
        _FakeWhisperModel.calls.append((model_name, kwargs))

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        self.transcribe_kwargs = kwargs
        return iter([NS(text=" hola "), NS(text="mundo")]), NS(
            language="es", language_probability=0.91
        )


def _cfg(download_root: str, device: str = "auto") -> STTConfig:
    return STTConfig(
        engine="faster_whisper",
        fallback_engine="whisper_cpp",
        language="es",
        n_threads=2,
        hotwords=["Bogota", "Samantha"],
        faster_whisper={
            "model": "large-v3-turbo",
            "device": device,
            "compute_type": "default",
            "beam_size": 5,
            "download_root": download_root,
        },
    )


class FasterWhisperEngineTest(unittest.TestCase):
    def setUp(self):
        _FakeWhisperModel.calls = []
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_transcribe_uses_large_model_and_prompt(self):
        cfg = _cfg(self.tmp.name)
        with patch("faster_whisper.WhisperModel", _FakeWhisperModel):
            engine = FasterWhisperEngine(cfg)

        audio = np.array([0, 1000, -1000, 0], dtype=np.int16)
        result = engine.transcribe(audio)

        self.assertEqual(result.text, "hola mundo")
        self.assertEqual(result.language, "es")
        self.assertAlmostEqual(result.confidence, 0.91)
        self.assertGreaterEqual(result.duration_ms, 0)
        model = engine._model
        self.assertEqual(model.model_name, "large-v3-turbo")
        self.assertEqual(model.transcribe_kwargs["language"], "es")
        self.assertEqual(model.transcribe_kwargs["beam_size"], 5)
        self.assertFalse(model.transcribe_kwargs["vad_filter"])
        self.assertFalse(model.transcribe_kwargs["condition_on_previous_text"])
        self.assertIn("Bogota", model.transcribe_kwargs["initial_prompt"])
        self.assertIn("Samantha", model.transcribe_kwargs["initial_prompt"])
        self.assertEqual(model.audio.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(model.audio))), 1.0)

    def test_cuda_failure_retries_cpu_int8(self):
        class _CudaFailThenOk(_FakeWhisperModel):
            def __init__(self, model_name, **kwargs):
                super().__init__(model_name, **kwargs)
                if len(self.calls) == 1:
                    raise RuntimeError("cuDNN missing")

        with patch("faster_whisper.WhisperModel", _CudaFailThenOk):
            engine = FasterWhisperEngine(_cfg(self.tmp.name, device="auto"))

        self.assertEqual(len(_CudaFailThenOk.calls), 2)
        self.assertEqual(_CudaFailThenOk.calls[1][1]["device"], "cpu")
        self.assertEqual(_CudaFailThenOk.calls[1][1]["compute_type"], "int8")
        self.assertEqual(engine._device, "cpu")

    def test_empty_audio_does_not_call_the_model(self):
        with patch("faster_whisper.WhisperModel", _FakeWhisperModel):
            engine = FasterWhisperEngine(_cfg(self.tmp.name))
        result = engine.transcribe(np.zeros(0, dtype=np.int16))
        self.assertEqual(result.text, "")
        self.assertEqual(_FakeWhisperModel.calls[0][0], "large-v3-turbo")

    def test_describe_mentions_model_and_cpu(self):
        class _Model:
            model = NS(device="cpu", compute_type="int8")

        with patch("faster_whisper.WhisperModel", _FakeWhisperModel):
            engine = FasterWhisperEngine(_cfg(self.tmp.name))
        engine._device = "cpu"
        engine._compute_type = "int8"
        self.assertEqual(engine.describe(), "faster-whisper large-v3-turbo (CPU int8)")

    def test_describe_mentions_gpu(self):
        with patch("faster_whisper.WhisperModel", _FakeWhisperModel):
            engine = FasterWhisperEngine(_cfg(self.tmp.name))
        engine._device = "cuda:0"
        engine._compute_type = "float16"
        self.assertEqual(
            engine.describe(), "faster-whisper large-v3-turbo (GPU float16)"
        )

    def test_cuda_dll_dirs_registered_on_windows(self):
        import os as _os
        from openocto.stt import faster_whisper as fw_mod

        if _os.name != "nt":
            self.skipTest("solo Windows")
        cfg = _cfg(self.tmp.name)
        cfg.faster_whisper.cuda_dll_dirs = [self.tmp.name]
        before = len(fw_mod._DLL_HANDLES)
        fw_mod._register_cuda_dll_dirs(cfg.faster_whisper)
        self.assertGreater(len(fw_mod._DLL_HANDLES), before)


class _PrimaryFail:
    def transcribe(self, audio, sample_rate=16000):
        raise RuntimeError("primary down")


class _PrimaryEmpty:
    def transcribe(self, audio, sample_rate=16000):
        return TranscriptionResult("", "es", 0.0, 1.0)


class _FallbackOk:
    def transcribe(self, audio, sample_rate=16000):
        return TranscriptionResult("texto base", "es", 0.0, 2.0)


class _Timed:
    def __init__(self, text: str, duration_ms: float) -> None:
        self.text = text
        self.duration_ms = duration_ms
        self.calls = 0

    def transcribe(self, audio, sample_rate=16000):
        self.calls += 1
        return TranscriptionResult(self.text, "es", 0.0, self.duration_ms)


class FallbackSTTEngineTest(unittest.TestCase):
    def test_primary_error_uses_fallback(self):
        wrapper = FallbackSTTEngine(_PrimaryFail(), _FallbackOk())
        with patch("openocto.stt.fallback.logger.exception"):
            result = wrapper.transcribe(np.zeros(10, dtype=np.int16))
        self.assertEqual(result.text, "texto base")

    def test_primary_empty_uses_fallback(self):
        wrapper = FallbackSTTEngine(_PrimaryEmpty(), _FallbackOk())
        result = wrapper.transcribe(np.zeros(10, dtype=np.int16))
        self.assertEqual(result.text, "texto base")

    def test_slow_primary_switches_for_next_turns(self):
        primary = _Timed("grande", 5000)
        fallback = _Timed("base", 100)
        wrapper = FallbackSTTEngine(primary, fallback, max_latency_ms=4000)

        first = wrapper.transcribe(np.zeros(10, dtype=np.int16))
        second = wrapper.transcribe(np.zeros(10, dtype=np.int16))

        self.assertEqual(first.text, "grande")
        self.assertEqual(second.text, "base")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)

    def test_fast_primary_stays(self):
        primary = _Timed("grande", 1000)
        fallback = _Timed("base", 100)
        wrapper = FallbackSTTEngine(primary, fallback, max_latency_ms=4000)

        first = wrapper.transcribe(np.zeros(10, dtype=np.int16))
        second = wrapper.transcribe(np.zeros(10, dtype=np.int16))

        self.assertEqual(first.text, "grande")
        self.assertEqual(second.text, "grande")
        self.assertEqual(primary.calls, 2)
        self.assertEqual(fallback.calls, 0)

    def test_threshold_zero_disables_latency_switch(self):
        primary = _Timed("grande", 5000)
        fallback = _Timed("base", 100)
        wrapper = FallbackSTTEngine(primary, fallback, max_latency_ms=0)

        wrapper.transcribe(np.zeros(10, dtype=np.int16))
        wrapper.transcribe(np.zeros(10, dtype=np.int16))

        self.assertEqual(primary.calls, 2)
        self.assertEqual(fallback.calls, 0)

    def test_describe_shows_chain_and_latency(self):
        class _D:
            def __init__(self, name):
                self._name = name

            def describe(self):
                return self._name

            def transcribe(self, audio, sample_rate=16000):
                return TranscriptionResult("x", "es", 0.0, 9000.0)

        wrapper = FallbackSTTEngine(_D("grande"), _D("base"), max_latency_ms=4000)
        self.assertEqual(wrapper.describe(), "grande -> base")
        wrapper.transcribe(np.zeros(10, dtype=np.int16))
        wrapper.transcribe(np.zeros(10, dtype=np.int16))
        self.assertIn("latencia", wrapper.describe())


class STTFactoryTest(unittest.TestCase):
    def test_primary_failure_falls_back_to_whisper_cpp(self):
        from openocto.stt.factory import create_stt_engine

        fake = object()
        with (
            patch(
                "openocto.stt.faster_whisper.FasterWhisperEngine",
                side_effect=RuntimeError("model missing"),
            ),
            patch("openocto.stt.factory.logger.exception"),
            patch("openocto.stt.factory.WhisperCppEngine", return_value=fake),
        ):
            self.assertIs(create_stt_engine(_cfg(tempfile.gettempdir())), fake)


if __name__ == "__main__":
    unittest.main()
