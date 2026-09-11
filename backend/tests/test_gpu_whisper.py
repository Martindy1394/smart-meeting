"""Unit tests for Whisper / mBART CUDA device resolution (no model download)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mock_torch(cuda_ok: bool):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = cuda_ok
    return mock_torch


def _mock_ct2(n: int = 0):
    mock = MagicMock()
    mock.get_cuda_device_count.return_value = n
    return mock


class WhisperDeviceTests(unittest.TestCase):
    def test_auto_uses_cuda_and_float16(self):
        from app.config import settings
        from app.services import transcription

        with (
            patch.object(settings, "whisper_device", "auto"),
            patch.object(settings, "whisper_compute_type", "auto"),
            patch.dict("sys.modules", {"torch": _mock_torch(True)}),
        ):
            self.assertEqual(transcription.resolve_whisper_device(), "cuda")
            self.assertEqual(
                transcription.resolve_whisper_compute_type("cuda"), "float16"
            )

    def test_auto_falls_back_to_cpu_int8(self):
        from app.config import settings
        from app.services import transcription

        with (
            patch.object(settings, "whisper_device", "auto"),
            patch.object(settings, "whisper_compute_type", "auto"),
            patch.dict(
                "sys.modules",
                {"torch": _mock_torch(False), "ctranslate2": _mock_ct2(0)},
            ),
        ):
            self.assertEqual(transcription.resolve_whisper_device(), "cpu")
            self.assertEqual(transcription.resolve_whisper_compute_type("cpu"), "int8")

    def test_cuda_setting_falls_back_without_gpu(self):
        from app.config import settings
        from app.services import transcription

        with (
            patch.object(settings, "whisper_device", "cuda"),
            patch.dict(
                "sys.modules",
                {"torch": _mock_torch(False), "ctranslate2": _mock_ct2(0)},
            ),
        ):
            self.assertEqual(transcription.resolve_whisper_device(), "cpu")

    def test_int8_on_cuda_uses_int8_float16(self):
        from app.config import settings
        from app.services import transcription

        with patch.object(settings, "whisper_compute_type", "int8"):
            self.assertEqual(
                transcription.resolve_whisper_compute_type("cuda"), "int8_float16"
            )
            self.assertEqual(transcription.resolve_whisper_compute_type("cpu"), "int8")


class MBartDeviceTests(unittest.TestCase):
    def test_auto_uses_cuda_when_available(self):
        from app.config import settings
        from app.services import llm

        with (
            patch.object(settings, "mbart_device", "auto"),
            patch.dict("sys.modules", {"torch": _mock_torch(True)}),
        ):
            self.assertEqual(llm.resolve_mbart_device(), "cuda")

    def test_auto_falls_back_to_cpu_without_gpu(self):
        from app.config import settings
        from app.services import llm

        with (
            patch.object(settings, "mbart_device", "auto"),
            patch.dict("sys.modules", {"torch": _mock_torch(False)}),
        ):
            self.assertEqual(llm.resolve_mbart_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
