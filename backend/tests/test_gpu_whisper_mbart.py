"""Unit tests for Whisper / mBART CUDA device resolution (no model download)."""
from unittest.mock import MagicMock, patch


def _mock_torch(cuda_ok: bool):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = cuda_ok
    return mock_torch


def _mock_ct2(n: int = 0):
    mock = MagicMock()
    mock.get_cuda_device_count.return_value = n
    return mock


def test_whisper_auto_uses_cuda_when_available():
    from app.config import settings
    from app.services import transcription

    with (
        patch.object(settings, "whisper_device", "auto"),
        patch.object(settings, "whisper_compute_type", "auto"),
        patch.dict("sys.modules", {"torch": _mock_torch(True)}),
    ):
        assert transcription.resolve_whisper_device() == "cuda"
        assert transcription.resolve_whisper_compute_type("cuda") == "float16"


def test_whisper_auto_falls_back_to_cpu_without_gpu():
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
        assert transcription.resolve_whisper_device() == "cpu"
        assert transcription.resolve_whisper_compute_type("cpu") == "int8"


def test_whisper_cuda_setting_falls_back_without_gpu():
    from app.config import settings
    from app.services import transcription

    with (
        patch.object(settings, "whisper_device", "cuda"),
        patch.dict(
            "sys.modules",
            {"torch": _mock_torch(False), "ctranslate2": _mock_ct2(0)},
        ),
    ):
        assert transcription.resolve_whisper_device() == "cpu"


def test_mbart_auto_uses_cuda_when_available():
    from app.config import settings
    from app.services import llm

    with (
        patch.object(settings, "mbart_device", "auto"),
        patch.dict("sys.modules", {"torch": _mock_torch(True)}),
    ):
        assert llm.resolve_mbart_device() == "cuda"


def test_mbart_auto_falls_back_to_cpu_without_gpu():
    from app.config import settings
    from app.services import llm

    with (
        patch.object(settings, "mbart_device", "auto"),
        patch.dict("sys.modules", {"torch": _mock_torch(False)}),
    ):
        assert llm.resolve_mbart_device() == "cpu"
