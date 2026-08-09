"""Auth refresh + revocation, faithfulness, crypto-at-rest, MT protocol."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class AuthRefreshTests(unittest.TestCase):
    def test_issue_refresh_and_revoke(self):
        from app.security import (
            decode_access_token,
            decode_refresh_token,
            issue_token_pair,
            revoke_token,
        )

        pair = issue_token_pair("user-1", extra={"username": "demo"})
        self.assertTrue(pair["access_token"])
        self.assertTrue(pair["refresh_token"])
        access = decode_access_token(pair["access_token"])
        refresh = decode_refresh_token(pair["refresh_token"])
        self.assertEqual(access["sub"], "user-1")
        self.assertEqual(access["type"], "access")
        self.assertEqual(refresh["type"], "refresh")
        revoke_token(pair["access_token"])
        self.assertIsNone(decode_access_token(pair["access_token"]))
        # Refresh still valid until revoked.
        self.assertIsNotNone(decode_refresh_token(pair["refresh_token"]))
        revoke_token(pair["refresh_token"])
        self.assertIsNone(decode_refresh_token(pair["refresh_token"]))


class FaithfulnessTests(unittest.TestCase):
    def test_flags_untraced_action_line(self):
        from app.services import llm

        source = (
            "The committee discussed the quarterly budget. "
            "They approved the travel policy revision."
        )
        summary = (
            "Discussion\n"
            "• The committee discussed the quarterly budget.\n\n"
            "Decisions\n"
            "• They approved the travel policy revision.\n\n"
            "Action items\n"
            "• Launch a satellite office on Mars next week."
        )
        report = llm.assess_minutes_faithfulness(summary, source)
        self.assertEqual(report["status"], "warn")
        self.assertGreaterEqual(report["checked"], 2)
        self.assertTrue(any("Mars" in u["line"] for u in report["untraced"]))

    def test_ok_when_grounded(self):
        from app.services import llm

        source = "Board approved the budget and assigned Maria the follow-up."
        summary = (
            "Decisions\n"
            "• Board approved the budget.\n\n"
            "Action items\n"
            "• Assigned Maria the follow-up."
        )
        report = llm.assess_minutes_faithfulness(summary, source)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["untraced"], [])


class CryptoAtRestTests(unittest.TestCase):
    def test_roundtrip_with_key(self):
        from cryptography.fernet import Fernet

        from app.config import settings
        from app.services import crypto_at_rest

        key = Fernet.generate_key().decode()
        original = settings.data_encryption_key
        try:
            settings.data_encryption_key = key
            self.assertTrue(crypto_at_rest.encryption_enabled())
            plain = b"RIFF....WAVEtestdata"
            enc = crypto_at_rest.encrypt_bytes(plain)
            self.assertTrue(enc.startswith(b"SMENC1\n"))
            self.assertNotEqual(enc, plain)
            self.assertEqual(crypto_at_rest.decrypt_bytes(enc), plain)
            # Legacy plaintext passthrough.
            self.assertEqual(crypto_at_rest.decrypt_bytes(plain), plain)
        finally:
            settings.data_encryption_key = original

    def test_wrong_key_raises_decryption_error(self):
        from cryptography.fernet import Fernet

        from app.config import settings
        from app.services import crypto_at_rest

        key_a = Fernet.generate_key().decode()
        key_b = Fernet.generate_key().decode()
        original = settings.data_encryption_key
        try:
            settings.data_encryption_key = key_a
            enc = crypto_at_rest.encrypt_bytes(b"RIFF....WAVE")
            settings.data_encryption_key = key_b
            with self.assertRaises(crypto_at_rest.DecryptionError) as ctx:
                crypto_at_rest.decrypt_bytes(enc)
            self.assertIn("Decryption failed / Data corrupted", str(ctx.exception))
        finally:
            settings.data_encryption_key = original

    def test_missing_key_raises_on_ciphertext(self):
        from cryptography.fernet import Fernet

        from app.config import settings
        from app.services import crypto_at_rest

        key = Fernet.generate_key().decode()
        original = settings.data_encryption_key
        try:
            settings.data_encryption_key = key
            enc = crypto_at_rest.encrypt_bytes(b"secret-audio")
            settings.data_encryption_key = ""
            with self.assertRaises(crypto_at_rest.DecryptionError) as ctx:
                crypto_at_rest.decrypt_bytes(enc)
            self.assertIn("DATA_ENCRYPTION_KEY", str(ctx.exception))
        finally:
            settings.data_encryption_key = original

    def test_redis_get_wav_raises_on_decrypt_failure(self):
        from cryptography.fernet import Fernet

        from app.config import settings
        from app.services import crypto_at_rest, redis_store

        class _FakeClient:
            def __init__(self, payload: bytes):
                self._payload = payload

            def get(self, _key):
                return self._payload

        key = Fernet.generate_key().decode()
        original = settings.data_encryption_key
        try:
            settings.data_encryption_key = key
            enc = crypto_at_rest.encrypt_bytes(b"RIFFWAVEblob")
            settings.data_encryption_key = Fernet.generate_key().decode()
            with patch.object(redis_store, "get_client", return_value=_FakeClient(enc)):
                with self.assertRaises(crypto_at_rest.DecryptionError):
                    redis_store.get_wav_bytes("meeting-1")
        finally:
            settings.data_encryption_key = original

    def test_redis_get_wav_empty_on_miss(self):
        from app.services import redis_store

        class _FakeClient:
            def get(self, _key):
                return None

        with patch.object(redis_store, "get_client", return_value=_FakeClient()):
            self.assertEqual(redis_store.get_wav_bytes("missing"), b"")

        with patch.object(redis_store, "get_client", return_value=None):
            self.assertEqual(redis_store.get_wav_bytes("no-redis"), b"")


class HiligaynonForcingTests(unittest.TestCase):
    def test_pipeline_reports_never_tl(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with (
            patch("app.main.asr.is_available", return_value=True),
            patch("app.main.redis_store.is_available", return_value=False),
            patch("app.main.audio.ffmpeg_available", return_value=False),
            patch("app.main.llm.summarizer_available", return_value=True),
        ):
            client = TestClient(app)
            res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        pipe = res.json()["pipeline"]["whisper"]
        self.assertIsNone(pipe["hiligaynon_forced_language"])
        self.assertIn("never tl", pipe["hiligaynon_decode"])


class MtBenchmarkProtocolTests(unittest.TestCase):
    def test_fixtures_only_protocol(self):
        import tempfile

        script = ROOT / "scripts/ph_mt/benchmark_mbart_tags.py"
        self.assertTrue(script.is_file())
        # Import as module via runpy path
        import importlib.util

        spec = importlib.util.spec_from_file_location("benchmark_mbart_tags", script)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            code = mod.main(
                [
                    "--fixtures-only",
                    "--out-dir",
                    tmp,
                ]
            )
            self.assertEqual(code, 0)
            report = Path(tmp) / "report.json"
            self.assertTrue(report.is_file())
            sheet = Path(tmp) / "hiligaynon_id_ID_human_eval.md"
            self.assertTrue(sheet.is_file())


if __name__ == "__main__":
    unittest.main()
