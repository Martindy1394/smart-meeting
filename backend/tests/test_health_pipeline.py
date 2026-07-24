"""Health endpoint exposes the three-model pipeline map."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class HealthPipelineTests(unittest.TestCase):
    def test_health_includes_pipeline_stages(self):
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
        data = res.json()
        self.assertIn("pipeline", data)
        pipe = data["pipeline"]
        self.assertEqual(pipe["whisper"]["role"], "asr")
        self.assertEqual(pipe["mbart_nllb"]["role"], "translation")
        self.assertEqual(pipe["bart"]["role"], "summarization")
        self.assertTrue(pipe["whisper"]["available"])
        self.assertTrue(pipe["bart"]["available"])
        self.assertIn("nllb_model", pipe["mbart_nllb"])
        self.assertIn("model", pipe["bart"])


if __name__ == "__main__":
    unittest.main()
