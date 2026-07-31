"""Tests for multi-format meeting export (TXT / DOCX / PDF)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services import export as export_svc  # noqa: E402


def _sample_meeting():
    seg = SimpleNamespace(
        text="Maayong aga.",
        start_time=1.5,
        end_time=3.0,
    )
    return SimpleNamespace(
        title="Board huddle",
        venue="Iloilo Hall",
        language="hil",
        status="finalized",
        meeting_date=None,
        final_transcript="Maayong aga. Padayon kita.",
        translation="Good morning. Let us continue.",
        translation_language="en",
        summary="• Continue the project\n• Assign owners",
        summary_format="bullets",
        segments=[seg],
    )


class ExportServiceTests(unittest.TestCase):
    def test_txt_contains_core_sections(self):
        data = export_svc.render_txt(_sample_meeting()).decode("utf-8")
        self.assertIn("Verbatim transcript", data)
        self.assertIn("Maayong aga", data)
        self.assertIn("English translation", data)
        self.assertIn("Good morning", data)
        self.assertIn("Structured summary", data)
        self.assertIn("Timestamped segments", data)

    def test_docx_and_pdf_nonempty(self):
        meeting = _sample_meeting()
        docx_bytes, docx_media, docx_name = export_svc.export_meeting(meeting, "docx")
        pdf_bytes, pdf_media, pdf_name = export_svc.export_meeting(meeting, "pdf")
        self.assertTrue(docx_bytes.startswith(b"PK"))
        self.assertIn("wordprocessingml", docx_media)
        self.assertTrue(docx_name.endswith(".docx"))
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertEqual(pdf_media, "application/pdf")
        self.assertTrue(pdf_name.endswith(".pdf"))

    def test_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            export_svc.export_meeting(_sample_meeting(), "rtf")


if __name__ == "__main__":
    unittest.main()
