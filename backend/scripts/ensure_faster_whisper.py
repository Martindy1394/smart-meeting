"""Install faster-whisper into the current interpreter if it is missing.

Windows Python 3.13 often cannot install pinned ``requirements-ml.txt``
(torch==2.5.1 / sentencepiece==0.2.0). Whisper only needs faster-whisper.
"""
from __future__ import annotations

import subprocess
import sys

FASTER_WHISPER_PIP = "faster-whisper==1.1.0"


def main() -> int:
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        print(f"faster-whisper already installed ({sys.executable})")
        return 0
    except Exception as exc:
        print(f"faster-whisper missing in {sys.executable}: {exc}")

    print(f"Installing {FASTER_WHISPER_PIP} …")
    return subprocess.call(
        [sys.executable, "-m", "pip", "install", FASTER_WHISPER_PIP]
    )


if __name__ == "__main__":
    raise SystemExit(main())
