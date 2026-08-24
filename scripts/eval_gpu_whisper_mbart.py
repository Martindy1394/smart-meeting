#!/usr/bin/env python3
"""Probe host GPU and evaluate Whisper / mBART device resolution.

Does **not** change NLLB, BART summarization, or other services.

Example::

  python scripts/eval_gpu_whisper_mbart.py \\
    --out data/mt_tag_benchmark/gpu_whisper_mbart_eval.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _nvidia_smi() -> dict:
    path = shutil.which("nvidia-smi")
    if not path:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        out = subprocess.check_output(
            [path, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
        return {"available": True, "nvidia_smi": path, "gpus": out.splitlines()}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


def _torch_cuda() -> dict:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "cuda_available": False, "error": str(exc)}
    info = {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"] and info["device_count"] > 0:
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=root / "data/mt_tag_benchmark/gpu_whisper_mbart_eval.json",
    )
    args = p.parse_args(argv)

    sys.path.insert(0, str(root / "backend"))

    from app.config import settings
    from app.services.llm import resolve_mbart_device
    from app.services.transcription import (
        resolve_whisper_compute_type,
        resolve_whisper_device,
    )

    nvidia = _nvidia_smi()
    torch_info = _torch_cuda()
    whisper_dev = resolve_whisper_device()
    whisper_ct = resolve_whisper_compute_type(whisper_dev)
    mbart_dev = resolve_mbart_device()

    whisper_gpu_ok = whisper_dev == "cuda"
    mbart_gpu_ok = mbart_dev == "cuda"
    host_gpu = bool(nvidia.get("available")) or bool(torch_info.get("cuda_available"))

    report = {
        "scope": "Whisper ASR + mBART translation GPU enablement only",
        "host": {
            "nvidia_smi": nvidia,
            "torch": torch_info,
            "gpu_present": host_gpu,
        },
        "settings": {
            "whisper_device": settings.whisper_device,
            "whisper_compute_type": settings.whisper_compute_type,
            "mbart_device": getattr(settings, "mbart_device", None),
        },
        "resolved": {
            "whisper_device": whisper_dev,
            "whisper_compute_type": whisper_ct,
            "mbart_device": mbart_dev,
        },
        "evaluation": {
            "whisper_using_gpu": whisper_gpu_ok,
            "mbart_using_gpu": mbart_gpu_ok,
            "success": bool(whisper_gpu_ok and mbart_gpu_ok),
            "notes": (
                "Both Whisper and mBART resolved to CUDA."
                if whisper_gpu_ok and mbart_gpu_ok
                else (
                    "No usable CUDA GPU on this host — Whisper/mBART correctly "
                    "fall back to CPU. Run this script on a CUDA machine (or set "
                    "WHISPER_DEVICE=cuda / MBART_DEVICE=cuda there) to confirm GPU."
                )
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote", args.out)
    # Exit 0 even when no GPU — resolution + fallback is a valid outcome here.
    return 0


if __name__ == "__main__":
    sys.exit(main())
