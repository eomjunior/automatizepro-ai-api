"""
Optimized GPU-accelerated captioning pipeline
=============================================
• Robust patches for ffmpeg-python and Triton so Whisper can mutate `.src`
  without crashing.
• Uses CUDA decode (NVDEC) and NVENC encode when available.
• Delegates subtitle styling/event generation to `caption_helpers`.
• Filter graph now runs on CPU (`scale` + `subtitles`) to avoid CUDA↔CPU
  mismatch errors.
GPL-2.0-or-later — 2025 Stephen G. Pope
"""

from __future__ import annotations

import importlib
import logging
import os
import shlex
import subprocess
from typing import Dict, List, Tuple, Union
from urllib.parse import urlparse

import ffmpeg  # probe + used by Whisper internally
import requests
import srt
import torch
import whisper  # type: ignore

# Prefer relative import if helper lives alongside this file; fall back to absolute.
try:
    from .caption_helpers import process_subtitle_events
except ImportError:  # pragma: no cover
    from caption_helpers import process_subtitle_events  # type: ignore

from config import LOCAL_STORAGE_PATH
from services.cloud_storage import upload_file  # noqa: F401
from services.file_management import download_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Patch 1: make ffmpeg.nodes.Node.src write-able (Whisper loader)
# ---------------------------------------------------------------------------
try:
    from ffmpeg.nodes import Node as FFNode  # type: ignore[attr-defined]

    if not getattr(FFNode, "_nca_src_writeable", False):
        if hasattr(FFNode, "src") and isinstance(FFNode.src, property):
            _orig_get = FFNode.src.fget  # type: ignore[attr-defined]

            def _src_set(self, value):  # noqa: ANN001
                object.__setattr__(self, "_Node__src", value)
                object.__setattr__(self, "_hash", None)

            FFNode.src = property(_orig_get, _src_set)  # type: ignore[assignment]
            FFNode._nca_src_writeable = True  # type: ignore[attr-defined]
            logger.debug("Patched ffmpeg.Node.src mutator")
except Exception as exc:  # pragma: no cover
    logger.debug("ffmpeg.Node patch skipped/failed: %s", exc)

# ---------------------------------------------------------------------------
# Patch 2: allow Triton Kernel.src reassignment (covers old & new Triton)
# ---------------------------------------------------------------------------


def _patch_triton_kernel() -> None:
    """Make Triton Kernel.src mutable if Triton is present; stay silent otherwise."""
    for mod_path in ("triton.runtime.jit", "triton.runtime"):
        try:
            mod = importlib.import_module(mod_path)
            TritonKernel = getattr(mod, "Kernel", None)
            if TritonKernel and not getattr(TritonKernel, "_nca_src_writeable", False):
                _orig_setattr = TritonKernel.__setattr__

                def _kernel_setattr(self, name, value):  # noqa: ANN001
                    if name == "src":
                        if hasattr(self, "_unsafe_update_src"):
                            self._unsafe_update_src(value)  # type: ignore[attr-defined]
                        else:
                            object.__setattr__(self, name, value)
                        if hasattr(self, "_hash"):
                            object.__setattr__(self, "_hash", None)
                    else:
                        _orig_setattr(self, name, value)

                TritonKernel.__setattr__ = _kernel_setattr  # type: ignore[assignment]
                TritonKernel._nca_src_writeable = True  # type: ignore[attr-defined]
                logger.debug("Patched Triton Kernel.__setattr__ for 'src' (%s)", mod_path)
            return
        except ModuleNotFoundError:
            continue  # Triton not installed under this path — try next
        except Exception as exc:  # pragma: no cover
            logger.debug("Triton Kernel patch attempt failed (%s): %s", mod_path, exc)
            return
    logger.debug("Triton not available — Kernel patch skipped")


_patch_triton_kernel()

# ---------------------------------------------------------------------------
# CUDA / Whisper
# ---------------------------------------------------------------------------
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")
logger.info("CUDA %s", f"enabled (GPU {CUDA_DEVICE})" if USE_CUDA else "not available; using CPU")

WHISPER_DEVICE = "cuda" if USE_CUDA else "cpu"
WHISPER_MODEL = None  # lazy-loaded global

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_video_resolution(path: str) -> Tuple[int, int]:
    """Return (w, h) via ffprobe or default 1280×720 on failure."""
    try:
        probe = ffmpeg.probe(path)
        vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
        return int(vs["width"]), int(vs["height"])
    except Exception as exc:  # pragma: no cover
        logger.warning("ffprobe failed (%s); defaulting 1280×720", exc)
        return 1280, 720


def _is_url(text: str) -> bool:
    try:
        return urlparse(text).scheme in {"http", "https"}
    except Exception:
        return False


def _download_captions(url: str) -> str:
    logger.info("Downloading captions → %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Whisper transcription helper
# ---------------------------------------------------------------------------


def _generate_transcription(video_path: str, language: str = "auto") -> Dict:
    """Run Whisper transcription with optional word timestamps."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        logger.info("Loading Whisper-base model → %s", WHISPER_DEVICE)
        WHISPER_MODEL = whisper.load_model("base", device=WHISPER_DEVICE)

    opts = {
        "word_timestamps": True,
        "language": None if language == "auto" else language,
        "verbose": False,
    }

    try:
        return WHISPER_MODEL.transcribe(video_path, **opts)
    except AttributeError as err:
        if "Cannot set attribute 'src'" in str(err):
            logger.warning(
                "Triton mutation bug encountered. "
                "Retrying transcription without word-level timing."
            )
            opts["word_timestamps"] = False
            return WHISPER_MODEL.transcribe(video_path, **opts)
        raise


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_captioning_v1(
    video_url: str,
    captions: Union[str, None],
    settings: Dict,
    replace: List[Dict[str, str]],
    job_id: str,
    language: str = "auto",
):
    """
    1. Download video
    2. Transcribe (or load SRT/ASS)
    3. Build ASS with caption_helpers
    4. Burn subtitles using NVENC
    """
    try:
        # -----------------------------------------------------------------
        # 1. Download source video
        # -----------------------------------------------------------------
        try:
            video_path = download_file(video_url, LOCAL_STORAGE_PATH)
            logger.info("[%s] Video downloaded → %s", job_id, video_path)
        except Exception as exc:
            return {"error": f"Download failed: {exc}"}

        # -----------------------------------------------------------------
        # 2. Prepare vars
        # -----------------------------------------------------------------
        width, height = _get_video_resolution(video_path)
        style = settings.get("style", "classic").lower()
        replace_dict = {item["find"]: item["replace"] for item in replace}

        # -----------------------------------------------------------------
        # 3. Get captions / transcription
        # -----------------------------------------------------------------
        if captions:
            cap_content = _download_captions(captions) if _is_url(captions) else captions
            if "[Script Info]" in cap_content:  # already ASS
                ass_text = cap_content
            else:  # treat as SRT
                transcription_like = {
                    "segments": [
                        {
                            "start": s.start.total_seconds(),
                            "end": s.end.total_seconds(),
                            "text": s.content,
                            "words": [],
                        }
                        for s in srt.parse(cap_content)
                    ]
                }
                ass_text = process_subtitle_events(
                    transcription_like, style, settings, replace_dict, (width, height)
                )
        else:
            transcription = _generate_transcription(video_path, language)
            ass_text = process_subtitle_events(
                transcription, style, settings, replace_dict, (width, height)
            )

        # -----------------------------------------------------------------
        # 4. Save ASS file
        # -----------------------------------------------------------------
        ass_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.ass")
        with open(ass_path, "w", encoding="utf-8") as fp:
            fp.write(ass_text)
        logger.info("[%s] Subtitles saved → %s", job_id, ass_path)

        # -----------------------------------------------------------------
        # 5. Burn-in subtitles with FFmpeg
        # -----------------------------------------------------------------
        output_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_captioned.mp4")

        vf_parts: List[str] = []
        # Ensure even dimensions for h264
        if width % 2 or height % 2:
            vf_parts.append(f"scale=w={width//2*2}:h={height//2*2}")  # CPU scale

        vf_parts.append(f"subtitles={shlex.quote(ass_path)}")
        vf_arg = ",".join(vf_parts)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",       # NVDEC decode (frames land in CPU memory)
            "-i",
            video_path,
            "-vf",
            vf_arg,
            "-c:v",
            "h264_nvenc",  # NVENC encode
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-qmin",
            "19",
            "-qmax",
            "23",
            "-c:a",
            "copy",
            output_path,
        ]
        logger.info("[%s] FFmpeg cmd: %s", job_id, " ".join(cmd))

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("[%s] FFmpeg stderr:\n%s", job_id, res.stderr)
            return {"error": f"FFmpeg failed: {res.stderr.strip()}"}

        logger.info("[%s] Output video → %s", job_id, output_path)
        return output_path

    except Exception as exc:  # pragma: no cover
        logger.exception("[%s] Unhandled exception: %s", job_id, exc)
        return {"error": str(exc)}
