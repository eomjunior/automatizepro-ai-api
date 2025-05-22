"""
Optimized GPU‑accelerated captioning pipeline (robust patch)
===========================================================
• Eliminates Node .src mutation crash regardless of ffmpeg‑python version.
• Conditional monkey‑patch only if the property actually exists.
• Fallbacks gracefully if no patch needed.

Copyright (c) 2025 Stephen G. Pope
GPL‑2.0‑or‑later
"""

import os
import logging
import shlex
import subprocess
from urllib.parse import urlparse
from typing import Dict, List, Union

import ffmpeg  # only for probe + whisper internals
import requests
import srt
import torch
import whisper  # type: ignore

from services.file_management import download_file
from services.cloud_storage import upload_file  # noqa: F401 (integration hook)
from config import LOCAL_STORAGE_PATH

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
# Safe monkey‑patch for ffmpeg‑python Node.src (needed by Whisper <2.0)
# ---------------------------------------------------------------------------
try:
    from ffmpeg.nodes import Node as _FFNode  # type: ignore

    if not hasattr(_FFNode, "_patched_allow_src_mutation"):
        if hasattr(_FFNode, "src") and isinstance(_FFNode.src, property):
            _orig_get = _FFNode.src.fget  # type: ignore[attr-defined]

            def _set_src(self: "_FFNode", value):  # noqa: D401, ANN001
                """Allow runtime stream relinking used by whisper’s audio loader."""
                object.__setattr__(self, "_Node__src", value)
                # Clear cached hash so graph stays consistent
                object.__setattr__(self, "_hash", None)

            _FFNode.src = property(_orig_get, _set_src)  # type: ignore[assignment]
            logger.debug("Patched ffmpeg.Node.src setter for safe mutation")
        else:
            logger.debug("ffmpeg.Node has no 'src' property; monkey‑patch skipped")
        _FFNode._patched_allow_src_mutation = True  # type: ignore[attr-defined]
except Exception as patch_err:
    logger.warning(f"Could not patch ffmpeg.Node.src (safe to ignore if not needed): {patch_err}")

# ---------------------------------------------------------------------------
# CUDA / Whisper settings
# ---------------------------------------------------------------------------
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")

if USE_CUDA:
    logger.info(f"CUDA available — using GPU {CUDA_DEVICE}")
else:
    logger.warning("CUDA not available; falling back to CPU (slower)")

WHISPER_DEVICE = "cuda" if USE_CUDA else "cpu"
WHISPER_MODEL = None  # lazy‑load global

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_video_resolution(path: str) -> tuple[int, int]:
    try:
        p = ffmpeg.probe(path)
        vs = next(s for s in p["streams"] if s["codec_type"] == "video")
        return int(vs["width"]), int(vs["height"])
    except Exception as exc:
        logger.warning(f"ffprobe failed ({exc}); default 1280×720")
        return 1280, 720


def is_url(text: str) -> bool:
    try:
        return urlparse(text).scheme in {"http", "https"}
    except Exception:
        return False


def download_captions(url: str) -> str:
    logger.info(f"Fetching captions → {url}")
    rsp = requests.get(url, timeout=30)
    rsp.raise_for_status()
    return rsp.text

# Style‑handlers placeholder (import from your original implementation)
try:
    from caption_helpers import process_subtitle_events  # external helper
except ImportError:
    def process_subtitle_events(*args, **kwargs):  # type: ignore
        raise NotImplementedError(
            "process_subtitle_events() missing. Import or merge style logic into this file."
        )

# ---------------------------------------------------------------------------
# Whisper transcription (float32)
# ---------------------------------------------------------------------------

def generate_transcription(video_path: str, language: str = "auto") -> Dict:
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        logger.info("Loading Whisper‑base (float32)")
        WHISPER_MODEL = whisper.load_model("base", device=WHISPER_DEVICE)
    opts = {
        "word_timestamps": True,
        "language": None if language == "auto" else language,
        "verbose": False,
    }
    return WHISPER_MODEL.transcribe(video_path, **opts)

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
    try:
        # 1) Download video -------------------------------------------------
        try:
            video_path = download_file(video_url, LOCAL_STORAGE_PATH)
            logger.info(f"[{job_id}] Video downloaded → {video_path}")
        except Exception as e:
            return {"error": f"Download failed: {e}"}

        # 2) Setup ---------------------------------------------------------
        width, height = get_video_resolution(video_path)
        style = settings.get("style", "classic").lower()
        replace_dict = {item["find"]: item["replace"] for item in replace}

        # 3) Captions / transcription -------------------------------------
        if captions:
            cap_text = download_captions(captions) if is_url(captions) else captions
            if "[Script Info]" in cap_text:
                ass_text = cap_text  # already ASS
            else:
                tr_like = {
                    "segments": [
                        {
                            "start": s.start.total_seconds(),
                            "end": s.end.total_seconds(),
                            "text": s.content,
                            "words": [],
                        }
                        for s in srt.parse(cap_text)
                    ]
                }
                ass_text = process_subtitle_events(tr_like, style, settings, replace_dict, (width, height))
        else:
            transcription = generate_transcription(video_path, language)
            ass_text = process_subtitle_events(transcription, style, settings, replace_dict, (width, height))

        # 4) Write ASS file -----------------------------------------------
        ass_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.ass")
        with open(ass_path, "w", encoding="utf-8") as fh:
            fh.write(ass_text)
        logger.info(f"[{job_id}] Subtitles saved → {ass_path}")

        # 5) Burn‑in with raw FFmpeg --------------------------------------
        out_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_captioned.mp4")
        vf_parts = []
        if width % 2 or height % 2:
            vf_parts.append(f"scale_npp=w={width//2*2}:h={height//2*2}")
        vf_parts.append(f"subtitles={shlex.quote(ass_path)}")
        vf_arg = ",".join(vf_parts)

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", video_path,
            "-vf", vf_arg,
            "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-qmin", "19", "-qmax", "23",
            "-c:a", "copy",
            out_path,
        ]
        logger.info(f"[{job_id}] FFmpeg cmd: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"[{job_id}] FFmpeg stderr:\n{res.stderr}")
            return {"error": f"FFmpeg failed: {res.stderr.strip()}"}

        logger.info(f"[{job_id}] Output → {out_path}")
        return out_path

    except Exception as exc:
        logger.exception(f"[{job_id}] Unhandled exception: {exc}")
        return {"error": str(exc)}
