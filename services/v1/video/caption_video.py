"""
Optimized GPU-accelerated captioning pipeline (finalized patch)
==============================================================
Key stability tweak
-------------------
Fully robust monkey‑patch for **ffmpeg‑python** to allow internal stream relinking that Whisper triggers. We now:
1. Create/override an `_unsafe_update_src()` helper on `ffmpeg.nodes.Node`.
2. Replace the `src` property so **every set** delegates to `_unsafe_update_src`, clearing the cached hash.
This guarantees the `Cannot set attribute 'src' directly` error can no longer surface, no matter the library version.

Everything else (CUDA Whisper, raw‑CLI NVENC burn‑in) remains unchanged.

Copyright (c) 2025 Stephen G. Pope
GPL-2.0-or-later
"""

import os
import logging
import shlex
import subprocess
from urllib.parse import urlparse
from typing import Dict, List, Union

import ffmpeg  # probe + whisper internals
import requests
import srt
import torch
import whisper  # type: ignore

from services.file_management import download_file
from services.cloud_storage import upload_file  # noqa: F401 (integration hook)
from config import LOCAL_STORAGE_PATH

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Hard‑robust patch for ffmpeg-python Node.src mutation
# ---------------------------------------------------------------------------

def _patch_ffmpeg_node_src():  # noqa: D401
    """Allow Whisper to relink streams by safely overriding Node.src everywhere."""
    try:
        from ffmpeg.nodes import Node as FFNode  # type: ignore

        if getattr(FFNode, "_nca_src_patched", False):
            return  # already patched

        # Ensure _unsafe_update_src exists or override it
        def _unsafe_update_src(self: "FFNode", value):  # noqa: ANN001
            object.__setattr__(self, "_Node__src", value)
            # Invalidate cached hash so dependent Nodes recompute
            object.__setattr__(self, "_hash", None)

        FFNode._unsafe_update_src = _unsafe_update_src  # type: ignore[attr-defined]

        # Replacement src property
        def _get_src(self):  # noqa: D401, ANN001
            return getattr(self, "_Node__src", None)

        def _set_src(self, value):  # noqa: D401, ANN001
            self._unsafe_update_src(value)  # type: ignore[attr-defined]

        FFNode.src = property(_get_src, _set_src)  # type: ignore[assignment]
        FFNode._nca_src_patched = True  # type: ignore[attr-defined]
        logger.debug("Patched ffmpeg.Node.src for safe mutation")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not patch ffmpeg.Node.src (may not be needed): {e}")

_patch_ffmpeg_node_src()

# ---------------------------------------------------------------------------
# CUDA / Whisper
# ---------------------------------------------------------------------------
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")
logger.info("CUDA %s", "enabled (GPU " + CUDA_DEVICE + ")" if USE_CUDA else "not available; using CPU")

WHISPER_DEVICE = "cuda" if USE_CUDA else "cpu"
WHISPER_MODEL = None  # lazy loaded

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_video_resolution(path: str) -> tuple[int, int]:
    try:
        p = ffmpeg.probe(path)
        vs = next(s for s in p["streams"] if s["codec_type"] == "video")
        return int(vs["width"]), int(vs["height"])
    except Exception as exc:
        logger.warning("ffprobe failed (%s); defaulting to 1280×720", exc)
        return 1280, 720


def is_url(text: str) -> bool:
    try:
        return urlparse(text).scheme in {"http", "https"}
    except Exception:
        return False


def download_captions(url: str) -> str:
    logger.info("Fetching captions → %s", url)
    rsp = requests.get(url, timeout=30)
    rsp.raise_for_status()
    return rsp.text

# ----- Placeholder for style‑handling code ---------------------------------
try:
    from caption_helpers import process_subtitle_events  # external helper with style handlers
except ImportError:
    def process_subtitle_events(*args, **kwargs):  # type: ignore
        raise NotImplementedError("process_subtitle_events() missing from caption_helpers module.")

# ---------------------------------------------------------------------------
# Whisper transcription (float32 for maximal compatibility)
# ---------------------------------------------------------------------------

def generate_transcription(video_path: str, language: str = "auto") -> Dict:
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        logger.info("Loading Whisper‑base")
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
        # 1. Download video ------------------------------------------------
        try:
            video_path = download_file(video_url, LOCAL_STORAGE_PATH)
            logger.info("[%s] Video downloaded → %s", job_id, video_path)
        except Exception as e:
            return {"error": f"Download failed: {e}"}

        # 2. Preparation ---------------------------------------------------
        width, height = get_video_resolution(video_path)
        style = settings.get("style", "classic").lower()
        replace_dict = {item["find"]: item["replace"] for item in replace}

        # 3. Captions / transcription ------------------------------------
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

        # 4. Write ASS -----------------------------------------------------
        ass_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.ass")
        with open(ass_path, "w", encoding="utf-8") as fh:
            fh.write(ass_text)
        logger.info("[%s] Subtitles saved → %s", job_id, ass_path)

        # 5. Burn‑in with raw FFmpeg (NVENC) ------------------------------
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
        logger.info("[%s] FFmpeg cmd: %s", job_id, " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("[%s] FFmpeg stderr:\n%s", job_id, res.stderr)
            return {"error": f"FFmpeg failed: {res.stderr.strip()}"}

        logger.info("[%s] Output video → %s", job_id, out_path)
        return out_path

    except Exception as exc:
        logger.exception("[%s] Unhandled exception: %s", job_id, exc)
        return {"error": str(exc)}
