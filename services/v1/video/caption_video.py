"""
Optimized GPU-accelerated captioning pipeline (stable)
====================================================
Changes in this revision
-----------------------
1. **Avoids ffmpeg‑python Node mutation error** by calling FFmpeg via `subprocess.run()` instead of chaining streams.
2. Keeps Whisper on float32 to prevent dtype mismatches.
3. Maintains CUDA NVDEC → GPU filters → NVENC path.

Copyright (c) 2025 Stephen G. Pope
GPL-2.0-or-later
"""

import os
import logging
import shlex
import subprocess
from urllib.parse import urlparse
from typing import Dict, List, Union

import ffmpeg  # still used only for `probe`
import requests
import srt
import torch
import whisper  # type: ignore

from services.file_management import download_file
from services.cloud_storage import upload_file  # noqa: F401 (integration hook)
from config import LOCAL_STORAGE_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(_h)

# ----------------------------------------------------------------------------
# CUDA helpers
# ----------------------------------------------------------------------------
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")

if USE_CUDA:
    logger.info(f"CUDA available — using GPU {CUDA_DEVICE}")
else:
    logger.warning("CUDA not available, falling back to CPU")

WHISPER_DEVICE = "cuda" if USE_CUDA else "cpu"
WHISPER_MODEL = None  # lazy load

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def get_video_resolution(path: str) -> tuple[int, int]:
    try:
        probe = ffmpeg.probe(path)
        vid = next(s for s in probe["streams"] if s["codec_type"] == "video")
        return int(vid["width"]), int(vid["height"])
    except Exception as exc:
        logger.warning(f"ffprobe failed: {exc}; defaulting to 1280x720")
        return 1280, 720


def is_url(text: str) -> bool:
    try:
        return urlparse(text).scheme in {"http", "https"}
    except Exception:
        return False


def download_captions(url: str) -> str:
    logger.info(f"Downloading captions → {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

# Placeholder for style handling helpers
try:
    from caption_helpers import process_subtitle_events  # external module
except ImportError:
    def process_subtitle_events(*args, **kwargs):  # type: ignore
        raise NotImplementedError(
            "process_subtitle_events() is missing. Import it from the original "
            "captioning code or integrate the style-handler section here."
        )

# ----------------------------------------------------------------------------
# Whisper transcription (float32)
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def process_captioning_v1(
    video_url: str,
    captions: Union[str, None],
    settings: Dict,
    replace: List[Dict[str, str]],
    job_id: str,
    language: str = "auto",
):
    try:
        # ---------------------------------------------------------------
        # Download video
        # ---------------------------------------------------------------
        try:
            video_path = download_file(video_url, LOCAL_STORAGE_PATH)
            logger.info(f"[{job_id}] Video downloaded → {video_path}")
        except Exception as e:
            return {"error": f"Download failed: {e}"}

        # ---------------------------------------------------------------
        # Prepare params
        # ---------------------------------------------------------------
        width, height = get_video_resolution(video_path)
        style_type = settings.get("style", "classic").lower()
        replace_dict = {item["find"]: item["replace"] for item in replace}

        # ---------------------------------------------------------------
        # Captions handling / transcription
        # ---------------------------------------------------------------
        if captions:
            cap_content = download_captions(captions) if is_url(captions) else captions
            if "[Script Info]" in cap_content:
                subtitle_text = cap_content  # already ASS
            else:
                transcription_like = {
                    "segments": [
                        {
                            "start": sub.start.total_seconds(),
                            "end": sub.end.total_seconds(),
                            "text": sub.content,
                            "words": [],
                        }
                        for sub in srt.parse(cap_content)
                    ]
                }
                subtitle_text = process_subtitle_events(
                    transcription_like, style_type, settings, replace_dict, (width, height)
                )
        else:
            transcription = generate_transcription(video_path, language)
            subtitle_text = process_subtitle_events(
                transcription, style_type, settings, replace_dict, (width, height)
            )

        # ---------------------------------------------------------------
        # Write ASS file
        # ---------------------------------------------------------------
        subtitle_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.ass")
        with open(subtitle_path, "w", encoding="utf-8") as fh:
            fh.write(subtitle_text)
        logger.info(f"[{job_id}] Subtitles saved → {subtitle_path}")

        # ---------------------------------------------------------------
        # Burn in subtitles via raw FFmpeg command
        # ---------------------------------------------------------------
        output_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_captioned.mp4")

        vf_filters = []
        # ensure even dimensions for NVENC
        if width % 2 or height % 2:
            vf_filters.append(f"scale_npp=w={width//2*2}:h={height//2*2}")
        vf_filters.append(f"subtitles={shlex.quote(subtitle_path)}")
        vf_param = ",".join(vf_filters)

        cmd = [
            "ffmpeg", "-y",  # overwrite
            "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", video_path,
            "-vf", vf_param,
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-qmin", "19",
            "-qmax", "23",
            "-c:a", "copy",
            output_path,
        ]

        logger.info(f"[{job_id}] Running FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"[{job_id}] FFmpeg stderr:\n{result.stderr}")
            return {"error": f"FFmpeg failed: {result.stderr.strip()}"}

        logger.info(f"[{job_id}] Captioned video written → {output_path}")
        return output_path

    except Exception as exc:
        logger.exception(f"[{job_id}] Unhandled exception: {exc}")
        return {"error": str(exc)}
