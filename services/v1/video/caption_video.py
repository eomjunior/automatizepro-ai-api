"""
Optimized GPU-accelerated captioning pipeline (safe precision)
============================================================
• CUDA acceleration for Whisper + FFmpeg.
• Whisper remains in *float32* to avoid dtype mismatch errors.
• NVDEC → GPU filters → NVENC for subtitle burn‑in.

Copyright (c) 2025 Stephen G. Pope
GPL-2.0-or-later
"""

import os
import logging
from urllib.parse import urlparse
from typing import Dict, List, Union

import ffmpeg
import requests
import srt
import torch
import whisper  # type: ignore

from services.file_management import download_file
from services.cloud_storage import upload_file  # noqa: F401 (integration hook)
from config import LOCAL_STORAGE_PATH

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(_handler)

# ----------------------------------------------------------------------------
# CUDA helpers
# ----------------------------------------------------------------------------
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")

if USE_CUDA:
    logger.info(f"CUDA available — using GPU {CUDA_DEVICE}")
else:
    logger.warning("CUDA not available, falling back to CPU (slower)")

WHISPER_DEVICE = "cuda" if USE_CUDA else "cpu"
WHISPER_MODEL = None  # lazy‑load global

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def rgb_to_ass_color(rgb: str) -> str:
    rgb = rgb.lstrip("#")
    if len(rgb) != 6:
        return "&H00FFFFFF"
    r, g, b = (int(rgb[i : i + 2], 16) for i in (0, 2, 4))
    return f"&H00{b:02X}{g:02X}{r:02X}"


def get_video_resolution(path: str) -> tuple[int, int]:
    try:
        probe = ffmpeg.probe(path)
        vid_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        return int(vid_stream["width"]), int(vid_stream["height"])
    except Exception as exc:
        logger.warning(f"Resolution probe failed: {exc}. Falling back to 1280x720")
        return 1280, 720


def is_url(text: str) -> bool:
    try:
        return urlparse(text).scheme in {"http", "https"}
    except Exception:
        return False


def download_captions(url: str) -> str:
    logger.info(f"Fetching captions: {url}")
    rsp = requests.get(url, timeout=30)
    rsp.raise_for_status()
    return rsp.text

# ---------------------------------------------------------------------------
# Placeholder import for complex style handlers
# ---------------------------------------------------------------------------
try:
    from caption_helpers import process_subtitle_events  # external module
except ImportError:
    def process_subtitle_events(*args, **kwargs):  # type: ignore
        raise NotImplementedError(
            "process_subtitle_events() helper not found. Import from the original "
            "code or merge the style‑handler section into this file."
        )

# ----------------------------------------------------------------------------
# Whisper transcription (kept in float32 to avoid dtype mismatch with kernels)
# ----------------------------------------------------------------------------

def generate_transcription(video_path: str, language: str = "auto") -> Dict:
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        logger.info("Loading Whisper‑base model (float32)")
        WHISPER_MODEL = whisper.load_model("base", device=WHISPER_DEVICE)
    opts = {
        "word_timestamps": True,
        "language": None if language == "auto" else language,
        "verbose": False,
    }
    return WHISPER_MODEL.transcribe(video_path, **opts)

# ----------------------------------------------------------------------------
# Main GPU pipeline
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
        # Download video ----------------------------------------------------
        try:
            video_path = download_file(video_url, LOCAL_STORAGE_PATH)
            logger.info(f"[{job_id}] Video downloaded → {video_path}")
        except Exception as e:
            return {"error": f"Video download failed: {e}"}

        # Setup -------------------------------------------------------------
        resolution = get_video_resolution(video_path)
        style_type = settings.get("style", "classic").lower()
        replace_dict = {item["find"]: item["replace"] for item in replace}

        # Captions / transcription -----------------------------------------
        if captions:
            cap_content = download_captions(captions) if is_url(captions) else captions
            if "[Script Info]" in cap_content:
                subtitle_content = cap_content  # already ASS
            else:
                # quick SRT -> transcription result
                transcription_result = {
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
                subtitle_content = process_subtitle_events(
                    transcription_result, style_type, settings, replace_dict, resolution
                )
        else:
            transcription = generate_transcription(video_path, language)
            subtitle_content = process_subtitle_events(
                transcription, style_type, settings, replace_dict, resolution
            )

        # Write subtitles ---------------------------------------------------
        subtitle_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.ass")
        with open(subtitle_path, "w", encoding="utf-8") as fh:
            fh.write(subtitle_content)
        logger.info(f"[{job_id}] Subtitles saved → {subtitle_path}")

        # Burn in with FFmpeg + NVENC --------------------------------------
        output_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_captioned.mp4")
        vf_chain = [f"subtitles={subtitle_path}"]
        if resolution[0] % 2 or resolution[1] % 2:
            vf_chain.insert(0, f"scale_npp=w={resolution[0]//2*2}:h={resolution[1]//2*2}")
        vf_param = ",".join(vf_chain)

        try:
            (
                ffmpeg
                .input(video_path, hwaccel="cuda", hwaccel_output_format="cuda")
                .output(
                    output_path,
                    vf=vf_param,
                    vcodec="h264_nvenc",
                    acodec="copy",
                    preset="p4",
                    rc="vbr",
                    qmin=19,
                    qmax=23,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"[{job_id}] FFmpeg completed → {output_path}")
            return output_path
        except ffmpeg.Error as err:
            stderr = err.stderr.decode() if err.stderr else "(no stderr)"
            logger.error(f"[{job_id}] FFmpeg error:\n{stderr}")
            return {"error": f"FFmpeg error: {stderr}"}

    except Exception as exc:
        logger.exception(f"[{job_id}] Unhandled exception: {exc}")
        return {"error": str(exc)}
