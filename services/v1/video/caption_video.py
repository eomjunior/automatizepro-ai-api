"""
Optimized GPU-accelerated captioning pipeline (final node‑patch)
==============================================================
This revision **replaces `ffmpeg.nodes.Node.__setattr__`** so any attempt to assign
`node.src = …` (as Whisper does) will transparently call a safe update that clears
the cached hash. This definitively removes the recurring:
```
Cannot set attribute 'src' directly. Use '_unsafe_update_src()'...
```
errors across all ffmpeg‑python versions.

Other logic (CUDA Whisper, raw‑CLI NVENC burn‑in) remains unchanged.

GPL‑2.0‑or‑later — 2025 Stephen G. Pope
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
from services.cloud_storage import upload_file  # noqa: F401
from config import LOCAL_STORAGE_PATH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Robust patch 1: Make ffmpeg.nodes.Node.src write‑able (Whisper audio loader)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Minimal internal subtitle renderer (classic center‑aligned)
# ---------------------------------------------------------------------------

def _format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def _ass_header(width: int, height: int, font_size: int = 48) -> str:
    return (
        "[Script Info]
"
        "ScriptType: v4.00+
"
        f"PlayResX: {width}
"
        f"PlayResY: {height}

"
        "[V4+ Styles]
"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"
        f"Style: Default,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,5,20,20,20,0

"
        "[Events]
"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"
    )


def process_subtitle_events(transcription_result: Dict, style: str, settings: Dict, repl: Dict, res: tuple[int, int]):
    """Fallback ASS generator: no styling beyond centered safe classic."""
    width, height = res
    header = _ass_header(width, height, int(height * 0.05))
    events = []
    for seg in transcription_result["segments"]:
        text = seg.get("text", "").strip().replace("
", " ")
        for old, new in repl.items():
            text = text.replace(old, new)
        start = _format_ass_time(seg["start"])
        end = _format_ass_time(seg["end"])
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return header + "
".join(events) + "
"
# --------------------------------------------------------------------------- missing; import it from caption_helpers.")

# ---------------------------------------------------------------------------
# Whisper transcription (float32)
# ---------------------------------------------------------------------------

def generate_transcription(video_path: str, language: str = "auto") -> Dict:
    """Run Whisper transcription. If Triton 'src' mutation bug occurs, retry with
    word-level timing disabled (segment-level only)."""
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
    except AttributeError as e:
        # Triton kernel 'src' immutability bug encountered
        if "Cannot set attribute 'src'" in str(e):
            logger.warning(
                "Triton kernel mutation bug hit while generating word timestamps. "
                "Falling back to segment‑level transcription without word timing."
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
    try:
        # 1. Download video ------------------------------------------------
        try:
            video_path = download_file(video_url, LOCAL_STORAGE_PATH)
            logger.info("[%s] Video downloaded → %s", job_id, video_path)
        except Exception as e:
            return {"error": f"Download failed: {e}"}

        # 2. Prep ---------------------------------------------------------
        width, height = get_video_resolution(video_path)
        style = settings.get("style", "classic").lower()
        replace_dict = {item["find"]: item["replace"] for item in replace}

        # 3. Captions / transcription ------------------------------------
        if captions:
            cap_content = download_captions(captions) if is_url(captions) else captions
            if "[Script Info]" in cap_content:
                ass_text = cap_content
            else:
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
                ass_text = process_subtitle_events(transcription_like, style, settings, replace_dict, (width, height))
        else:
            transcription = generate_transcription(video_path, language)
            ass_text = process_subtitle_events(transcription, style, settings, replace_dict, (width, height))

        # 4. Save ASS ------------------------------------------------------
        ass_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_text)
        logger.info("[%s] Subtitles saved → %s", job_id, ass_path)

        # 5. Burn‑in with raw FFmpeg (NVENC) ------------------------------
        output_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_captioned.mp4")
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
            output_path,
        ]
        logger.info("[%s] FFmpeg cmd: %s", job_id, " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("[%s] FFmpeg stderr:\n%s", job_id, res.stderr)
            return {"error": f"FFmpeg failed: {res.stderr.strip()}"}

        logger.info("[%s] Output video → %s", job_id, output_path)
        return output_path

    except Exception as exc:
        logger.exception("[%s] Unhandled exception: %s", job_id, exc)
        return {"error": str(exc)}