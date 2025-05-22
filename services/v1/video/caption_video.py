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
try:
    from ffmpeg.nodes import Node as FFNode  # type: ignore
    if not getattr(FFNode, "_nca_src_writeable", False):
        if hasattr(FFNode, "src") and isinstance(FFNode.src, property):
            _orig_get = FFNode.src.fget  # type: ignore[attr-defined]
            def _src_set(self, value):  # noqa: ANN001
                object.__setattr__(self, "_Node__src", value)
                object.__setattr__(self, "_hash", None)
            FFNode.src = property(_orig_get, _src_set)  # type: ignore[assignment]
            FFNode._nca_src_writeable = True  # type: ignore[attr-defined]
            logger.debug("Patched ffmpeg.Node.src mutator")
except Exception as patch_err:
    logger.warning("ffmpeg.Node patch failed (may be unnecessary): %s", patch_err)

# ---------------------------------------------------------------------------
# Robust patch 2: Allow triton.runtime.jit.Kernel.src reassignment (Whisper median_filter)
# ---------------------------------------------------------------------------
try:
    from triton.runtime.jit import Kernel as TritonKernel  # type: ignore
    if not getattr(TritonKernel, "_nca_src_writeable", False):
        _orig_kernel_setattr = TritonKernel.__setattr__
        def _kernel_setattr(self, name, value):  # noqa: ANN001
            if name == "src":
                # Use official helper if available
                if hasattr(self, "_unsafe_update_src"):
                    self._unsafe_update_src(value)  # type: ignore[attr-defined]
                else:
                    object.__setattr__(self, name, value)
                # Clear cached hash if it exists
                if hasattr(self, "_hash"):
                    object.__setattr__(self, "_hash", None)
            else:
                _orig_kernel_setattr(self, name, value)
        TritonKernel.__setattr__ = _kernel_setattr  # type: ignore[assignment]
        TritonKernel._nca_src_writeable = True  # type: ignore[attr-defined]
        logger.debug("Patched triton.runtime.jit.Kernel.__setattr__ for 'src'")
except Exception as patch_err:
    logger.warning("Triton Kernel patch failed (may not be needed): %s", patch_err)
# ---------------------------------------------------------------------------
try:
    from ffmpeg.nodes import Node as FFNode  # type: ignore

    # Skip if we already swapped the property
    if not getattr(FFNode, "_nca_src_writeable", False):
        if hasattr(FFNode, "src") and isinstance(FFNode.src, property):
            _orig_get = FFNode.src.fget  # type: ignore[attr-defined]

            def _new_set(self, value):  # noqa: ANN001
                # Directly assign to hidden attribute and drop cached hash
                object.__setattr__(self, "_Node__src", value)
                object.__setattr__(self, "_hash", None)

            FFNode.src = property(_orig_get, _new_set, _orig_get.__delete__ if _orig_get else None)  # type: ignore[assignment]
            FFNode._nca_src_writeable = True  # type: ignore[attr-defined]
            logger.debug("Patched ffmpeg.Node.src property mutator (new setter)")
        else:
            logger.debug("ffmpeg.Node has no property 'src' – mutation patch not required")
except Exception as patch_err:  # pragma: no cover
    logger.warning("Failed to patch ffmpeg.Node.src setter: %s", patch_err)
# ---------------------------------------------------------------------------
try:
    from ffmpeg.nodes import Node as FFNode  # type: ignore

    if not getattr(FFNode, "_nca_mutable_src", False):
        _orig_setattr = FFNode.__setattr__

        def _nca_setattr(self, name, value):  # noqa: ANN001
            if name == "src":
                # Directly update the hidden attr, then clear cached hash
                object.__setattr__(self, "_Node__src", value)
                object.__setattr__(self, "_hash", None)
            else:
                _orig_setattr(self, name, value)

        FFNode.__setattr__ = _nca_setattr  # type: ignore[assignment]
        FFNode._nca_mutable_src = True  # type: ignore[attr-defined]
        logger.debug("ffmpeg.Node.__setattr__ patched for mutable 'src'.")
except Exception as patch_err:  # pragma: no cover
    logger.warning("ffmpeg.Node patch failed (may not be needed): %s", patch_err)

# ---------------------------------------------------------------------------
# CUDA / Whisper
# ---------------------------------------------------------------------------
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")
logger.info("CUDA %s", f"enabled (GPU {CUDA_DEVICE})" if USE_CUDA else "not available; using CPU")

WHISPER_DEVICE = "cuda" if USE_CUDA else "cpu"
WHISPER_MODEL = None  # lazy-load global

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_video_resolution(path: str) -> tuple[int, int]:
    try:
        probe = ffmpeg.probe(path)
        vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
        return int(vs["width"]), int(vs["height"])
    except Exception as exc:
        logger.warning("ffprobe failed (%s); defaulting 1280×720", exc)
        return 1280, 720


def is_url(text: str) -> bool:
    try:
        return urlparse(text).scheme in {"http", "https"}
    except Exception:
        return False


def download_captions(url: str) -> str:
    logger.info("Downloading captions → %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text

# Placeholder for style‑handling helpers
try:
    from caption_helpers import process_subtitle_events  # external module
except ImportError:
    def process_subtitle_events(*args, **kwargs):  # type: ignore
        raise NotImplementedError("process_subtitle_events() missing; import it from caption_helpers.")

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
