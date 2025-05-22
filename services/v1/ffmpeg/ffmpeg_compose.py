# Copyright (c) 2025 Stephen G. Pope
# GPL-2.0-or-later
#
# GPU-optimised version – adds CUDA hw-accel decoding and NVENC encoding
# whenever the caller hasn’t already chosen explicit codecs / options.

import os
import subprocess
import json
from typing import List, Tuple

from services.file_management import download_file
from config import LOCAL_STORAGE_PATH


# ──────────────────────────── Helpers ────────────────────────────

def get_extension_from_format(format_name: str) -> str:
    mapping = {
        'mp4': 'mp4', 'mov': 'mov', 'avi': 'avi', 'mkv': 'mkv', 'webm': 'webm',
        'gif': 'gif', 'apng': 'apng', 'jpg': 'jpg', 'jpeg': 'jpg',
        'png': 'png', 'image2': 'png', 'rawvideo': 'raw',
        'mp3': 'mp3', 'wav': 'wav', 'aac': 'aac', 'flac': 'flac', 'ogg': 'ogg'
    }
    return mapping.get(format_name.lower(), 'mp4')


def option_present(opts: List[dict], flag: str) -> bool:
    """True if an FFmpeg option (e.g. -c:v) already exists."""
    return any(o["option"] == flag for o in opts)


def get_metadata(filename: str, meta_req: dict) -> dict:
    """Unchanged – trimmed for brevity."""
    # … keep your existing implementation here …
    # (No GPU-specific changes needed)
    return {}


# ───────────────────────── Process function ──────────────────────

def process_ffmpeg_compose(data: dict, job_id: str,
                           enable_gpu: bool = True) -> Tuple[List[str], List[dict]]:
    """
    Build and run FFmpeg command described by *data*.
    If *enable_gpu* is True (default) the function injects:
      • Global:     -hwaccel cuda -hwaccel_output_format cuda
      • Per output: -c:v h264_nvenc  (unless caller set -c:v / the output is audio-only)
    """
    outputs_meta = []
    output_files = []

    # ── 1. Base command and global opts ───────────────────────────
    cmd = ["ffmpeg", "-y"]                       # overwrite
    if enable_gpu:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

    for g in data.get("global_options", []):
        cmd += [g["option"]] + ([str(g["argument"])] if g.get("argument") is not None else [])

    # ── 2. Inputs — download + attach ────────────────────────────
    for inp in data["inputs"]:
        for o in inp.get("options", []):
            cmd += [o["option"]] + ([str(o["argument"])] if o.get("argument") is not None else [])
        inp_path = download_file(inp["file_url"], LOCAL_STORAGE_PATH)
        cmd += ["-i", inp_path]
        inp["_local_path"] = inp_path           # remember for later cleanup

    # ── 3. Filter graph (unchanged) ──────────────────────────────
    if data.get("filters"):
        cmd += ["-filter_complex", ";".join(f["filter"] for f in data["filters"])]

    # ── 4. Outputs  ──────────────────────────────────────────────
    for idx, out in enumerate(data["outputs"]):
        opts = out.get("options", [])
        fmt_flag = next((o.get("argument") for o in opts if o["option"] == "-f"), None)
        ext = get_extension_from_format(fmt_flag) if fmt_flag else "mp4"

        # inject GPU encoder if (a) user didn’t pick one AND (b) container is video
        if enable_gpu and ext not in {"mp3", "wav", "flac", "aac", "ogg"} \
           and not option_present(opts, "-c:v"):
            opts.extend([
                {"option": "-c:v", "argument": "h264_nvenc"},
                # you can tweak NVENC preset / bitrate here if desired
                {"option": "-preset", "argument": "p4"}  # good Turing/Ampere default
            ])

        # append options to command
        for o in opts:
            cmd += [o["option"]] + ([str(o["argument"])] if o.get("argument") is not None else [])

        out_name = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_output_{idx}.{ext}")
        output_files.append(out_name)
        cmd.append(out_name)

    # ── 5. Run FFmpeg ────────────────────────────────────────────
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed ({e.returncode}):\n{e.stderr}") from e

    # ── 6. Cleanup inputs ────────────────────────────────────────
    for inp in data["inputs"]:
        try:
            os.remove(inp["_local_path"])
        except FileNotFoundError:
            pass

    # ── 7. Optional metadata ────────────────────────────────────
    if data.get("metadata"):
        for f in output_files:
            outputs_meta.append(get_metadata(f, data["metadata"]))

    return output_files, outputs_meta
