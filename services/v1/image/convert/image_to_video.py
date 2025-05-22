# Copyright (c) 2025 Stephen G. Pope
# GPL-2.0-or-later

import os
import subprocess
import logging
from typing import Optional
from services.file_management import download_file
from PIL import Image

STORAGE_PATH = "/tmp/"
logger = logging.getLogger(__name__)


def process_image_to_video(
    image_url: str,
    length: float,
    frame_rate: int,
    zoom_speed: float,
    job_id: str,
    webhook_url: Optional[str] = None,
    *,
    use_gpu: bool = True,
    gpu_id: int = 0,
) -> str:
    """Create a Ken-Burns-style zoom video, accelerated with NVENC/CUDA."""
    try:
        # ── 1. Fetch image & inspect size ───────────────────────────────
        image_path = download_file(image_url, STORAGE_PATH)
        with Image.open(image_path) as img:
            width, height = img.size
        logger.info("Downloaded %s (%dx%d)", image_path, width, height)

        # ── 2. Geometry ─────────────────────────────────────────────────
        def even(v: int) -> int:     # NVENC requires even dimensions
            return v & ~1

        if width > height:           # landscape
            scale_dims = "7680:4320"
            out_w, out_h = even(1920), even(1080)
        else:                        # portrait
            scale_dims = "4320:7680"
            out_w, out_h = even(1080), even(1920)

        output_dims = f"{out_w}x{out_h}"
        total_frames = int(length * frame_rate)
        zoom_factor = 1 + zoom_speed * length
        output_path = os.path.join(STORAGE_PATH, f"{job_id}.mp4")

        # ── 3. Filter graph ────────────────────────────────────────────
        vf = (
            f"scale={scale_dims},"
            f"zoompan="
                f"z='min(1+({zoom_speed}*{length})*on/{total_frames},{zoom_factor})':"
                f"d={total_frames}:"
                f"x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':"
                f"s={output_dims},"
            f"fps={frame_rate},"
            "format=nv12,hwupload_cuda"
        )

        # ── 4. Encoder & flags ─────────────────────────────────────────
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-framerate", str(frame_rate),
            "-i", image_path,
            "-t", str(length),
            "-vf", vf,
        ]

        if use_gpu:
            cmd += [
                "-c:v", "h264_nvenc",
                "-gpu", str(gpu_id),
                "-preset", "p4",
                "-b:v", "10M",
            ]
        else:  # pure CPU fall-back
            cmd += [
                "-c:v", "libx264",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-b:v", "10M",
            ]

        cmd += [
            "-movflags", "+faststart",
            "-y",
            output_path,
        ]

        # ── 5. Run FFmpeg ──────────────────────────────────────────────
        logger.info("Running FFmpeg:\n%s", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode:
            logger.error("FFmpeg failed (%s):\n%s", res.returncode, res.stderr)
            raise subprocess.CalledProcessError(
                res.returncode, cmd, res.stdout, res.stderr
            )

        # ── 6. Cleanup & return ────────────────────────────────────────
        os.remove(image_path)
        logger.info("Video created: %s", output_path)
        return output_path

    except Exception:
        logger.exception("process_image_to_video encountered an error")
        raise
