import os
import subprocess
import logging
from typing import Optional
from services.file_management import download_file     # unchanged
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
    use_gpu: bool = True,       # ⬅ enable/disable NVIDIA acceleration
    gpu_id: int = 0,            # ⬅ which GPU to use when multiple are present
) -> str:
    """
    Download an image and create an MP4 with a smooth Ken-Burns-style zoom.
    If `use_gpu` is True and FFmpeg is compiled with NVENC, encoding is done
    by the GPU (h264_nvenc). CPU encoding is used as a fallback.

    Returns
    -------
    str
        Absolute path of the rendered video.
    """
    try:
        # ------------------------------------------------------------------
        # 1. Download and inspect the image
        # ------------------------------------------------------------------
        image_path = download_file(image_url, STORAGE_PATH)
        logger.info("Downloaded image to %s", image_path)

        with Image.open(image_path) as img:
            width, height = img.size
        logger.info("Original image dimensions: %dx%d", width, height)

        # ------------------------------------------------------------------
        # 2. Build output & filter parameters
        # ------------------------------------------------------------------
        output_path = os.path.join(STORAGE_PATH, f"{job_id}.mp4")

        # ensure even numbers (required by many codecs – especially NVENC)
        def _even(value: int) -> int:
            return value - (value % 2)

        if width > height:            # landscape
            scale_dims = "7680:4320"
            output_dims = f"{_even(1920)}x{_even(1080)}"
        else:                         # portrait
            scale_dims = "4320:7680"
            output_dims = f"{_even(1080)}x{_even(1920)}"

        total_frames = int(length * frame_rate)
        zoom_factor = 1 + zoom_speed * length  # target zoom at last frame

        logger.info(
            "scale=%s, output=%s | length=%ss @ %dfps (%d frames) | zoom ×%.2f",
            scale_dims,
            output_dims,
            length,
            frame_rate,
            total_frames,
            zoom_factor,
        )

        # ------------------------------------------------------------------
        # 3. Choose encoder & hardware-accel flags
        # ------------------------------------------------------------------
        if use_gpu:
            encoder = "h264_nvenc"
            preset  = "p4"             # good quality / speed balance
            hwaccel = [
                "-hwaccel", "cuda",
                "-hwaccel_device", str(gpu_id),
            ]
        else:
            encoder = "libx264"
            preset  = "slow"
            hwaccel = []

        # ------------------------------------------------------------------
        # 4. Assemble and run the FFmpeg command
        # ------------------------------------------------------------------
        vf = (
            f"scale={scale_dims},"
            f"zoompan="
            f"z='min(1+({zoom_speed}*{length})*on/{total_frames},{zoom_factor})':"
            f"d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"s={output_dims}"
        )

        cmd = [
            "ffmpeg",
            "-y",                         # overwrite output if it exists
            *hwaccel,
            "-framerate", str(frame_rate),
            "-loop", "1",
            "-i", image_path,
            "-vf", vf,
            "-c:v", encoder,
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-t", str(length),
            "-movflags", "+faststart",    # web-friendly moov atom
            output_path,
        ]

        logger.info("Running FFmpeg:\n  %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("FFmpeg failed:\n%s", result.stderr)
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        logger.info("✅  Video created: %s", output_path)

        # ------------------------------------------------------------------
        # 5. Clean up
        # ------------------------------------------------------------------
        os.remove(image_path)
        return output_path

    except Exception as exc:
        logger.exception("process_image_to_video failed: %s", exc)
        raise
