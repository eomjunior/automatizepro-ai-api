# Copyright (c) 2025 Stephen G. Pope
# GNU GPL v2 or later – see original header for details
import os
import subprocess
import logging
from shutil import which
from services.file_management import download_file
from PIL import Image
from config import LOCAL_STORAGE_PATH   # ← unchanged

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU-optimised version – uses CUDA/NVENC
# ---------------------------------------------------------------------------
def process_image_to_video(
    image_url: str,
    length: float,
    frame_rate: int,
    zoom_speed: float,
    job_id: str,
    webhook_url: str | None = None,
    gpu_device: int = 0,                     # ★ new – choose /dev/nvidiaN
    bitrate: str = "10M",                   # ★ new – constant-bitrate hint
) -> str:
    """
    Creates a zoom-and-pan video from a still image, encoding with NVENC.

    Parameters
    ----------
    image_url   : remote or local path to the source image
    length      : video duration in seconds
    frame_rate  : output FPS
    zoom_speed  : zoom-in multiplier per second (e.g. 0.05 → 5 %/s)
    job_id      : unique id => filename <job_id>.mp4
    webhook_url : (unused here, reserved for caller)
    gpu_device  : which CUDA device to use (0-n)
    bitrate     : target bitrate for h264_nvenc (e.g. '10M', '25M')
    """
    # --- sanity checks -----------------------------------------------------
    if which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg not found in PATH")
    try:
        codec_list = subprocess.check_output(
            ["ffmpeg", "-v", "quiet", "-encoders"]
        ).decode()
    except subprocess.CalledProcessError as err:          # pragma: no cover
        logger.error("Cannot list ffmpeg encoders: %s", err)
        raise
    if "h264_nvenc" not in codec_list:
        raise RuntimeError(
            "ffmpeg was built without NVENC support (missing 'h264_nvenc')."
        )

    # --- stage 1 : fetch image locally ------------------------------------
    image_path = download_file(image_url, LOCAL_STORAGE_PATH)
    logger.info("Downloaded image → %s", image_path)

    # --- stage 2 : probe image size ---------------------------------------
    with Image.open(image_path) as img:
        width, height = img.size
    landscape = width >= height
    logger.info("Image size : %dx%d (%s)", width, height,
                "landscape" if landscape else "portrait")

    # --- stage 3 : derive video parameters --------------------------------
    output_path = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.mp4")
    scale_dims  = "7680:4320" if landscape else "4320:7680"
    output_dims = "1920x1080" if landscape else "1080x1920"
    output_dims_colon = output_dims.replace("x", ":")   # ← add or keep this line
    total_frames = int(length * frame_rate)
    zoom_factor  = 1 + zoom_speed * length            # final zoom at last frame

    logger.info("Frames : %d @ %dfps → %.2fs", total_frames,
                frame_rate, length)
    logger.info("Scale   : %s → %s", scale_dims, output_dims)
    logger.info("Zoom    : speed %.4f, final ×%.3f", zoom_speed, zoom_factor)


    encoder = "h264_nvenc"
    preset = "p4"
    hwaccel = ["-hwaccel", "cuda", "-hwaccel_device", str(0)]
    gpu_opt = ["-gpu", str(0)]   # harmless on new FFmpeg

    # ----------------------------------------------------------------------
    # GPU workflow:
    #   1. Zoom/pan (CPU – filter exists only in software)
    #   2. Upload each frame to GPU   -> hwupload_cuda
    #   3. Resize on-GPU              -> scale_cuda
    #   4. Encode NVENC               -> h264_nvenc
    # ----------------------------------------------------------------------
    vf = (
    	    f"scale={scale_dims},"
            f"zoompan="
            f"z='min(1+({zoom_speed}*{length})*on/{total_frames},{zoom_factor})':"
            f"d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"s={output_dims},"
            f"fps={frame_rate},"
            # Hand off to GPU
            f"format=nv12,hwupload_cuda,"
            # Correct CUDA resizer syntax (WIDTH:HEIGHT)
            f"scale_cuda={output_dims_colon}:interp_algo=lanczos"
    )
    cmd = [
    	    "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            *hwaccel,
            "-loop", "1",
            "-framerate", str(frame_rate),
            "-i", image_path,
            "-t", str(length),
            '-vf', f"scale={scale_dims},zoompan=z='min(1+({zoom_speed}*{length})*on/{total_frames}, {zoom_factor})':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={output_dims},fps={frame_rate},format=nv12",
            "-c:v", encoder,
            *gpu_opt,
            "-preset", preset,
            "-b:v", "10M",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y",
            output_path
    ]

    logger.info("FFmpeg (GPU) : %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("FFmpeg failed :\n%s", res.stderr)
        raise subprocess.CalledProcessError(res.returncode, cmd,
                                            res.stdout, res.stderr)

    logger.info("Video ready : %s (%.1f MiB)",
                output_path,
                os.path.getsize(output_path) / (1024 ** 2))

    # --- tidy up -----------------------------------------------------------
    os.remove(image_path)
    return output_path
