# Copyright (c) 2025 Stephen G. Pope
# GPL-2.0-or-later

import os
import ffmpeg
import requests  # kept for future webhook or uploads if you need it
from services.file_management import download_file
from config import LOCAL_STORAGE_PATH



def process_video_concatenate(media_urls, job_id, webhook_url=None, *,
                                  gpu_index: int = 0,
                                  vb='10M',       # target video bitrate
                                  ab='192k',      # target audio bitrate
                                  preset='p5',    # p1-p7 (p1 fastest, p7 best quality)
                                  cq=19):         # constant-quality target (lower = better)
    """
    Concatenate multiple videos using NVIDIA NVDEC/NVENC.

    Parameters
    ----------
    media_urls : list[dict]
        Each dict must contain a ``video_url`` key pointing to a downloadable video.
    job_id : str
        Used for temporary and output filenames.
    webhook_url : str | None
        Optional URL that can be POSTed when processing finishes.
    gpu_index : int
        GPU to use (important for multi-GPU systems).
    vb, ab, preset, cq
        NVENC tuning parameters for speed/quality trade-off.
    """
    input_files: list[str] = []
    output_filename = f"{job_id}.mp4"
    output_path = os.path.join(LOCAL_STORAGE_PATH, output_filename)

    try:
        # ------------------------------------------------------------------
        # 1) Download each source video locally
        # ------------------------------------------------------------------
        for i, item in enumerate(media_urls):
            url = item["video_url"]
            local_path = download_file(
                url,
                os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_in_{i}")
            )
            input_files.append(local_path)

        # ------------------------------------------------------------------
        # 2) Build a concat list for the FFmpeg concat demuxer
        # ------------------------------------------------------------------
        concat_list = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_files.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in input_files:
                f.write(f"file '{os.path.abspath(p)}'\n")

        # ------------------------------------------------------------------
        # 3) Run FFmpeg with NVDEC + NVENC, keeping frames on the GPU
        # ------------------------------------------------------------------
        (
            ffmpeg
            .input(
                concat_list,
                format="concat",
                safe=0,
                hwaccel="cuda",                 # NVDEC hardware decode
                hwaccel_device=gpu_index,
                hwaccel_output_format="cuda",   # keep frames in GPU memory
                **{"c:v": "h264_cuvid"}         # explicit CUDA decoder
            )
            .output(
                output_path,
                vcodec="h264_nvenc",            # NVENC encoder
                preset=preset,
                rc="vbr",                       # variable bit rate
                cq=cq,                          # constant-quality target
                video_bitrate=vb,
                maxrate=vb,
                bufsize=str(int(vb.rstrip("M")) * 2) + "M",
                acodec="aac",
                audio_bitrate=ab,
                movflags="+faststart",          # progressive playback hint
                gpu=gpu_index
            )
            .global_args("-y")                  # overwrite existing file
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # ------------------------------------------------------------------
        # 4) Cleanup temporary files
        # ------------------------------------------------------------------
        for p in input_files:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

        os.remove(concat_list)

        # Ensure output exists
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"Output file {output_path} was not created."
            )

        print(f"[GPU concat] Success → {output_path}")
        # If you want to notify a webhook, you could post here
        # e.g. requests.post(webhook_url, json={"status": "done", "file": output_path})

        return output_path

    # ----------------------------------------------------------------------
    # Error handling
    # ----------------------------------------------------------------------
    except ffmpeg.Error as ff:
        # Provide FFmpeg stderr for easier debugging
        stderr = ff.stderr.decode(errors="replace") if ff.stderr else ""
        print("[GPU concat] FFmpeg error:\n", stderr)
        raise
    except Exception as ex:
        print("[GPU concat] Unexpected error:", ex)
        raise
