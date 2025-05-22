# Copyright (c) 2025 Stephen G. Pope
# GPL-2.0-or-later
import os
import ffmpeg
import requests  # still here if you later add remote uploads
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
        Identifier used for temp/output file names.
    webhook_url : str | None
        Optional callback URL you can POST to when done.
    gpu_index : int
        Which GPU to use if you have more than one (default 0).
    vb, ab, preset, cq
        NVENC encoding tuneables.
    """
    input_files: list[str] = []
    output_filename = f"{job_id}.mp4"
    output_path = os.path.join(LOCAL_STORAGE_PATH, output_filename)

    try:
        # 1. Download source files
        for i, item in enumerate(media_urls):
            url = item["video_url"]
            local = download_file(url, os.path.join(LOCAL_STORAGE_PATH,
                                                    f"{job_id}_in_{i}"))
            input_files.append(local)

        # 2. Build concat list
        concat_list = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_files.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in input_files:
                f.write(f"file '{os.path.abspath(p)}'\n")

        # 3. Launch FFmpeg
        (
            ffmpeg
            .input(
                concat_list,
                format='concat',
                safe=0,
                hwaccel='cuda',                # NVDEC
                hwaccel_device=gpu_index,
                hwaccel_output_format='cuda',  # keep frames on GPU RAM
                c:v='h264_cuvid'               # explicit CUDA decoder
            )
            .output(
                output_path,
                vcodec='h264_nvenc',           # NVENC encoder
                preset=preset,
                rc='vbr',                      # variable-bit-rate (good balance)
                cq=cq,                         # quality target
                video_bitrate=vb,
                maxrate=vb,
                bufsize=str(int(vb.rstrip('M')) * 2) + 'M',
                acodec='aac',
                audio_bitrate=ab,
                movflags='+faststart',         # hint for progressive playback
                gpu=gpu_index
            )
            .global_args('-y')                 # overwrite without prompt
            .global_args('-hwaccel_output_format', 'cuda')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # 4. Clean-up
        for f in input_files:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        os.remove(concat_list)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output {output_path} missing after encode.")

        print(f"[GPU concat] Success → {output_path}")
        return output_path

    except ffmpeg.Error as ff:
        print("[GPU concat] FFmpeg error:", ff.stderr.decode())
        raise
    except Exception:
        raise
