# Copyright (c) 2025 Stephen G. Pope … (GPL header unchanged)

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import ffmpeg                                # pip install ffmpeg-python
from services.file_management import download_file
from config import LOCAL_STORAGE_PATH

logger = logging.getLogger(__name__)

CUDA_GLOBAL_ARGS = [
    '-hwaccel', 'cuda',                      # let FFmpeg use the GPU if it can
    '-hwaccel_output_format', 'cuda',        # keep frames in GPU memory (N/A for audio, but harmless)
    # you can add “-threads 1” after exhaustive profiling, but FFmpeg generally
    # chooses well for single-stream audio tasks.
]

def _download(url_and_path):
    """Helper used by the thread pool."""
    url, tmp_path = url_and_path
    return download_file(url, tmp_path)

def process_audio_concatenate(media_urls, job_id, webhook_url=None):
    """
    Download a list of audio URLs and concatenate them into a single MP3.
    GPU acceleration is requested for *decoding* via CUDA; MP3 encoding still
    relies on libmp3lame (CPU-only in FFmpeg 7.x).
    """
    output_filename = f'{job_id}.mp3'
    output_path = os.path.join(LOCAL_STORAGE_PATH, output_filename)
    concat_file_path = os.path.join(LOCAL_STORAGE_PATH, f'{job_id}_concat_list.txt')
    tmp_template = os.path.join(LOCAL_STORAGE_PATH, f'{job_id}_input_{{:03d}}')

    input_files = []

    try:
        # --- 1.  Parallel download ---------------------------------------------------
        with ThreadPoolExecutor(max_workers=min(8, len(media_urls))) as pool:
            futures = {
                pool.submit(_download, (item['audio_url'], tmp_template.format(i)))
                : i for i, item in enumerate(media_urls)
            }
            for future in as_completed(futures):
                input_files.append(future.result())

        # --- 2.  Build concat list ---------------------------------------------------
        with open(concat_file_path, 'w', encoding='utf-8') as f:
            for fp in input_files:
                f.write(f"file '{os.path.abspath(fp)}'\n")

        # --- 3.  Run FFmpeg ----------------------------------------------------------
        (
            ffmpeg
            .input(concat_file_path, format='concat', safe=0)
            .output(
                output_path,
                acodec='libmp3lame', audio_bitrate='192k',
                # keep sample rate/channel-layout unchanged; change here if needed
            )
            .global_args(*CUDA_GLOBAL_ARGS, '-y', '-hide_banner')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # --- 4.  Verify and return ---------------------------------------------------
        if not os.path.exists(output_path):
            raise FileNotFoundError(f'FFmpeg finished but {output_path} is missing.')

        logger.info('Audio concatenation successful → %s', output_path)
        return output_path

    except Exception:
        logger.exception('Audio concatenation failed')
        raise

    finally:
        # --- 5.  Clean-up ------------------------------------------------------------
        for fp in input_files:
            try:
                os.remove(fp)
            except FileNotFoundError:
                pass

        try:
            os.remove(concat_file_path)
        except FileNotFoundError:
            pass
