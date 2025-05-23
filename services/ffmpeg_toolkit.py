import os
import ffmpeg
import requests
from services.file_management import download_file

STORAGE_PATH = "/tmp/"

def process_conversion(media_url, job_id, bitrate='128k', webhook_url=None):
    """Convert media to MP3 format (using GPU acceleration where possible)."""
    input_filename = download_file(media_url, os.path.join(STORAGE_PATH, f"{job_id}_input"))
    output_filename = f"{job_id}.mp3"
    output_path = os.path.join(STORAGE_PATH, output_filename)

    try:
        # If input is a video file, use GPU decoding (e.g., h264_cuvid) to extract audio faster
        (
            ffmpeg
            .input(input_filename, hwaccel='cuda')
            .output(output_path, acodec='libmp3lame', audio_bitrate=bitrate)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        os.remove(input_filename)
        print(f"Conversion successful: {output_path} with bitrate {bitrate}")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file {output_path} does not exist after conversion.")

        return output_path

    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        raise


def process_video_combination(media_urls, job_id, webhook_url=None):
    """Combine multiple videos into one using GPU acceleration."""
    input_files = []
    output_filename = f"{job_id}.mp4"
    output_path = os.path.join(STORAGE_PATH, output_filename)

    try:
        for i, media_item in enumerate(media_urls):
            url = media_item['video_url']
            input_filename = download_file(url, os.path.join(STORAGE_PATH, f"{job_id}_input_{i}"))
            input_files.append(input_filename)

        concat_file_path = os.path.join(STORAGE_PATH, f"{job_id}_concat_list.txt")
        with open(concat_file_path, 'w') as concat_file:
            for input_file in input_files:
                concat_file.write(f"file '{os.path.abspath(input_file)}'\n")

        # Use GPU decoding (if input is video) and GPU encoding (h264_nvenc)
        (
            ffmpeg
            .input(concat_file_path, format='concat', safe=0)
            .output(output_path, vcodec='h264_nvenc', acodec='aac')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        for f in input_files:
            os.remove(f)
        os.remove(concat_file_path)

        print(f"Video combination successful: {output_path}")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file {output_path} does not exist after combination.")

        return output_path

    except Exception as e:
        print(f"Video combination failed: {str(e)}")
        raise
