import logging
import subprocess
from typing import Optional
import whisper
import torch
from moviepy.editor import VideoFileClip

# ----------------------------
# Configuration and Setup
# ----------------------------

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------
# Helper Functions
# ----------------------------

def format_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(round(seconds)):02},{int(seconds * 1000 % 1000):03}"


def check_file_exists(path: str, description: str) -> None:
    """Check if file exists, raise error if not."""
    if not torch.path.exists(path):
        logging.error(f"{description} file not found: {path}")
        raise FileNotFoundError(f"{description} file not found: {path}")


# ----------------------------
# Main Function
# ----------------------------

def generate_subtitles(
    file_name: str,
    video_ext: str = "mkv",
    model_size: str = "turbo",
    device: str = "cuda",
    output_dir: str = "output"
) -> str:
    """
    Generate SRT subtitles from a video file.

    Args:
        file_name: Name of the video file (without extension).
        video_ext: Extension of the video file (e.g., 'mkv').
        model_size: Whisper model size (e.g., 'turbo').
        device: Computation device (e.g., 'cuda' or 'cpu').
        output_dir: Directory to save the output SRT file.

    Returns:
        Path to the generated SRT file.
    """
    try:
        # ----------------------------
        # 1. Define File Paths
        # ----------------------------
        video_path = f"{file_name}.{video_ext}"
        audio_path = f"{file_name}.wav"
        output_srt_path = f"{output_dir}/{file_name}.srt"

        # ----------------------------
        # 2. Check File Existence
        # ----------------------------
        check_file_exists(video_path, "Video")

        # ----------------------------
        # 3. Extract Audio with ffmpeg
        # ----------------------------
        logging.info("Extracting audio from video...")
        subprocess.run(
            ["ffmpeg", "-i", video_path, audio_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # ----------------------------
        # 4. Transcribe Audio with Whisper
        # ----------------------------
        logging.info("Transcribing audio...")
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(audio_path)

        # ----------------------------
        # 5. Format SRT Output
        # ----------------------------
        logging.info("Formatting subtitles...")
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], start=1):
                start_time = format_time(segment["start"])
                end_time = format_time(segment["end"])
                text = segment["text"].replace("\n", " ")
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        logging.info(f"Subtitles saved successfully: {output_srt_path}")
        return output_srt_path

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
