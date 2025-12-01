# Refactored subtitle generator script

import logging
from whisper import load_model
import moviepy as mp
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def format_time(time_value):
    """
    Format time in seconds to HH:MM:SS format.

    Args:
        time_value (int): Time value in seconds.

    Returns:
        str: Formatted time string.
    """
    hours, remainder = divmod(abs(time_value), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def extract_audio(video_file):
    """
    Extract audio from video file using moviepy.

    Args:
        video_file (str): Path to the input video file.

    Returns:
        AudioClip: Extracted audio content.
    """
    try:
        video = mp.VideoFileClip(video_file)
        return video.audio
    except Exception as e:
        logging.error(f"Error extracting audio from {video_file}: {e}")
        raise

def transcribe_audio(audio_data, model):
    """
    Transcribe extracted audio using Whisper model.

    Args:
        audio_data (Audio): Extracted audio content.
        model (Model): Loaded Whisper model instance.

    Returns:
        str: Transcribed text as a string.
    """
    try:
        transcription = model.transcribe(audio_data)
        return transcription
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise

def generate_subtitles(video_file, model):
    """
    Generate subtitles for the given video file.

    Args:
        video_file (str): Path to the input video file.
        model (Model): Loaded Whisper model instance.

    Returns:
        str: Generated subtitle text as a string.
    """
    audio_data = extract_audio(video_file)
    transcription = transcribe_audio(audio_data, model)

    # Apply formatting to transcription
    formatted_transcription = " ".join(transcription.split())

    return formatted_transcription

def save_subtitles(subtitle_text, video_file):
    """
    Save generated subtitles to a file.

    Args:
        subtitle_text (str): Generated subtitle text.
        video_file (str): Path to the input video file.
    """
    filename = os.path.basename(video_file).split(".")[0] + ".srt"
    with open(filename, "w") as f:
        # Write subtitles to SRT format
        f.write(f"{os.path.getctime(video_file)}\n")
        f.write(subtitle_text)

if __name__ == "__main__":
    # model = load_model() # original
    model = load_model("turbo", device="cuda") # manual change
    video_files = ["videos/369k.mkv"]
    for video_file in video_files:
        subtitles = generate_subtitles(video_file, model)
        save_subtitles(subtitles, video_file)
