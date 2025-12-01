# subtitle_generator.py

import logging
import whisper
import moviepy as mp
from re import DEBUG
from venv import logger

logging.basicConfig(level=logging.DEBUG)

def generate_subtitles(file_name, video_ext="mkv", language="en"):
    """Generates subtitles for a video using Whisper."""
    video_path = f"videos/{file_name}.{video_ext}"
    audio_path = f"audio/{file_name}.wav"
    output_srt_path = f"{file_name}.srt"

    try:
        # Load the Whisper model
        model = whisper.load_model("turbo", device="cuda")  # Or another model like "small", "medium", "large", "turbo"
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    try:
        # Extract audio from the video
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        logging.error(f"Failed to process video: {e}")
        return

    try:
        # Transcribe the audio
        result = model.transcribe(audio_path, verbose=False, condition_on_previous_text=False, language=language)
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {e}")
        return

    # Generate SRT file
    try:
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                start = format_time(segment["start"])
                end = format_time(segment["end"])
                text = segment["text"]

                f.write(f"{i+1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    except Exception as e:
        logging.error(f"Failed to write SRT file: {e}")

def format_time(seconds):
    """Formats time in seconds to SRT format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(round(seconds * 1000) % 1000):03}"

if __name__ == "__main__":
    # vibing with gemini
    file_name = "369k"
    video_ext = "mkv"
    generate_subtitles(file_name, video_ext, language="en")
