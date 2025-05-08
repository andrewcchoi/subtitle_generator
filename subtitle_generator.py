# %%
from re import DEBUG
from venv import logger
import whisper
import moviepy as mp

import logging

logging.addLevelName(__name__, DEBUG)


# %%

def generate_subtitles(file_name, language="en"):
    """Generates subtitles for a video using Whisper."""

    video_path = f"video/{file_name}.mkv"
    audio_path = f"audio/{file_name}.wav"
    output_srt_path = f"{file_name}.srt"

    # Load the Whisper model
    model = whisper.load_model("turbo", device="cuda")  # Or another model like "small", "medium", "large", "turbo"

    # Extract audio from the video
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

    # Transcribe the audio
    result = model.transcribe(audio_path, verbose=False, condition_on_previous_text=False, language=language)

    # Generate SRT file
    with open(output_srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]

            f.write(f"{i+1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")

def format_time(seconds):
    """Formats time in seconds to SRT format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(seconds*1000 % 1000):03}"


# %%
# %%timeit
if __name__ == "__main__":
    # vibing with gemini
    file_name = "Sherlock.and.Daughter.S01E03.720p.HEVC.x265-MeGusta"
    generate_subtitles(file_name, language="en")
