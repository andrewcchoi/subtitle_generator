import logging
from pathlib import Path

import moviepy as mp
import torch

""" notes
did not include f-string and had a template to hard code variables.

1st run
Traceback (most recent call last):
  File "D:\!wip\subtitle_generator\twinny_codellama7binstruct.py", line 22, in <module>
    model = torch.load("turbo", map_location="cuda")  # Or another model like "small", "medium", "large", "turbo"
  File "D:\!wip\subtitle_generator\venv\lib\site-packages\torch\serialization.py", line 1319, in load
    with _open_file_like(f, "rb") as opened_file:
  File "D:\!wip\subtitle_generator\venv\lib\site-packages\torch\serialization.py", line 659, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "D:\!wip\subtitle_generator\venv\lib\site-packages\torch\serialization.py", line 640, in __init__
    super().__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'turbo'

"""

file_name = "samples/369k"
video_ext = "mkv"

# Define the paths for the input and output files
input_path = f"videos/{file_name}.{video_ext}"
output_path = f"{file_name}.srt"

# Load the Whisper model
model = torch.load("turbo", map_location="cuda")  # Or another model like "small", "medium", "large", "turbo"

def generate_subtitles(input_path: Path, output_path: Path, language: str) -> None:
    """Generates subtitles for a video using Whisper."""

    # Extract audio from the video
    video = mp.VideoFileClip(input_path)
    audio_path = input_path.with_suffix(".wav")
    video.audio.write_audiofile(audio_path) 

    # Transcribe the audio
    result = model.transcribe(audio_path, verbose=False, condition_on_previous_text=False, language=language)

    # Generate SRT file
    with output_path.open("w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]

            f.write(f"{i+1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")

def format_time(seconds: float) -> str:
    """Formats time in seconds to SRT format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(seconds*1000 % 1000):03}"
