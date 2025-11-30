# %%
from re import DEBUG
from venv import logger
import whisper
import moviepy as mp

import logging

logging.addLevelName(__name__, DEBUG)


# %%

def generate_subtitles(file_name, video_ext="mkv", language="en"):
    """Generates subtitles for a video using Whisper."""

    video_path = f"videos/{file_name}.{video_ext}"
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
    """
    \lib\site-packages\whisper\__init__.py:150: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    checkpoint = torch.load(fp, map_location=device)
    """
    # vibing with gemini
    file_name = "video_file"
    video_ext = "mkv"
    generate_subtitles(file_name, video_ext, language="en")
