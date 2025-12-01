import torch
from typing import Optional
import whisper
import moviepy as mp
import enum

class Devices(str, enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"

def initialize_model(model_name: str) -> whisper:
    device_type = Devices.CUDA if torch.cuda.is_available() else Devices.CPU
    return whisper.load_model(model_name, device=device_type.value)

EXTENSIONS = {
    "video": "mkv",
    "audio": "wav"
}

def generate_subtitles(
        file_name: str,
        video_ext: Optional[str] = EXTENSIONS["video"],
        language: Optional[str] = None
) -> None:
    """
    Generates subtitles for a video using Whisper.

    Args:
        file_name (str): File name of the video.
        video_ext (Optional[str]): Extension of the video file. Defaults to "mkv".
        language (Optional[str], optional): Language code. Defaults to None, in which case language autodetection is used.
    """
    audio_file = f"audio/{file_name}.{EXTENSIONS['audio']}"
    output_srt_path = f"{file_name}.srt"

    model = initialize_model("turbo")

    video_path = f"videos/{file_name}.{video_ext}"
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_file)

    transcription_result = model.transcribe(
        audio_file,
        verbose=False,
        condition_on_previous_text=False,
        language=language
    )

    with open(output_srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(transcription_result["segments"]):
            time_text = format_time(segment["start"]), format_time(segment["end"])
            text = segment["text"]
            f.write(f"{i+1}\n{time_text} --> {text}\n\n")

def format_time(time_in_seconds: float) -> str:
    """Formats time in seconds to SRT format."""
    mm, ss = divmod(int(time_in_seconds), 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02.2f}"
