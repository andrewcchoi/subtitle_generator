import logging
import os
from typing import List, Dict
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip
import logging.config

# Set up logging configuration
logging.config.fileConfig('logger.conf')
logging.basicConfig(level=logging.INFO)

def load_audio(video_path: str) -> str:
    """
    Loads audio from a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        str: The path to the loaded audio file.
    """
    try:
        video = VideoFileClip(video_path)
        audio_path = f"audio/{video_path.split('/')[-1]}.wav"
        video.audio.write_audiofile(audio_path)
        return audio_path
    except Exception as e:
        logging.error(f"Error loading audio: {e}")
        raise

def transcribe_audio(audio_path: str, language: str = "en") -> Dict:
    """
    Transcribes an audio file using Whisper.

    Args:
        audio_path (str): The path to the audio file.
        language (str, optional): The target language. Defaults to "en".

    Returns:
        Dict: The transcribed text segments.
    """
    try:
        model = whisper.load_model("turbo", device="cuda")
        result = model.transcribe(audio_path, verbose=False, condition_on_previous_text=False, language=language)
        return result
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise

def generate_srt(output_srt_path: str, segments: List[Dict]) -> None:
    """
    Generates an SRT file from transcribed text segments.

    Args:
        output_srt_path (str): The path to the output SRT file.
        segments (List[Dict]): The transcribed text segments.
    """
    with open(output_srt_path, "w") as f:
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]

            f.write(f"1\n")
            # Convert timestamp to HH:MM:SS format
            f.write("{} --> {}\n\n".format(format_time(start), format_time(end)))
            # Ensure newline after each line
            if text.strip() != "":
                f.write(text + "\n")

def generate_subtitles(file_name: str, video_ext: str = "mkv", language: str = "en") -> None:
    """
    Generates subtitles for a video using Whisper.

    Args:
        file_name (str): The name of the video file.
        video_ext (str, optional): The video extension. Defaults to "mkv".
        language (str, optional): The target language. Defaults to "en".

    Returns:
        None
    """
    audio_path = load_audio(f"videos/{file_name}.{video_ext}")
    try:
        result = transcribe_audio(audio_path, language)
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise

    output_srt_path = f"{file_name}.srt"
    segments = result["segments"]
    generate_srt(output_srt_path, segments)


def format_time(timestamp: str) -> str:
    """
    Format timestamp to HH:MM:SS.

    Args:
        timestamp (str): The timestamp in seconds since Unix Epoch.

    Returns:
        str: The formatted timestamp.
    """
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) / 60)
    seconds = timestamp % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    generate_subtitles("369k")
"""notes
(venv) PS D:\!wip\subtitle_generator> py twinny_llama323b.py
Traceback (most recent call last):
  File "D:\!wip\subtitle_generator\twinny_llama323b.py", line 5, in <module>
    from moviepy.editor import VideoFileClip, AudioFileClip
ModuleNotFoundError: No module named 'moviepy.editor'
"""