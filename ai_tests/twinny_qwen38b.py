import logging
import whisper
import moviepy as mp
import torch
from datetime import timedelta
import os

os.makedirs("videos", exist_ok=True)
os.makedirs("audio", exist_ok=True)

file_name = "369k"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("turbo", device=device)

def format_time(seconds):
    """Format time in SRT format."""
    return f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{int(seconds * 1000 % 1000):03}"

# checkpoint = torch.load(fp, map_location=device, weights_only=True)

def generate_subtitles(file_name, video_ext="mkv", language="en"):
    try:
        video_path = f"videos/{file_name}.{video_ext}"
        audio_path = f"audio/{file_name}.wav"
        output_srt_path = f"{file_name}.srt"

        # Ensure directories exist
        os.makedirs("videos", exist_ok=True)
        os.makedirs("audio", exist_ok=True)

        # Load Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("turbo", device=device)

        # Extract audio
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)

        # Transcribe audio
        result = model.transcribe(audio_path, language=language)

        # Generate SRT
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                f.write(f"{i+1}\n")
                f.write(f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n")
                f.write(f"{segment['text']}\n\n")

    except Exception as e:
        logging.error(f"Error generating subtitles: {e}")
