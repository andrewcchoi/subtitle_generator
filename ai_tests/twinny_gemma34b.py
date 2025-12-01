import logging
import os
from venv import logger
import whisper
import moviepy.editor as mp

logging.addLevelName(__name__, logging.DEBUG)

def generate_subtitles(video_file_name, video_ext="mkv", language="en"):
    """Generates subtitles for a video using Whisper."""
    video_path = f"videos/{video_file_name}.{video_ext}"
    audio_path = f"audio/{video_file_name}.wav"
    output_srt_path = f"{video_file_name}.srt"

    try:
        # Load the Whisper model
        model = whisper.load_model("turbo", device="cuda")

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

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False # Indicate failure

    return True # Indicate success

def format_time(seconds):
    """Formats time in seconds to SRT format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(seconds*1000 % 1000):03}"

#Example Usage (Illustrative)
if __name__ == "__main__":
    video_file_name = input("Enter video filename: ")
    generate_subtitles(video_file_name)
