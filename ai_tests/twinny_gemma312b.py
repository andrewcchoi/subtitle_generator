from whisper import load_model
import moviepy as mp
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_subtitles(file_name, video_ext="mkv", language="en", whisper_model_size="small", output_srt_path=None):
    """Generates subtitles for a video using Whisper."""
    try:
        video_path = os.path.join("videos", f"{file_name}.{video_ext}")
        audio_path = os.path.join("audio", f"{file_name}.wav")

        if output_srt_path is None:
            output_srt_path = f"{file_name}.srt"

        # Load the Whisper model
        try:
            model = load_model(whisper_model_size, device="cuda")
            logging.info(f"Loaded Whisper model: {whisper_model_size}")
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            return

        # Extract audio from the video
        try:
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)
            logging.info(f"Extracted audio to: {audio_path}")
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            return

        # Transcribe the audio
        try:
            result = model.transcribe(audio_path, verbose=False, condition_on_previous_text=False, language=language)
            logging.info("Audio transcription complete.")
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return

        # Generate SRT file
        try:
            with open(output_srt_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"]):
                    start = segment["start"]
                    end = segment["end"]
                    text = segment["text"]

                    f.write(f"{i+1}\n")
                    f.write(f"{format_time(start)} --> {format_time(end)}\n")
                    f.write(f"{text}\n\n")
            logging.info(f"Subtitles saved to: {output_srt_path}")
        except Exception as e:
            logging.error(f"Error writing SRT file: {e}")
            return

    except FileNotFoundError:
        logging.error(f"Video file not found: {video_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def format_time(seconds):
    """Formats time in seconds to SRT format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(seconds*1000 % 1000):03}"


if __name__ == "__main__":
    """
    Usage example:
    python subtitle_generator.py --file_name video_file --video_ext mkv --language en --whisper_model_size small
    """

    import argparse
    parser = argparse.ArgumentParser(description="Generate subtitles for a video using Whisper.")
    parser.add_argument("--file_name", default="369k", help="Name of the video file (without extension)")
    parser.add_argument("--video_ext", default="mkv", help="Video file extension (e.g., mkv, mp4)")
    parser.add_argument("--language", default="en", help="Language of the video")
    parser.add_argument("--whisper_model_size", default="turbo", help="Whisper model size (small, medium, large, turbo)")

    args = parser.parse_args()

    print("Loading Whisper...")
    generate_subtitles(args.file_name, args.video_ext, args.language, args.whisper_model_size)
