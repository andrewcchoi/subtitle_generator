import os
import logging
from dotenv import load_dotenv
import argparse
import whisper
from moviepy import editor

# Load environment variables from .env file
load_dotenv()

def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def generate_subtitles(file_name, video_ext="mkv", language="en"):
    try:
        logger.info("Starting subtitle generation for file: %s.%s", file_name, video_ext)
        
        # Extract the audio from the video file using moviepy
        video = editor.VideoFileClip(f"{file_name}.{video_ext}")
        audio = video.audio
        temp_audio_file = "temp_audio.wav"
        audio.write_audiofile(temp_audio_file, codec='pcm_s16le')
        
        # Load the Whisper model based on the provided language
        whisper_model = os.getenv('WHISPER_MODEL', 'turbo')
        logger.info("Loading Whisper model: %s", whisper_model)
        model = whisper.load_model(whisper_model)
        
        # Transcribe the audio file using the Whisper model
        result = model.transcribe(temp_audio_file, fp16=False, task="translate" if language != "en" else "transcribe", verbose=True)
        
        # Write the transcription to an SRT file
        with open('output_subtitles.srt', 'w') as srt_file:
            for i, segment in enumerate(result['segments']):
                start = format_time(segment['start'])
                end = format_time(segment['end'])
                text = segment['text'].strip().replace('\n', ' ')  # Normalize the subtitle text
                srt_file.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")
        
        logger.info("Subtitle generation completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_name}.{video_ext}")
        raise SystemExit(e)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise SystemExit(e)
    finally:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)  # Clean up the temporary audio file

def parse_args():
    parser = argparse.ArgumentParser(description="Generate subtitles for a given video file.")
    parser.add_argument("file_name", type=str, help="The name of the video file without extension.")
    parser.add_argument("--video_ext", type=str, default="mkv", help="File extension of the video file (default is mkv).")
    parser.add_argument("--language", type=str, default="en", help="Language code for transcription (default is en).")
    return parser.parse_args()

def main():
    args = parse_args()
    global logger
    logger = setup_logging()
    
    generate_subtitles(args.file_name, args.video_ext, args.language)

if __name__ == "__main__":
    main()


"""note

  File "D:\!wip\subtitle_generator\twinny_deepseekcoderv216b.py", line 6, in <module>
    from moviepy import editor
ImportError: cannot import name 'editor' from 'moviepy' (D:\!wip\subtitle_generator\venv\lib\site-packages\moviepy\__init__.py)

"""