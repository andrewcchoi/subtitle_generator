import logging
from typing import Optional, Tuple

# Import required modules
import whisper
from moviepy.video import VideoFileClip  # Updated import

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_subtitles(file_name: str, video_ext: str = "mkv", language: str = "en") -> Tuple[str, Optional[Exception]]:
    """Generates subtitles for a video using Whisper."""
    # Path definitions
    video_path = f"videos/{file_name}.{video_ext}"
    output_srt_path = f"{file_name}.srt"
    
    try:
        logger.info(f'Starting subtitle generation for {file_name}')
        
        # Load the Whisper model
        model = whisper.load_model("turbo", device="cuda")  # Or another model
        
        # Extract audio from video safely using context manager
        with VideoFileClip(video_path) as video:
            audio_file = video.audio
            audio_file.write_audiofile(f"{file_name}.wav")
 
        # Transcribe the audio
        logger.info('Starting transcription')
        result = model.transcribe(
            f"{file_name}.wav",
            verbose=False,
            condition_on_previous_text=False,
            language=language
        )
        
        # Generate SRT file with better formatting and error handling
        def format_time(seconds: float) -> str:
            """Convert seconds to HH:MM:SS,mmm format."""
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int((seconds * 1000) % 1000):03}"
        
        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
            logger.info('Writing subtitles to file')
            for i, segment in enumerate(result["segments"]):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                
                srt_file.write(f"{i+1}\n")
                srt_file.write(f"{format_time(start)} --> {format_time(end)}\n")
                srt_file.write(f"{text}\n\n")

        logger.info('Subtitle generation completed successfully')

    except Exception as e:
        logger.error(f'Subtitle generation failed: {str(e)}')
        return (str(e), None)
    
    finally:
        # Clean up audio file
        if os.path.exists(f"{file_name}.wav"):
            os.close(f"{file_name}.wav")
            os.remove(f"{file_name}.wav")

        logger.info('Audio file cleaned up')

    return ('', None)


if __name__ == "__main__":
    import os
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate subtitles for a video file')
    parser.add_argument('369k', type=str, help='Name of the video file without extension')
    parser.add_argument('--ext', type=str, default='mkv', 
                       choices=['avi', 'mkv', 'mp4', 'mov'],
                       help='Video file extension (default: mkv)')
    
    args = parser.parse_args()

    # Call generate_subtitles with the provided arguments
    error_message, _ = generate_subtitles(args.filename, video_ext=args.ext)
    
    if error_message:
        print(f'Error occurred: {error_message}')
        exit(1)
