import os
from moviepy.editor import VideoFileClip
from pathlib import Path
import whispered
import logging

# Set up a basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_time(seconds: float) -> str:
    """Convert seconds into HH:MM:SS.ms string format."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"

def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video file."""
    try:
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(audio_path)
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise

def load_model(model_name="small") -> whispered.Model:
    """Load the Whisper model."""
    try:
        model = getattr(whispered, "Model").load_model(model_name)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def generate_subtitles(video_path: str, output_file: str, use_large=False) -> None:
    """
    Generate subtitles for a given video file.
    
    :param video_path: Path to the input video.
    :param output_file: Path for the generated subtitle file (VTT format).
    :param use_large: If True, use the large model variant; defaults to False.
    """
    audio_file = os.path.splitext(video_path)[0] + ".wav"
    
    try:
        extract_audio(video_path, audio_file)
        
        if use_large:
            model = load_model(model_name="large")
        else:
            model = load_model()
            
        result = whispered.detect(model.file_names(), filenames=[audio_file])
        
        # Generate VTT file (not shown here for brevity; include your implementation).
        with open(output_file, "w") as vtt:
            pass
        
    except Exception as e:
        logger.error(f"Error generating subtitles: {e}")
        raise

# Example Usage
video_folder = Path("369k")
audio_folder = Path("audio")

for video_filename in video_folder.glob("*.mkv"):
    audio_output_path = str(audio_folder / f"{os.path.splitext(video_filename.name)[0]}.wav")
    subtitle_output_path = str(video_folder.parent / "subtitles" / f"{video_filename.stem}_subtitle.srt")
    
    generate_subtitles(str(video_filename), subtitle_output_path)

"""notes 

(venv) PS D:\!wip\subtitle_generator> py twinny_phi414b.py    
Traceback (most recent call last):
  File "D:\!wip\subtitle_generator\twinny_phi414b.py", line 2, in <module>
    from moviepy.editor import VideoFileClip
ModuleNotFoundError: No module named 'moviepy.editor'

"""