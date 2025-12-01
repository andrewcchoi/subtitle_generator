# subtitle_generator.py
from moviepy import VideoFileClip, AudioFileClip
import whisper
from typing import List
import os


def extract_audio(video_path: str) -> None:
    """Extracts audio from a video file and saves it as an WAV format."""
    try:
        # Load the video clip
        video = VideoFileClip(video_path)
        
        # Convert to mono channel audio
        audio = video.audio.to_soundarray(fps=160000, nbytes=2, buffersize=50000)
          
        audio = AudioFileClip(audio, fps=160000, format="wav")
            
        # Save the audio as a WAV file in specified location
        video_path_split = os.path.splitext(video_path)
        audio_path = "audio/" + video_path_split[0] + ".wav"
        audio.write_audiofile(audio_path)
    
    except Exception as e:
        print("Error extracting audio.", e)  # log error message here
    # consider using a context manager instead of directly calling write_audiofile() for better resource management and exception handling
    
def load_and_transcribe(audio_path: str, model: whisper.utils.model.TranscribentModel, language="en") -> List[str]:
    """Loads the Whisper model and transcribes audio file to text segments."""
    try:
        # Transcribe the audio using the model and return a list of strings corresponding to segments of subtitles.
        result = model.transcribe(audio_path, verbose=False, condition_on_previous_text=False, language="en")
        return [segment["text"] for segment in result["segments"]]   
    except Exception as e:
        print("Error transcribing audio.", e)  # log error message here

def save_srt(subtitles: List[str], output_path: str) -> None:
    """Writes a list of subtitles to an SRT file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f: 
            for i, text in enumerate(subtitles):
                start = i * 10.0 # Hardcoded - adjust this value depending on the segment duration.
                end = (i+1) * 10.0 # Same as above. This is just an example.
                subtitle_block = f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
                f.write(subtitle_block)
    except Exception as e:
        print("Error saving SRT file.") # log error message here. Alternatively, you can keep a record of errors in an error log.
        
def format_time(seconds: float) -> str:
    """Formats time in seconds to SRT format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:.2f},{int((seconds*100) % 100)}"

def gen_subtitles(file_name: str, video_ext: str="mkv", language="en") -> None:
    """Generates subtitles for a given video file."""
    # Load the Whisper model from `turbo` and set the device to GPU by default.
    model = whisper.load_model("turbo", device="cuda") # Or another model like "small", "medium", "large", "turbo"
    
    # Extract audio from video file
    video_path = f"videos/{file_name}.{video_ext}"
    extract_audio(video_path)
        
    # Get the appropriate WAV path
    audio_path = os.path.join("audio", f"{os.path.splitext(file_name)[0]}.wav")  # This assumes that you extracted audio into "audio" directory.
     
    # Transcribe audio and save as SRT subtitles
    srt_path = file_name + ".srt"
    segments = load_and_transcribe(audio_path, model, language)
    save_srt(segments, srt_path)  
