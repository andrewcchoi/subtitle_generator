import logging
import os
import argparse
from moviepy.editor import VideoFileClip
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)

def format_time(seconds):
    """
    Convert seconds to SRT-compatible time format (HH:MM:SS,mmm)
    
    Args:
        seconds: Float representing time duration in seconds
    
    Returns:
        String in HH:MM:SS,mmm format (e.g., "00:00:01,234")
    """
    ms = int(seconds * 1000)
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    ms_remaining = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms_remaining:03d}"

def generate_subtitles(video_path, output_srt_path):
    """
    Generate subtitles from a video file and save as SRT format
    
    Args:
        video_path: Path to input video file
        output_srt_path: Path to output SRT file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)
        
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        video.audio.write_audiofile(audio_path)
        video.close()

        # Load Whisper model with optimal settings
        model = whisper.load_model("base", device="cuda")
        
        # Transcribe audio to text
        result = model.transcribe(audio_path, language="en", temperature=0.0)
        
        # Write subtitles to SRT file
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                
                # Format time to SRT standard
                start_time = format_time(start)
                end_time = format_time(end)
                
                # Write subtitle block
                f.write(f"{i+1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        logging.info(f"Subtitles generated successfully at {output_srt_path}")
        return True

    except Exception as e:
        logging.error(f"Error generating subtitles: {str(e)}")
        return False

def main():
    """Entry point for command-line interface"""
    parser = argparse.ArgumentParser(
        description='Generate subtitles from video using Whisper model',
        epilog='Usage: python generate_subtitles.py [video_path] [output_srt_path]'
    )
    parser.add_argument('video_path', 
                        help='Path to the input video file (supports most common formats)')
    parser.add_argument('output_srt_path', 
                        help='Path to save the generated SRT file')
    args = parser.parse_args()

    generate_subtitles(args.video_path, args.output_srt_path)

if __name__ == "__main__":
    main()
