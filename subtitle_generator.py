# %%
import whisper
import moviepy as mp


# %%

def generate_subtitles(video_path, audio_path, output_srt_path, language="en"):
    """Generates subtitles for a video using Whisper."""

    # Load the Whisper model
    model = whisper.load_model("turbo", device="cuda")  # Or another model like "small", "medium", "large", "turbo"

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

def format_time(seconds):
    """Formats time in seconds to SRT format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(seconds*1000 % 1000):03}"


# %%
# %%timeit
if __name__ == "__main__":
    audio_path = "videos/temp_audio_gemini.wav"
    file_name = "file_name"
    video_path = f"videos/{file_name}.mkv"
    output_srt_path = f"videos/{file_name}_gemini.srt"
    generate_subtitles(video_path, audio_path, output_srt_path, language="en")
