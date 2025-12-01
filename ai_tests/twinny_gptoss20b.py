import logging
import pathlib
import tempfile
import torch
from typing import Optional

import moviepy.editor as mp
import whisper

# ---------- Configuration ----------
VIDEO_DIR = pathlib.Path("videos")
AUDIO_DIR = pathlib.Path("audio")
OUTPUT_DIR = pathlib.Path("subtitles")

for d in (VIDEO_DIR, AUDIO_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Logger Setup ----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------- Whisper Engine ----------
class WhisperEngine:
    def __init__(self, model_name: str = "turbo", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(self, audio_path: pathlib.Path, language: str = "en") -> dict:
        logger.debug(
            "Transcribing %s with Whisper %s on %s",
            audio_path,
            self.model_name,
            self.device,
        )
        return self.model.transcribe(str(audio_path), verbose=False, language=language)


# ---------- Time Formatting ----------


def format_time(seconds: float) -> str:
    hrs, remainder = divmod(seconds, 3600)
    mins, secs = divmod(remainder, 60)
    ms = int((secs - int(secs)) * 1000)
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02},{ms:03}"


# ---------- Subtitle Generation Core ----------


def generate_subtitles(
    file_name: str,
    video_ext: str = "mkv",
    language: str = "en",
    engine: Optional[WhisperEngine] = None,
) -> pathlib.Path:
    """Generates SRT subtitle file from a video."""
    engine = engine or WhisperEngine(model_name="turbo", device="cuda")

    video_path = VIDEO_DIR / f"{file_name}.{video_ext}"
    audio_path = AUDIO_DIR / f"{file_name}.wav"
    output_path = OUTPUT_DIR / f"{file_name}.srt"

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Extract audio once to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        logger.info("Extracting audio to %s", tmp.name)
        clip = mp.VideoFileClip(str(video_path))
        clip.audio.write_audiofile(tmp.name)
        audio_path = pathlib.Path(tmp.name)

    # Transcribe
    transcript = engine.transcribe(audio_path, language=language)

    # Write SRT
    with output_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(transcript["segments"], 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")

    # Clean up temp audio file
    audio_path.unlink(missing_ok=True)
    logger.info("Finished %s, output -> %s", file_name, output_path)

    return output_path


# ---------- CLI Wrapper ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate subtitles from a video."
    )
    parser.add_argument("file_name", help="Base file name (without extension)")
    parser.add_argument(
        "--ext", default="mkv", help="Video extension"
    )  # Added default value
    parser.add_argument(
        "-l",
        "--lang",
        default="en",
        help="Subtitle language",
    )  # Improved description
    parser.add_argument(
        "--model",
        default="turbo",
        help="Whisper model to use",
    )  # Added default value
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Compute device",
    )
    args = parser.parse_args()

    engine = WhisperEngine(model_name=args.model, device=args.device)
    output = generate_subtitles(
        args.file_name, video_ext=args.ext, language=args.lang, engine=engine
    )
    print(f"Subtitle file created: {output}")
