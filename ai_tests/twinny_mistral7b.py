import argparse
from re import DEBUG
from venv import logger
import torchaudio
import os
import sys
from whisper.utils.download import download_files_from_google_drive, setup_logger
from whisper.whisperpadder import WhisperPadder as WhisperTokenpadder
from whisper import WhisperAudioLoader, WhisperTokenizer, WhisperEncoderDecoder
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip


class WhisperTranscription:
    def __init__(self, model='small', device="cuda"):
        import torchaudio
        from whisper.utils.download import download_files_from_google_drive
        from whisper import WhisperAudioLoader, WhisperTokenizer, WhisperEncoderDecoder

        self.model = {
            'tiny': (WhisperTokenizer.get_tiny(), WhisperEncoderDecoder.get_tiny()),
            'base': (WhisperTokenizer.get_base(), WhisperEncoderDecoder.get_base()),
            'small': (WhisperTokenpadder.get('small'), WhisperEncoderDecoder.get('small')),
        }[model]

        self.device = torch.device(device)

    def transcode_wav_to_text(self, file):
        audio, _ = torchaudio.load(file, sample_rate=16000, device="cpu")
        tokenizer, encoder, decoder = self.model

        lengths = list(map(lambda x: x.size(0), audio))
        lengths += [0]  # empty input
        encodings = WhisperPadder(tokenizer, lengths).encoder_padded(audio)
        outputs = decoder(encodings[1:], encodings[1:], lengths[1:])[0]
        transcript = tokenizer.decode_batch(outputs)
        return transcript
``
    def transcribe(self, audio_path):
        from google.colab import files
        # Download the Whisper model if needed

        drive_id = '1ZXJp5LHnz9d3PVNb4xR2UfEGvu0D8cNk'
        file_name, extension = files.download(drive_id).split(".")
        downloaded_file = f"whisper_{file_name}.qdmg"

        if not audio_path.endswith(".wav"):
            convert_audio(audio_path, downloaded_file)

        transcription = self.transcode_wav_to_text(downloaded_file)
