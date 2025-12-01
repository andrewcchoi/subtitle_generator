import os
import glob
import torch
import time
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from subtitle_generator import generate_subtitles
from transform import build_diarization_model
torch.backends.cudnn.benchmark = True  # set this line to avoid "Cuda out of memory" error

# Define paths for video and audio files, output folder path
video_path = "path/to/video.mp4"
audio_path = "path/to/audio.wav"
output_folder = "path/to/subtitles"

# Make sure the output folder exists, otherwise create it
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Check if captions are already included in the video file
# If so, skip generating subtitles
try:
    with open(video_path, "rb") as f:  # open video file in binary mode
        meta = f.read(4)  # read metadata from beginning of video file
        if b"mvhd" in meta and b"tfcd" in meta:
            print("Captions already included in the video file, skipping subtitles")
            return
except IOError:
    pass

# Create subtitles for a video using speech to text
start_time = time.time()  # measure processing time
generate_subtitles(
    subtitle_file_path=f"{output_folder}/subtitles.srt",
    audio_file_path=audio_path,
    output_dir=output_folder,
