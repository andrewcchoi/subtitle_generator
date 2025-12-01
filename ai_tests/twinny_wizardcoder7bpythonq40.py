import time
from typing import List, Tuple
import os
import math
import re
import shutil
import subprocess as sp
from pathlib import Path
import argparse
import logging

import torch

from tqdm.auto import tqdm

try:
    from utils.audio_preprocessing import preprocess_audio
except ModuleNotFoundError: 
    from .utils.audio_preprocessing import preprocess_audio

def format_time(seconds):
    return datetime.fromtimestamp(seconds).strftime("%H:%M:%S.%f")[:-3]

def generate_subtitles(audio_path: str, srt_output_path: str = None, model_type="large", language="en",
                    video_ext: str = "mp4", force_reprocess=False) -> str:
    """Generate subtitles using Whisper and save them as SRT files.
        
    Params:
    - audio_path (str): path to the audio file to generate subtitles for
    - srt_output_path (optional, str): path where to save the subtitles; defaults to `<audio_name>.srt` in same directory as input
    
    Returns:
    - subtitles_path (str): full path to generated SRT file
    """
    start = time.time()
    model, device = load_whisper_model(model_type)
    audio_path = Path(audio_path).resolve()    
    output_dir = audio_path.parent
    
    if not srt_output_path:
        audio_name = audio_path.stem + "." + video_ext  # type: ignore
        subtitles_path = (audio_path.parent / f"{audio_name}.srt").resolve()
    else:
        srt_output_path = Path(srt_output_path).resolve() 
        subtitles_path = str(srt_output_path)
        audio_name = srt_output_path.stem + video_ext  # type: ignore   
    
    caption_file = (output_dir / f"{audio_name}.txt")
    os.makedirs(str(caption_file.parent), exist_ok=True)
        
    if not force_reprocess and caption_file.is_file():
        subs = preprocess_text(caption_file.read_text())
        return subtitles_path
    else:
        audio_fpath = str(audio_path)
        tmp_wav = f"{str(output_dir / 'audio.wav')}"
        preprocess_audio(audio_fpath, output=tmp_wav)
        
        with torch.no_grad():
            subprocess_args = [
                "python", "-m",
                "whispertrain.run_model",
                "--load-path", str(Path(__file__).resolve().parent.parent / f"models/{model_type}/checkpoints/final_weights.pt"),  # type: ignore
                "--output-dir", output_dir,
                "--test",
                tmp_wav
            ]
            sp.run(subprocess_args, check=True) 
            
        subs = extract_captions("translations_large.txt") # type: ignore
    if not subs:  # something went wrong with generating captions, return empty string    
        logging.error("Failed to generate subtitles for %s", audio_path)   
        return ""
    
    with open(subtitles_path, "w") as srt_file:
        srt_file.write(_generate_srt(audio_path.stem, subs))  # type: ignore
        
    elapsed = time.time() - start
    print(f"Subtitles generated in {elapsed:.2f} seconds")
    return subtitles_path
