import argparse
from pathlib import Path
import torch
import whisper
import subprocess

# Disabling future warnings - known bug with PyTorch and Whisper
torch.utils.backcompat.broadcast_warning.enabled = False
torch.utils.backcompat.keepdim_warning.enabled = False

def extract_audio(video_file: Path) -> Path:
    # Extract audio from the video file using ffmpeg and save it in .wav format
    audio_file = video_file.with_suffix('.wav')
    subprocess.call(['ffmpeg', '-y', '-i', str(video_file), str(audio_file)])
    return audio_file

def transcribe_audio(audio_file: Path) -> list[dict]:
    # Use Whisper for automatic speech recognition to convert audio to text
    model = whisper.load_model('base')
    result = model.transcribe(str(audio_file))
    return result['segments']

def format_time(seconds: float) -> str:
    # Convert seconds into hh:mm:ss,ms format for subtitles
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{int(hours):02}:{int(minutes):02}:{seconds:.2f}'.replace('00:', '')

def write_subtitles(segments: list[dict], subtitle_file: Path) -> None:
    # Write the text segments into a .srt file formatted for subtitles
    with open(subtitle_file, 'w') as f:
        for i, segment in enumerate(segments):
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            text = segment['text'].strip().replace('\n', ' ')
            f.write(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n")

def main() -> None:
    # Parse command-line arguments for input video file
    parser = argparse.ArgumentParser(description='Generate subtitles for a given video')
    parser.add_argument('video', type=Path, help='Path to the video file')
    args = parser.parse_args()

    # Extract audio and transcribe it into text segments
    segments = transcribe_audio(extract_audio(args.video))

    # Write subtitles into a .srt file in the same directory as input
    subtitle_file = args.video.with_stem(f'{args.video.stem}_subtitles').with_suffix('.srt')
    write_subtitles(segments, subtitle_file)

if __name__ == '__main__':
    main()

    """notes
    generated wav file and started making srt file and ran into an error.
    (venv) PS D:\!wip\subtitle_generator> py twinny_codestral22b.py "videos/samples/369k.mkv"
ffmpeg version 2024-11-28-git-bc991ca048-full_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers
  built with gcc 14.2.0 (Rev1, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-fontconfig --enable-iconv --enable-gnutls --enable-libxml2 --enable-gmp --enable-bzlib --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-sdl2 --enable-libaribb24 --enable-libaribcaption --enable-libdav1d --enable-libdavs2 --enable-libopenjpeg --enable-libquirc --enable-libuavs3d --enable-libxevd --enable-libzvbi --enable-libqrencode --enable-librav1e --enable-libsvtav1 --enable-libvvenc --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxeve --enable-libxvid --enable-libaom --enable-libjxl --enable-libvpx --enable-mediafoundation --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-libharfbuzz --enable-liblensfun --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-ffnvcodec --enable-libvpl --enable-nvdec --enable-nvenc --enable-vaapi --enable-libshaderc --enable-vulkan --enable-libplacebo --enable-opencl --enable-libcdio --enable-libgme --enable-libmodplug --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame --enable-libvo-amrwbenc --enable-libcodec2 --enable-libilbc --enable-libgsm --enable-liblc3 --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-libflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint
  libavutil      59. 47.101 / 59. 47.101
  libavcodec     61. 26.100 / 61. 26.100
  libavformat    61.  9.100 / 61.  9.100
  libavdevice    61.  4.100 / 61.  4.100
  libavfilter    10.  6.101 / 10.  6.101
  libswscale      8. 12.100 /  8. 12.100
  libswresample   5.  4.100 /  5.  4.100
  libpostproc    58.  4.100 / 58.  4.100
Input #0, matroska,webm, from 'videos\samples\369k.mkv':
  Metadata:
    title           : Ghosts.2021.S04E01
    COMMENT         : ELiTE-Fri-18-Oct-2024,09:18:01,1080p,21,faster,Y,9459766,1920,1080,1.78
    ENCODER         : Lavf61.1.100
  Duration: 00:21:05.34, start: -0.005000, bitrate: 2391 kb/s
  Stream #0:0: Video: hevc (Main 10), yuv420p10le(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 23.98 fps, 23.98 tbr, 1k tbn (default)
    Metadata:
      ENCODER         : Lavc61.3.100 libx265
      DURATION        : 00:21:05.306000000
  Stream #0:1(eng): Audio: ac3, 48000 Hz, 5.1(side), fltp, 448 kb/s (default)
    Metadata:
      ENCODER         : Lavc61.3.100 ac3
      DURATION        : 00:21:05.344000000
  Stream #0:2(eng): Subtitle: subrip (srt)
    Metadata:
      DURATION        : 00:20:49.813000000
Stream mapping:
  Stream #0:1 -> #0:0 (ac3 (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to 'videos\samples\369k.wav':
  Metadata:
    INAM            : Ghosts.2021.S04E01
    ICMT            : ELiTE-Fri-18-Oct-2024,09:18:01,1080p,21,faster,Y,9459766,1920,1080,1.78
    ISFT            : Lavf61.9.100
  Stream #0:0(eng): Audio: pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, 5.1(side), s16, 4608 kb/s (default)
    Metadata:
      encoder         : Lavc61.26.100 pcm_s16le
      DURATION        : 00:21:05.344000000
[out#0/wav @ 000001f97b54ee40] video:0KiB audio:711756KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.000029%
size=  711756KiB time=00:21:05.34 bitrate=4608.0kbits/s speed= 768x
D:\!wip\subtitle_generator\venv\lib\site-packages\whisper\__init__.py:150: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(fp, map_location=device)
Traceback (most recent call last):
  File "D:\!wip\subtitle_generator\twinny_codestral22b.py", line 52, in <module>
    main()
  File "D:\!wip\subtitle_generator\twinny_codestral22b.py", line 49, in main
    write_subtitles(segments, subtitle_file)
  File "D:\!wip\subtitle_generator\twinny_codestral22b.py", line 36, in write_subtitles
    f.write(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n")
  File "C:\Users\choia\AppData\Local\Programs\Python\Python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u50b3' in position 31: character maps to <undefined>

"""
