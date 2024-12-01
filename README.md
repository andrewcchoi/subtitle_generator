readme stuff

# prerequisites
1. ffmpeg
1. python
1. cuda (cuda capable gpu required)
1. pytorch (cuda capable gpu required)

# installation on windows
## ffmpeg
[geekforgeeks: How to Install FFmpeg on Windows?](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)
1. [download ffmpeg file](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z)
1. extract all
1. rename folder to "ffmpeg"
1. move file to root of c: drive
1. set environment variable path in cmd
    ```cmd
    setx /m PATH "C:\ffmpeg\bin;%PATH%"
    ```
1. restart machine
1. confirm installation
    ```cmd
    ffmpeg -version
    ```

## python
```python
py -3.10 -m venv venv
venv/scripts/activate
pip install -r requirements.txt
```

## cuda
[nvidia - CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

## pytorch
[pytorch- Getting Started Locally](https://pytorch.org/get-started/locally/#supported-windows-distributions)

# resources
- [openai whisper repo](https://github.com/openai/whisper)