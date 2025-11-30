
import ffmpeg


def convert_audio(file_name: str, input_ext: str, output_ext: str, audio_bitrate: str='320k') -> None:
    """
    Converts an audio file from one format to another.
    
    Args:
        filename (str): File name of audio file (e.g., 'audio file').
        input_file (str): The extension of the input audio file (e.g., 'm4b').
        output_file (str): The extension of the output audio file (e.g., 'mp3').
        audio_bitrate (str): Bitrate of the output audio file (e.g., '320k').

    Low Quality:
        96 kbps (kilobits per second)
        128 kbps

    Medium Quality:
        160 kbps
        192 kbps

    High Quality:
        256 kbps
        320 kbps (maximum for MP3)
    """

    input_file = f"audio/{file_name}.{input_ext}"
    output_file = f"audio/{file_name}.{output_ext}"

    (
        ffmpeg
        .input(input_file)
        .output(output_file, audio_bitrate=audio_bitrate)
        .run()
    )   

    print(f"Audio converted from {input_file.split('.')[-1]} to {output_file.split('.')[-1]} and saved as '{output_file}'")


# %%
if __name__ == "__main__":
    # Load the audio file
    file_name = "sample_audio"
    input_ext = "m4b"  # Replace with input extension
    output_ext = "mp3"  # Replace with desired output extension

    convert_audio(file_name=file_name, input_ext=input_ext, output_ext=output_ext)
