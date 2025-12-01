import datetime
from whisper import Whisper
# Function to format time for SRT file generation
def format_time(hours, minutes):
 seconds = (hours * 3600) + (minutes * 60)
 return str(int(seconds))[:-3] # Remove seconds from the output
# Main function to generate subtitles from a video file
def generate_subtitles():
 video = "myvideo.mp4"
 subtitles = "subtitles.srt"
 
 whisper = Whisper()
 whisper.open(video, format="mpegts") # Open video file in MPEG-T format
 frames = [frame for frame in whisper.frames]
 # Extract audio and video streams from the raw video data using FFmpeg
 video_stream, audio_stream = whisper.video + whisper.audio
 
 # Generate a transcript of the video's audio track using speech recognition
 transcript = transcribe(audio_stream)
 
 # Add timestamps to the transcript frames based on FFmpeg timestamp data
 timestamp = []
 for frame in frames:
 timestamp.append((datetime.datetime.utcfromtimestamp(frame["pts"]), datetime.datetime.utcfromtimestamp(frame["dts"])))
 filtered_frames = [(start, end) for (start, end) in sorted(timestamp if start < transcript[-1].end)]
 
 # Generate SRT file
 with open(subtitles, "w") as srt_file:
 for frame in frames:
 duration = (frame["duration"] / 90)[-2:] * 60
 time = format_time(*(fr[1] if fr and dt[-1].secs < 60 else format_time(int(dt['H']), int(dt['M']) - 3)) for fr, dt in zip(frame["pts"], frames * len([filtered_frames for _f, d in filtered_frames]
