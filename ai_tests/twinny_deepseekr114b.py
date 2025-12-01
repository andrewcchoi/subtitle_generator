import logging
from typing import Optional
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set CUDA device for compatibility

# Initial Logging Setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('subtitle_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model():
    """Load Whisper model with CUDA acceleration."""
    try:
        from transformers import WhisperModel, WhisperProcessor
        global whisper_model, whisper_processor
        whisper_model = WhisperModel.from_pretrained("openai/whisper-large").to('cuda')
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        logger.info("Whisper model loaded on CUDA device.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def generate_subtitles(audio_path: str, output_path: Optional[str] = None) -> str:
    """Generate subtitles from an audio file using Whisper."""
    try:
        if not audio_path.endswith(('.wav', '.mp3')):
            logger.error("Unsupported audio format. Only WAV and MP3 are supported.")
            raise ValueError("Unsupported audio format")
            
        if not output_path:
            output_path = os.path.splitext(audio_path)[0] + ".srt"
            
        # Perform transcription using Whisper
        import torch
        
        input_sentences = []
        sample_rate = 16000
        
        logger.info(f"Processing audio file: {audio_path}")
        
        features = whisper_processor(
            audio_path,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        input_features = features.input_features.to('cuda')
        with torch.no_grad():
            outputs = whisper_model(input_features)
            
        predicted_ids = torch.argmax(outputs.logits, dim=-1).tolist()
        
        # Convert to English sentences
        transcriptions = [
            whisper_processor.decode(features) for features in input_features.unbind(1)
        ]
        
        subtitles = format_time_and_transcripts(transcriptions)
        
        # Write to SRT file
        srt_path = os.path.join(output_path, "captions.srt")
        write_srt(subtitles, srt_path)
        
        logger.info(f"Subtitles generated successfully at {srt_path}")
        
    except Exception as e:
        logger.error(f"Error in generate_subtitles: {str(e)}")
        raise
        
    return output_path


def format_time_and_transcripts(transcripts):
    """Format timestamps and text for SRT file."""
    formatted = []
    
    current_time = 0
    for i, transcript in enumerate(transcripts):
        milliseconds = (i * 1000)
        start = f"{current_time:02d}:{current_time // 10 % 60:02d}:{milliseconds % 1000:03d}"
        end = f"{(current_time + 1):02d}:{(current_time + 1) // 10 % 60:02d}:{milliseconds % 1000:03d}"
        formatted.append(f"{i+1}\n{start} --> {end}\n{transcript.text}\n")
    
    return "".join(formatted)


def write_srt(content, file_path):
    """Write content to an SRT file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully wrote to {os.path.basename(file_path)}")
    except Exception as e:
        logger.error(f"Failed writing to {os.path.basename(file_path)}: {str(e)}")
        raise


def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate subtitles from an audio file.')
    parser.add_argument('audio', help='Audio file path')
    parser.add_argument('-o', '--output', 
                        help='Output directory for subtitles. (default: input file directory)')
    args = parser.parse_args()
    return args.audio, args.output


if __name__ == "__main__":
    """Main execution block with command-line arguments parsing."""
    audio_path, output_dir = parse_args()
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        exit(1)
        
    load_model()  # Load model once at start for better performance
    
    try:
        generate_subtitles(audio_path, output_dir)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logger.error(f"Error during subtitle generation: {str(e)}")
