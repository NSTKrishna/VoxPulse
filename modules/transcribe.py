"""
transcribe.py

Production-ready transcription module using faster-whisper.
Optimized for high-speed, CPU-efficient, and low-memory inference using INT8 quantization.
"""

import os
import logging
from faster_whisper import WhisperModel
from typing import List, Dict, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    A robust class to handle audio transcription using faster-whisper.
    """
    def __init__(self, model_name: str = "tiny"):
        """
        Initializes the transcriber and loads the Whisper model.
        
        Args:
            model_name (str): The size of the whisper model to load ("tiny", "base", "small", "medium", "large").
                              Default is "tiny" for low-resource deployment.
        """
        self.model_name = model_name
        
        # Check for CUDA availability safely without crashing if PyTorch is not installed
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if has_cuda:
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            
        logger.info(f"Initializing AudioTranscriber with model '{model_name}' on device '{self.device}' with compute_type '{self.compute_type}'...")
        
        try:
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            logger.info("faster-whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}", exc_info=True)
            raise

    def transcribe_audio(self, file_path: str) -> Union[List[Dict[str, Union[float, str]]], None]:
        """
        Transcribes the given audio file using faster-whisper and returns timestamped segments.
        
        Args:
            file_path (str): The absolute or relative path to the audio file.
            
        Returns:
            List[Dict]: A list of segments containing 'start', 'end', and 'text'.
            None: If transcription fails.
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return None
            
        logger.info(f"Starting transcription with faster-whisper for file: {file_path}")
        
        try:
            # Run transcription using faster-whisper (generator-based)
            segments, info = self.model.transcribe(file_path, beam_size=5)
            
            # Format output strictly
            formatted_segments = []
            for segment in segments:
                formatted_segments.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                })
                
            logger.info(f"Successfully transcribed {len(formatted_segments)} segments.")
            return formatted_segments
            
        except Exception as e:
            logger.error(f"An error occurred during transcription: {e}", exc_info=True)
            return None

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Sample Usage
    # ---------------------------------------------------------
    print("--- Transcriber Sample Usage ---")
    
    import json
    
    # 1. Initialize transcriber
    transcriber = AudioTranscriber(model_name="tiny")
    
    # 2. Provide an audio file path
    sample_file = "../sample_audio/test_recording.mp3"
    
    if os.path.exists(sample_file):
        # 3. Perform transcription
        print(f"\nTranscribing {sample_file}...\n")
        segments = transcriber.transcribe_audio(sample_file)
        
        # 4. Handle output
        if segments:
            print("Transcription Output:")
            print(json.dumps(segments, indent=2))
        else:
            print("Transcription failed or returned no segments.")
    else:
        print(f"\n[Note] Please place a test audio file at '{os.path.abspath(sample_file)}' to run the sample usage script.")
