"""
transcribe.py

Production-ready transcription module using openai-whisper.

Best Practices Included:
1. **Model Loading:** The model is loaded once during class initialization to avoid overhead during repeated function calls.
2. **Device Management:** Automatically detects and utilizes GPU (CUDA or MPS) if available for faster local inference, falling back to CPU.
3. **Data Types:** Uses `fp16=True` when on a CUDA GPU to optimize memory and processing speed, and avoids fp16 warnings on CPU/MPS.
4. **Error Handling:** Graceful exception handling for missing files, unsupported formats, or internal library errors. Returns `None` on failure instead of crashing.
5. **Logging:** Standard Python logging is configured for tracking application flow and debugging effectively.
"""

import os
import logging
import torch
import whisper
from typing import List, Dict, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    A robust class to handle audio transcription using the OpenAI Whisper model.
    """
    def __init__(self, model_name: str = "base"):
        """
        Initializes the transcriber and loads the Whisper model into memory.
        
        Args:
            model_name (str): The size of the whisper model to load ("tiny", "base", "small", "medium", "large").
                              For local inference, "base" or "small" offer the best balance of speed and accuracy.
        """
        self.model_name = model_name
        
        # Determine appropriate device: CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        logger.info(f"Initializing AudioTranscriber with model '{model_name}' on device '{self.device}'...")
        
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

    def transcribe_audio(self, file_path: str) -> Union[List[Dict[str, Union[float, str]]], None]:
        """
        Transcribes the given audio file and returns timestamped segments.
        Supports .wav, .mp3, and .m4a out of the box via ffmpeg.
        
        Args:
            file_path (str): The absolute or relative path to the audio file.
            
        Returns:
            List[Dict]: A list of segments containing 'start', 'end', and 'text'.
            None: If transcription fails due to file absence or internal errors.
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return None
            
        supported_extensions = ['.wav', '.mp3', '.m4a']
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in supported_extensions:
            logger.warning(f"File extension '{ext}' is not explicitly listed as supported. Whisper will still attempt to decode.")
            
        logger.info(f"Starting transcription for file: {file_path}")
        
        try:
            # fp16 is natively supported on CUDA, otherwise default to fp32 to avoid warnings
            fp16_flag = True if self.device == "cuda" else False
            
            # Perform transcription
            result = self.model.transcribe(file_path, fp16=fp16_flag)
            
            # Format output strictly to the requested format
            formatted_segments = []
            for segment in result.get("segments", []):
                formatted_segments.append({
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "text": segment["text"].strip()
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
    
    # 1. Initialize transcriber (this downloads/loads the model only once)
    transcriber = AudioTranscriber(model_name="small")
    
    # 2. Provide an audio file path
    # Make sure you have a valid audio file here to test it.
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
