"""
diarization.py

Production-ready speaker diarization module using pyannote.audio.

Best Practices Included:
1. **Model Loading:** The pipeline is initialized once to save overhead.
2. **Authentication:** Securely loads the Hugging Face token from environment variables.
3. **Device Management:** Uses GPU acceleration (CUDA/MPS) if available.
4. **Error Handling:** Catches authentication issues, missing files, and pipeline errors.
5. **Logging:** Standardized logging for tracking and debugging.
"""

import os
import logging
import torch
from pyannote.audio import Pipeline
from typing import List, Dict, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioDiarizer:
    """
    A class to handle speaker diarization using the pyannote.audio pipeline.
    """
    def __init__(self, hf_token: str = None):
        """
        Initializes the diarizer and loads the pyannote pipeline into memory.
        
        Args:
            hf_token (str, optional): Hugging Face authentication token. 
                                      If None, attempts to read from the 'HF_TOKEN' environment variable.
        """
        self.token = hf_token or os.environ.get("HF_TOKEN")
        if not self.token:
            logger.error("Hugging Face token is missing. Diarization requires authentication.")
            raise ValueError(
                "HF_TOKEN is not set. Please provide it as an argument or set the HF_TOKEN environment variable. "
                "Ensure you have accepted the user conditions for pyannote/speaker-diarization-3.1 on Hugging Face."
            )
            
        # Determine appropriate device: CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Initializing AudioDiarizer on device '{self.device}'...")
        
        try:
            # Load the pre-trained diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=self.token
            )
            # Send pipeline to the selected device
            self.pipeline.to(self.device)
            logger.info("pyannote diarization pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load pyannote pipeline: {e}. Check your HF_TOKEN and internet connection.", exc_info=True)
            raise

    def diarize_audio(self, file_path: str, num_speakers: int = None, min_speakers: int = None, max_speakers: int = None) -> Union[List[Dict[str, Union[float, str]]], None]:
        """
        Diarizes the given audio file and returns timestamped speaker segments.
        
        Args:
            file_path (str): The path to the audio file (.wav format is highly recommended).
            num_speakers (int, optional): Exact number of speakers, if known.
            min_speakers (int, optional): Minimum number of speakers.
            max_speakers (int, optional): Maximum number of speakers.
            
        Returns:
            List[Dict]: A list of segments containing 'speaker', 'start', and 'end'.
            None: If diarization fails.
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return None
            
        logger.info(f"Starting diarization for file: {file_path}")
        
        try:
            # Perform diarization
            diarization = self.pipeline(
                file_path, 
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Format output strictly
            formatted_segments = []
            
            # itertracks yields: turn (segment with start/end), track, and speaker_label
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                formatted_segments.append({
                    "speaker": speaker, # typically formatted as "SPEAKER_00", "SPEAKER_01", etc.
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2)
                })
                
            logger.info(f"Successfully identified {len(formatted_segments)} speaker turns.")
            return formatted_segments
            
        except Exception as e:
            logger.error(f"An error occurred during diarization: {e}", exc_info=True)
            return None

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Sample Usage
    # ---------------------------------------------------------
    print("--- Diarizer Sample Usage ---")
    
    import json
    from dotenv import load_dotenv
    
    # Load environment variables (e.g., HF_TOKEN from .env file)
    load_dotenv()
    
    try:
        # 1. Initialize diarizer (requires HF_TOKEN in environment)
        diarizer = AudioDiarizer()
        
        # 2. Provide an audio file path
        # Pyannote works best with 16kHz mono .wav files.
        sample_file = "../sample_audio/test_recording.wav"
        
        if os.path.exists(sample_file):
            # 3. Perform diarization
            print(f"\nDiarizing {sample_file}...\n")
            segments = diarizer.diarize_audio(sample_file)
            
            # 4. Handle output
            if segments:
                print("Diarization Output:")
                print(json.dumps(segments, indent=2))
            else:
                print("Diarization failed or returned no segments.")
        else:
            print(f"\n[Note] Please place a test audio file at '{os.path.abspath(sample_file)}' to run the sample usage script.")
            
    except ValueError as ve:
        print(f"\nConfiguration Error: {ve}")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
