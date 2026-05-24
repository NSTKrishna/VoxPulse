import whisper
import torch

class TranscriptionService:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribes the audio file using Whisper.
        Returns the transcription result which includes text and word-level segments if enabled.
        """
        print(f"Transcribing {audio_path}...")
        # You can add word_timestamps=True if you need word-level alignment with diarization
        result = self.model.transcribe(audio_path, word_timestamps=True)
        return result
