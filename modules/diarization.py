import os
import torch
from pyannote.audio import Pipeline

class DiarizationService:
    def __init__(self):
        # The pipeline requires a Hugging Face token to download the pyannote models.
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN environment variable not set. Please set it to use pyannote.audio.")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Loading pyannote.audio pipeline on {self.device}...")
        # Note: You must accept the user conditions on Hugging Face for the models used by pyannote
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.token)
        self.pipeline.to(self.device)
        
    def diarize(self, audio_path: str):
        """
        Diarizes the audio file.
        Returns a pyannote Annotation object containing speaker segments.
        """
        print(f"Diarizing {audio_path}...")
        diarization = self.pipeline(audio_path)
        return diarization
        
    def align_transcription(self, whisper_result, diarization) -> list:
        """
        Aligns Whisper transcription segments with pyannote diarization segments.
        """
        aligned_segments = []
        for segment in whisper_result.get("segments", []):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            
            # Find the most overlapping speaker from diarization
            best_speaker = "Unknown"
            max_overlap = 0.0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turn_start, turn_end = turn.start, turn.end
                overlap = max(0, min(end, turn_end) - max(start, turn_start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker
            
            aligned_segments.append({
                "start": start,
                "end": end,
                "speaker": best_speaker,
                "text": text
            })
            
        return aligned_segments
