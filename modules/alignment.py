"""
alignment.py

Production-ready module to align Whisper transcripts with pyannote diarization.

Best Practices Included:
1. **Overlap Calculation:** Uses a robust mathematical overlap calculation to handle cross-talk.
2. **Speaker Mapping:** Allows dynamic mapping of generic speaker labels (e.g., SPEAKER_00) to actual roles (e.g., Agent/Customer).
3. **Text Formatting:** Automatically groups consecutive segments by the same speaker into cohesive paragraphs.
4. **File I/O:** Safely writes the aligned output to a text file.
"""

import os
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptAligner:
    """
    A class to align text segments from Whisper with speaker segments from pyannote.
    """
    
    @staticmethod
    def calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
        """
        Helper function to calculate the overlapping duration in seconds between two time segments.
        
        Args:
            start1, end1: Start and end times of the first segment.
            start2, end2: Start and end times of the second segment.
            
        Returns:
            float: Duration of overlap in seconds. Returns 0.0 if no overlap.
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0.0, overlap_end - overlap_start)

    @staticmethod
    def align_segments(whisper_segments: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
        """
        Assigns the correct speaker to each Whisper transcript segment based on maximum overlap.
        
        Args:
            whisper_segments: Output from AudioTranscriber.
            diarization_segments: Output from AudioDiarizer.
            
        Returns:
            List[Dict]: A list of transcript segments, each augmented with a 'speaker' key.
        """
        aligned_data = []
        
        for w_seg in whisper_segments:
            w_start = w_seg["start"]
            w_end = w_seg["end"]
            
            best_speaker = "Unknown"
            max_overlap = 0.0
            
            # Find the speaker with the maximum overlap for this text segment
            for d_seg in diarization_segments:
                overlap = TranscriptAligner.calculate_overlap(
                    w_start, w_end, d_seg["start"], d_seg["end"]
                )
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = d_seg["speaker"]
            
            aligned_data.append({
                "start": w_start,
                "end": w_end,
                "speaker": best_speaker,
                "text": w_seg["text"]
            })
            
        return aligned_data

    @staticmethod
    def format_conversation(aligned_segments: List[Dict], speaker_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Formats the aligned segments into a readable conversation format.
        Groups consecutive segments from the same speaker.
        
        Args:
            aligned_segments: Output from align_segments().
            speaker_mapping: Optional dict to map generic labels (e.g., {"SPEAKER_00": "Agent"}).
                             If None, it tries a default heuristic mapping.
                             
        Returns:
            str: The formatted conversation string.
        """
        if not aligned_segments:
            return ""
            
        # Default mapping heuristic if none provided (first speaker = Agent, second = Customer)
        if speaker_mapping is None:
            unique_speakers = []
            for seg in aligned_segments:
                if seg["speaker"] not in unique_speakers and seg["speaker"] != "Unknown":
                    unique_speakers.append(seg["speaker"])
                    
            speaker_mapping = {}
            if len(unique_speakers) >= 1:
                speaker_mapping[unique_speakers[0]] = "Agent"
            if len(unique_speakers) >= 2:
                speaker_mapping[unique_speakers[1]] = "Customer"
            
            # Map any additional speakers as Speaker 3, Speaker 4, etc.
            for i in range(2, len(unique_speakers)):
                speaker_mapping[unique_speakers[i]] = f"Speaker {i+1}"
                
        # Build the conversation string
        conversation_lines = []
        current_speaker = None
        current_text = []
        
        for seg in aligned_segments:
            raw_speaker = seg["speaker"]
            mapped_speaker = speaker_mapping.get(raw_speaker, raw_speaker)
            
            text = seg["text"].strip()
            if not text:
                continue
                
            # If the speaker changes, push the accumulated text to lines and reset
            if mapped_speaker != current_speaker:
                if current_speaker is not None:
                    # Combine the sentences for the previous speaker
                    conversation_lines.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = mapped_speaker
                current_text = [text]
            else:
                # Same speaker, keep appending to their current block
                current_text.append(text)
                
        # Append the very last block
        if current_speaker is not None:
            conversation_lines.append(f"{current_speaker}: {' '.join(current_text)}")
            
        return "\n".join(conversation_lines)

    @staticmethod
    def save_transcript(conversation_text: str, output_path: str = "transcript.txt"):
        """
        Saves the formatted conversation to a text file.
        
        Args:
            conversation_text (str): The formatted conversation string.
            output_path (str): The file path where it should be saved.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(conversation_text)
            logger.info(f"Transcript successfully saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save transcript to {output_path}: {e}")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Sample Usage
    # ---------------------------------------------------------
    print("--- Aligner Sample Usage ---")
    
    # Dummy data simulating Whisper output
    mock_whisper = [
        {"start": 0.0, "end": 2.5, "text": "Hello how may I help you?"},
        {"start": 2.8, "end": 5.0, "text": "I need information about my order."},
        {"start": 5.0, "end": 7.5, "text": "Sure, I can help with that."}
    ]
    
    # Dummy data simulating Pyannote output
    mock_diarization = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.6},
        {"speaker": "SPEAKER_01", "start": 2.7, "end": 5.1},
        {"speaker": "SPEAKER_00", "start": 4.9, "end": 7.8} # Slight overlap simulation
    ]
    
    # 1. Align the segments
    aligned_data = TranscriptAligner.align_segments(mock_whisper, mock_diarization)
    
    # 2. Format it into conversation
    # You can explicitly map speakers, or let it auto-map (SPEAKER_00 -> Agent)
    mapping = {"SPEAKER_00": "Agent", "SPEAKER_01": "Customer"}
    conversation = TranscriptAligner.format_conversation(aligned_data, speaker_mapping=mapping)
    
    print("\nFormatted Conversation:")
    print(conversation)
    
    # 3. Save to file
    TranscriptAligner.save_transcript(conversation, "../outputs/transcript.txt")
