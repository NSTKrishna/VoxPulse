import os
import tempfile
from pydub import AudioSegment

def convert_audio_to_wav(input_path: str) -> str:
    """
    Converts .mp3, .m4a, and other formats to .wav.
    pyannote.audio and Whisper work best with 16kHz mono WAV files.
    """
    ext = os.path.splitext(input_path)[1].lower()
    
    try:
        if ext == ".wav":
            # It's already a wav, try standard loading
            audio = AudioSegment.from_wav(input_path)
        elif ext == ".mp3":
            audio = AudioSegment.from_mp3(input_path)
        elif ext in [".m4a", ".mp4"]:
            audio = AudioSegment.from_file(input_path, format="m4a")
        else:
            # Fallback to general file loading
            audio = AudioSegment.from_file(input_path)
    except Exception:
        try:
            # Robust fallback to auto-detection with ffmpeg
            audio = AudioSegment.from_file(input_path)
        except Exception as e:
            raise ValueError(f"Failed to decode audio file: {str(e)}. Please check if the file format is supported and ffmpeg is installed.")

    # Set frame rate to 16000 and channels to 1 (Mono)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Create a temporary file to store the converted wav
    temp_dir = tempfile.gettempdir()
    base_name = os.path.basename(input_path)
    wav_filename = f"{os.path.splitext(base_name)[0]}_converted.wav"
    output_path = os.path.join(temp_dir, wav_filename)
    
    audio.export(output_path, format="wav")
    return output_path
