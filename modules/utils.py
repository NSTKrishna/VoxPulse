import os
import tempfile
import subprocess

def convert_audio_to_wav(input_path: str) -> str:
    """
    Converts audio files to 16kHz mono WAV using ffmpeg.
    Compatible with Whisper and pyannote.audio.
    """

    temp_dir = tempfile.gettempdir()
    base_name = os.path.basename(input_path)

    wav_filename = f"{os.path.splitext(base_name)[0]}_converted.wav"
    output_path = os.path.join(temp_dir, wav_filename)

    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to convert audio file: {str(e)}")

    return output_path