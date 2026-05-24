# VoxPulse: AI-Powered Voice Call Analysis System

VoxPulse is a complete, production-ready Python project that processes audio call recordings and generates structured feedback using entirely open-source models.

## Architecture

This project leverages the following technologies:
- **Audio Processing:** `pydub` (supports .mp3, .wav, .m4a)
- **Transcription:** `openai-whisper` (Base model by default, easily configurable)
- **Speaker Diarization:** `pyannote.audio` (identifies "who spoke when")
- **Analysis / LLM:** `Phi-3 Mini` via Hugging Face `transformers` (Small Language Model for local inference)
- **UI:** `gradio` (for optional web interface)

## Project Structure

```text
VoxPulse/
├── app.py                      # Gradio UI and main application entrypoint
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .env.example                # Example environment variables
├── modules/                    # Core modules
│   ├── __init__.py
│   ├── analysis.py             # SLM (Phi-3 Mini) integration
│   ├── diarization.py          # pyannote.audio integration
│   ├── report_generator.py     # Markdown and JSON export utilities
│   ├── transcription.py        # Whisper integration
│   └── utils.py                # Audio conversion and helpers
├── outputs/                    # Directory for generated JSON/Markdown reports
└── sample_audio/               # Directory to place test audio files
```

## Environment Setup Instructions

### 1. Prerequisites
- **Python 3.9+**
- **FFmpeg**: Required for audio processing.
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
- **Hugging Face Account**: You need an access token and must accept the user agreements for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face.

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your Hugging Face token.

### 4. Running the Application

To launch the Gradio web interface, run:

```bash
python app.py
```

The application will start a local server (typically at `http://localhost:7860`). Open this URL in your browser to upload an audio file and view the results.

The generated JSON and Markdown reports will be automatically saved in the `outputs/` directory.
