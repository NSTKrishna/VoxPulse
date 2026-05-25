---
title: voxpulse-call-qa
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
---

# 🎙️ VoxPulse: AI Voice Call Analysis System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)

**VoxPulse** is an advanced, fully open-source AI pipeline designed to automate customer support call quality assurance (QA). It transcribes audio, identifies distinct speakers, and utilizes a Small Language Model (SLM) to evaluate the interaction, generating professional Markdown and PDF reports.

![VoxPulse Demo Interface](docs/screenshot_placeholder.png)  
*Placeholder: A screenshot of the Gradio UI in action.*

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [The Models](#-the-models)
- [Installation](#-installation)
- [Usage & Demo Instructions](#-usage--demo-instructions)
- [Troubleshooting](#-troubleshooting)
- [Challenges & Limitations](#-challenges--limitations)
- [Future Improvements](#-future-improvements)

---

## 🚀 Project Overview

Traditionally, evaluating customer support calls requires humans to listen to hours of audio. VoxPulse automates this. By chaining together state-of-the-art open-source models, VoxPulse can:
1. Intake raw `.mp3`, `.wav`, or `.m4a` files.
2. Accurately transcribe the speech to text.
3. Map sentences to individual speakers (Agent vs. Customer).
4. Evaluate the agent's performance, customer sentiment, and adherence to compliance using **Microsoft Phi-3 Mini**.
5. Output structured JSON data, a formatted Markdown summary, and an enterprise-ready PDF report.

---

## 🏗️ Architecture

VoxPulse employs a modular, fail-forward architecture:

```text
VoxPulse/
├── app.py                      # Main Gradio interface & pipeline orchestrator
├── .env                        # Secret keys (Hugging Face)
├── requirements.txt            # System dependencies
├── outputs/                    # Exported JSON and PDF reports
├── sample_audio/               # Test audio directory
└── modules/                    
    ├── utils.py                # Audio preprocessing (pydub)
    ├── transcribe.py           # Whisper integration
    ├── diarization.py          # Pyannote speaker isolation
    ├── alignment.py            # Overlap math to merge text and speakers
    ├── analyze.py              # Phi-3 SLM logic & System Prompting
    ├── json_utils.py           # Regex-based JSON syntax repair
    └── report_generator.py     # Markdown/PDF export (WeasyPrint)
```

---

## 🧠 The Models

VoxPulse is completely free to run locally, relying entirely on open-source, weight-available models:

1. **Transcription (OpenAI Whisper):** We use the `base` or `small` variant of Whisper for rapid, robust speech-to-text decoding. It inherently handles multiple languages and thick accents.
2. **Diarization (Pyannote.audio 3.1):** Pyannote acts as the "ears" for speaker isolation. It maps acoustic embeddings to determine exactly *when* Speaker 0 stops talking and Speaker 1 begins.
3. **Evaluation (Microsoft Phi-3-Mini-4k-Instruct):** A highly capable 3.8B parameter Small Language Model. We chose Phi-3 over massive 70B parameter models because it excels at instruction-following (e.g., "Output ONLY JSON") while being small enough to run entirely on a consumer laptop or standard GPU.

---

## ⚙️ Installation

### 1. Prerequisites
- **Python 3.9+**
- **FFmpeg**: Required by `pydub` and `whisper` to decode audio.
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/) and add to PATH.
- **Hugging Face Account**: Required to download Pyannote. 
  - Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the terms.
  - Generate an Access Token in your HF Settings.

### 2. Setup the Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/VoxPulse.git
cd VoxPulse

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy the environment template and insert your Hugging Face token:
```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=your_actual_token_here
```

---

## 🎬 Usage & Demo Instructions

VoxPulse is optimized for live demonstrations. 

1. **Start the application:**
   ```bash
   python app.py
   ```
2. **Open the UI:** Navigate to `http://localhost:7860` in your web browser.
3. **Upload Audio:** Drag and drop a sample customer support call into the "Upload Call Recording" box.
4. **Run Analysis:** Click **Analyze Call**. 
   - *Note for live demos:* A detailed progress bar will appear, allowing the audience to see the exact step the AI is executing (e.g., Transcribing -> Diarizing -> Aligning).
5. **View Results:** The left panel will display the merged transcript, and the right panel will show the highlighted, structured JSON QA evaluation.
6. **Download:** Grab the generated PDF report from the top right panel.

![Report PDF Output Placeholder](docs/pdf_screenshot_placeholder.png)  
*Placeholder: Screenshot of the generated PDF report.*

---

## 🛠️ Troubleshooting

- **`ValueError: Could not download 'pyannote/speaker-diarization-3.1' model`**
  - Your `HF_TOKEN` is missing from the `.env` file, or you haven't clicked "Agree to terms" on the Pyannote Hugging Face page.
- **`JSONDecodeError` during Analysis**
  - The SLM hallucinated non-JSON text. The app uses a robust auto-repair script (`json_utils.py`), but if it fails repeatedly, try lowering the `temperature` in `analyze.py`.
- **`RuntimeError: CUDA out of memory`**
  - Whisper, Pyannote, and Phi-3 take up significant VRAM. If running on a GPU with <8GB VRAM, consider running the models on the CPU or using `bitsandbytes` to load Phi-3 in 4-bit precision.
- **PDF Generation Fails (`cairo` or `pango` missing)**
  - `WeasyPrint` requires specific C-libraries on your OS. If you are on Mac, run `brew install pango cairo`.

---

## 🚧 Challenges & Limitations

1. **Overlapping Speech (Cross-Talk):** Whisper natively outputs text in sentence blocks. If two speakers interrupt each other rapidly, the `alignment.py` module assigns the whole block to the speaker with the maximum overlap. This can lead to minor misattributions.
2. **Hardware Constraints:** Processing a 10-minute audio file requires heavy matrix multiplications. On a modern NVIDIA GPU, it takes ~30 seconds. On a CPU, it may take 5-10 minutes. 
3. **Hallucinations:** While mitigated by `json_utils.py`, language models occasionally struggle to output perfectly strictly typed data 100% of the time.

---

## 🔮 Future Improvements

- [ ] **Word-Level Timestamps:** Upgrade the Whisper pipeline to return `word_timestamps=True` for granular, exact speaker matching during cross-talk.
- [ ] **RAG Integration:** Allow the system to reference a company knowledge base (RAG) to determine if the agent provided *factually correct* information, rather than just polite information.
- [ ] **Batch Processing Mode:** Create a headless CLI script to process thousands of calls overnight without the Gradio UI.
- [ ] **Real-Time Streaming:** Migrate from batched transcription to a WebRTC streaming architecture for live, on-call agent coaching.
