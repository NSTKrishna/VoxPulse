"""
app.py

Modern, Demo-Ready Gradio Interface for VoxPulse
"""

import os
import json
import time
import traceback
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline modules
from modules.utils import convert_audio_to_wav
from modules.transcribe import AudioTranscriber
from modules.diarization import AudioDiarizer
from modules.alignment import TranscriptAligner
from modules.analyze import CallAnalyzer
from modules.report_generator import ReportGenerator

# Global instances (lazy loaded to save memory and startup time)
pipeline_instances = {}

def load_pipeline():
    """Lazy loads all heavy AI models to prevent crashing on app startup."""
    if not pipeline_instances:
        print("Initializing AI models... (this may take a minute)")
        # For local demos, "base" or "small" are recommended for whisper
        pipeline_instances['transcriber'] = AudioTranscriber(model_name="base")
        pipeline_instances['diarizer'] = AudioDiarizer()
        pipeline_instances['analyzer'] = CallAnalyzer()
        pipeline_instances['report_gen'] = ReportGenerator(output_dir="outputs")
        print("All models loaded successfully!")
    return pipeline_instances

def process_audio(audio_path, progress=gr.Progress()):
    """
    Main orchestration function. Connected directly to the Gradio button.
    progress parameter hooks into Gradio's live progress bar.
    """
    if not audio_path:
        raise gr.Error("Please upload an audio file before proceeding.")
        
    try:
        # 0. Load Models
        progress(0.05, desc="Loading AI Models (Whisper, Pyannote, Phi-3)...")
        models = load_pipeline()
        
        # 1. Audio Prep
        progress(0.1, desc="Preprocessing and normalizing audio...")
        wav_path = convert_audio_to_wav(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 2. Transcription
        progress(0.2, desc="Transcribing audio with Whisper...")
        whisper_segments = models['transcriber'].transcribe_audio(wav_path)
        if not whisper_segments:
            raise gr.Error("Transcription failed. Please check the logs.")
            
        # 3. Diarization
        progress(0.5, desc="Detecting speakers with Pyannote...")
        diarization_segments = models['diarizer'].diarize_audio(wav_path)
        if not diarization_segments:
            raise gr.Error("Diarization failed. Please check the logs.")
            
        # 4. Alignment
        progress(0.7, desc="Aligning speaker timelines with transcript...")
        aligned_data = TranscriptAligner.align_segments(whisper_segments, diarization_segments)
        conversation_text = TranscriptAligner.format_conversation(aligned_data)
        
        # 5. Analysis
        progress(0.8, desc="Analyzing interaction with Phi-3 Mini...")
        analysis_result = models['analyzer'].analyze_transcript(conversation_text)
        if not analysis_result:
            raise gr.Error("Analysis failed to return valid JSON.")
            
        # 6. Report Generation
        progress(0.9, desc="Exporting PDF and JSON reports...")
        paths = models['report_gen'].generate_all(base_name, analysis_result)
        
        # Cleanup temp wav file if one was created
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)
            
        progress(1.0, desc="Complete!")
        
        # Format outputs for the Gradio UI
        formatted_analysis = json.dumps(analysis_result, indent=4)
        downloadable_files = [paths.get('json'), paths.get('pdf')]
        
        return (
            conversation_text,         # Sent to the TextArea
            formatted_analysis,        # Sent to the Code block
            downloadable_files         # Sent to the File downloader
        )
        
    except Exception as e:
        traceback.print_exc()
        # Gradio gr.Error creates a nice red popup in the UI instead of crashing the app
        raise gr.Error(f"An unexpected error occurred: {str(e)}")

# =============================================================================
# GRADIO UI DEFINITION
# =============================================================================

# Define Custom CSS for a clean, professional look
custom_css = """
.gradio-container {
    font-family: 'Inter', system-ui, sans-serif;
}
.header-text {
    text-align: center;
    margin-bottom: 1.5rem;
}
"""

# Build the layout
with gr.Blocks(title="VoxPulse - Call Analysis", css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as app:
    
    # 1. Header Section
    with gr.Column(elem_classes="header-text"):
        gr.Markdown("# 🎙️ VoxPulse: AI Voice Call Analysis")
        gr.Markdown("Upload a customer support call to automatically transcribe, detect speakers, and generate an AI-driven quality assurance report.")
        
    # 2. Top Row (Inputs & File Outputs)
    with gr.Row():
        
        # Input Column
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Upload Call Recording")
            audio_input = gr.Audio(type="filepath", label="Input Audio (.mp3, .wav, .m4a)")
            analyze_btn = gr.Button("Analyze Call", variant="primary", size="lg")
            
        # File Download Column
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 2. Download Reports")
            report_files = gr.File(label="Generated Documents (PDF & JSON)", file_count="multiple", interactive=False)
            gr.Markdown("*Reports are automatically generated and stamped with metadata.*")
            
    # 3. Bottom Row (Text Visualizations)
    with gr.Row():
        
        # Transcript Viewer
        with gr.Column():
            gr.Markdown("### Diarized Transcript")
            transcript_output = gr.TextArea(
                label="Conversation",
                lines=18,
                show_copy_button=True,
                interactive=False
            )
            
        # Analysis Viewer
        with gr.Column():
            gr.Markdown("### AI Quality Assurance")
            analysis_output = gr.Code(
                label="Structured JSON Analysis",
                language="json",
                lines=18,
                interactive=False
            )
            
    # 4. Bind the Button to the Logic
    analyze_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcript_output, analysis_output, report_files]
    )

if __name__ == "__main__":
    # Launch the server (accessible at http://localhost:7860)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
