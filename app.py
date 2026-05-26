"""
app.py

Modern, Demo-Ready Gradio Interface for VoxPulse (Optimized for Render Free Tier)
"""

import os
import json
import time
import traceback

# -----------------------------------------------------------------------------
# Gradio, Pydantic, & Starlette Python 3.14 Compatibility Monkeypatches
# Prevents:
# - TypeError: argument of type 'bool' is not a container or iterable in get_type
# - AttributeError: 'bool' object has no attribute 'get' in _json_schema_to_python_type
# - Starlette >= 0.28 TemplateResponse signature compatibility gap with older Gradio
# -----------------------------------------------------------------------------
try:
    import gradio_client.utils as gradio_client_utils
    original_get_type = gradio_client_utils.get_type
    original_json_schema = gradio_client_utils._json_schema_to_python_type
    
    def patched_get_type(schema):
        if isinstance(schema, bool):
            return {}
        return original_get_type(schema)
        
    def patched_json_schema(schema, defs):
        if isinstance(schema, bool):
            schema = {}
        return original_json_schema(schema, defs)
        
    gradio_client_utils.get_type = patched_get_type
    gradio_client_utils._json_schema_to_python_type = patched_json_schema
except Exception as e:
    print(f"Gradio Client compatibility patch not applied: {e}")

try:
    from starlette.templating import Jinja2Templates
    original_template_response = Jinja2Templates.TemplateResponse
    
    def patched_template_response(self, *args, **kwargs):
        # Starlette >= 0.28 signature: (self, request, name, context=None, ...)
        # Gradio <= 4.44 signature: (self, name, context=None, ...)
        # If the first argument is a string, it is the template name (old signature).
        if len(args) > 0 and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.get("context", {})
            request = context.get("request") if isinstance(context, dict) else None
            
            # Reconstruct arguments for the new Starlette signature
            new_args = (request, name, context) + args[2:]
            return original_template_response(self, *new_args, **kwargs)
            
        return original_template_response(self, *args, **kwargs)
        
    Jinja2Templates.TemplateResponse = patched_template_response
except Exception as e:
    print(f"Starlette TemplateResponse compatibility patch not applied: {e}")
# -----------------------------------------------------------------------------

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline modules
from modules.utils import convert_audio_to_wav
from modules.transcribe import AudioTranscriber

# Global instances (lazy loaded to save memory and startup time)
pipeline_instances = {}

def load_pipeline():
    """Lazy loads all heavy AI models to prevent crashing on app startup."""
    expected_keys = ['transcriber']
    if not all(k in pipeline_instances for k in expected_keys):
        print("Initializing AI models... (this may take a minute)")
        try:
            if 'transcriber' not in pipeline_instances:
                # Use "tiny" model as recommended for the 512MB RAM free tier
                pipeline_instances['transcriber'] = AudioTranscriber(model_name="tiny")
            print("Transcription model loaded successfully!")
        except Exception as e:
            pipeline_instances.clear()
            raise e
    return pipeline_instances

def process_audio(audio_path, progress=gr.Progress()):
    """
    Main orchestration function. Connected directly to the Gradio button.
    progress parameter hooks into Gradio's live progress bar.
    """
    if not audio_path:
        raise gr.Error(
            "No audio detected! Please make sure to:\n"
            "1. Upload an audio file OR\n"
            "2. If recording via Microphone, click the 'Stop Recording' (square/pause icon) to finalize your recording before clicking 'Analyze Call'.\n"
            "3. Ensure your browser is allowed to access your microphone (only works on 'localhost' or secure HTTPS pages)."
        )
        
    try:
        # 0. Load Models
        progress(0.1, desc="Loading optimized Whisper model...")
        models = load_pipeline()
        
        # 1. Audio Prep
        progress(0.3, desc="Preprocessing and normalizing audio...")
        wav_path = convert_audio_to_wav(audio_path)
        
        # 2. Transcription
        progress(0.5, desc="Transcribing audio with optimized CPU engine...")
        whisper_segments = models['transcriber'].transcribe_audio(wav_path)
        if not whisper_segments:
            raise gr.Error("Transcription failed. Please check the logs.")
            
        # Format conversation with beautiful timestamps
        progress(0.8, desc="Formatting transcript...")
        conversation_text = ""
        for seg in whisper_segments:
            start = seg['start']
            end = seg['end']
            text = seg['text']
            minutes_start = int(start // 60)
            seconds_start = int(start % 60)
            minutes_end = int(end // 60)
            seconds_end = int(end % 60)
            timestamp = f"[{minutes_start:02d}:{seconds_start:02d} - {minutes_end:02d}:{seconds_end:02d}]"
            conversation_text += f"{timestamp} {text}\n"
            
        # Cleanup temp wav file if one was created
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)
            
        progress(1.0, desc="Complete!")
        
        # In a 512MB RAM environment, local SLM is bypassed to prevent memory exhaustion
        analysis_result = {
            "status": "Inference Optimized",
            "message": "Local AI Quality Assurance analysis and PDF generation are bypassed on the Render Free Tier (512MB RAM limit) to prevent system crashes.",
            "tip": "To enable complete speaker diarization, AI-driven evaluation scores, and dynamic PDF reports, deploy to Hugging Face Spaces (16GB RAM Free Tier) or use a paid/GPU-enabled tier."
        }
        
        formatted_analysis = json.dumps(analysis_result, indent=4)
        
        return (
            conversation_text,         # Sent to the TextArea
            formatted_analysis,        # Sent to the Code block
            []                         # Sent to the File downloader (empty)
        )
        
    except Exception as e:
        traceback.print_exc()
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
with gr.Blocks(title="VoxPulse - Call Analysis") as app:
    
    # 1. Header Section
    with gr.Column(elem_classes="header-text"):
        gr.Markdown("# 🎙️ VoxPulse: AI Voice Call Analysis")
        gr.Markdown("Upload a customer support call to automatically transcribe audio under optimized resource limits.")
        
    # 2. Top Row (Inputs & File Outputs)
    with gr.Row():
        
        # Input Column
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Upload or Record Call Recording")
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Input Audio (Upload .mp3/.wav or Record Microphone)"
            )
            analyze_btn = gr.Button("Analyze Call", variant="primary", size="lg")
            
        # File Download Column
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 2. Download Reports")
            report_files = gr.File(label="Generated Documents (PDF & JSON)", file_count="multiple", interactive=False)
            gr.Markdown("*Reports are disabled on the Render Free Tier to maintain stability.*")
            
    # 3. Bottom Row (Text Visualizations)
    with gr.Row():
        
        # Transcript Viewer
        with gr.Column():
            gr.Markdown("### Timestamped Transcript")
            transcript_output = gr.TextArea(
                label="Conversation",
                lines=18,
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
    # Get port from environment (Render dynamically assigns this, defaulting to 7860 for local)
    port = int(os.environ.get("PORT", 7860))
    # In cloud environments, avoid opening a public gradio tunnel since Render provides secure SSL routing
    is_prod = "RENDER" in os.environ
    
    # Launch the server
    app.launch(
        server_name="0.0.0.0", 
        server_port=port, 
        share=not is_prod
    )
