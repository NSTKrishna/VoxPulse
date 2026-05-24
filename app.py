import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables (e.g., HF_TOKEN)
load_dotenv()

from modules.utils import convert_audio_to_wav
from modules.transcription import TranscriptionService
from modules.diarization import DiarizationService
from modules.analysis import CallAnalyzerService
from modules.report_generator import ReportGenerator

# Global lazy-loaded services to avoid loading on app startup unless needed
services = {}

def get_services():
    if not services:
        print("Initializing services...")
        services['transcription'] = TranscriptionService(model_size="base")
        services['diarization'] = DiarizationService()
        services['analysis'] = CallAnalyzerService()
        services['report'] = ReportGenerator()
        print("Services initialized successfully.")
    return services

def process_audio(audio_path):
    if not audio_path:
        return "Please upload an audio file.", None, None
        
    try:
        svc = get_services()
        
        # 1. Convert to Wav if necessary
        print(f"Processing uploaded file: {audio_path}")
        wav_path = convert_audio_to_wav(audio_path)
        
        # 2. Transcribe
        whisper_result = svc['transcription'].transcribe(wav_path)
        
        # 3. Diarize
        diarization_result = svc['diarization'].diarize(wav_path)
        
        # 4. Align
        aligned_transcript = svc['diarization'].align_transcription(whisper_result, diarization_result)
        
        # Generate full text for analysis
        full_text = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in aligned_transcript])
        
        # 5. Analyze with Phi-3 Mini
        analysis_result = svc['analysis'].analyze(full_text)
        
        # 6. Generate Reports
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        report_data = {
            "transcript": aligned_transcript,
            "analysis": analysis_result
        }
        
        json_path = svc['report'].generate_json(base_name, report_data)
        md_path = svc['report'].generate_markdown(base_name, report_data)
        
        # Cleanup temp wav if it was created
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)
            
        # Format output for UI
        ui_transcript = ""
        for seg in aligned_transcript:
            ui_transcript += f"**{seg['speaker']}** ({seg['start']:.2f}s - {seg['end']:.2f}s): {seg['text']}\n\n"
            
        return ui_transcript, analysis_result, [json_path, md_path]
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error occurred: {str(e)}", "", None

# Gradio Interface
with gr.Blocks(title="VoxPulse - AI Voice Call Analysis") as app:
    gr.Markdown("# 🎙️ VoxPulse: AI Voice Call Analysis System")
    gr.Markdown("Upload a call recording (.mp3, .wav, .m4a) to generate a diarized transcription and an AI-powered feedback report using entirely open-source models (Whisper, pyannote.audio, Phi-3).")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            process_btn = gr.Button("Analyze Call", variant="primary")
            
        with gr.Column():
            report_files = gr.File(label="Downloadable Reports (JSON & Markdown)")
            
    with gr.Row():
        with gr.Column():
            transcript_output = gr.Markdown(label="Diarized Transcription")
            
        with gr.Column():
            analysis_output = gr.Markdown(label="AI Analysis & Feedback")
            
    process_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcript_output, analysis_output, report_files]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
