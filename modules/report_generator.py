import json
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_json(self, base_filename: str, report_data: dict) -> str:
        output_path = os.path.join(self.output_dir, f"{base_filename}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        return output_path
        
    def generate_markdown(self, base_filename: str, report_data: dict) -> str:
        output_path = os.path.join(self.output_dir, f"{base_filename}.md")
        
        md_content = f"# Voice Call Analysis Report\n\n"
        md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        md_content += f"## Transcription\n\n"
        for segment in report_data.get("transcript", []):
            start_fmt = f"{segment['start']:.2f}s"
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
            md_content += f"**[{start_fmt}] {speaker}:** {text}\n\n"
            
        md_content += f"## AI Analysis Feedback\n\n"
        md_content += report_data.get("analysis", "No analysis provided.") + "\n"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        return output_path
