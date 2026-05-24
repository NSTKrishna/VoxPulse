"""
report_generator.py

Production-ready module for generating comprehensive call analysis reports.
Supports output in JSON, Markdown, and PDF formats.

Best Practices Included:
1. **Separation of Concerns:** Distinct methods for each output format.
2. **Metadata Injection:** Automatically stamps reports with generation time and tracking IDs.
3. **Scalability:** Designed to easily accept new data fields without breaking the layout.
4. **Professional Formatting:** Utilizes clean Markdown and HTML-to-PDF conversion for enterprise-grade documents.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List

import markdown
from weasyprint import HTML, CSS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Handles the generation of analysis reports in JSON, Markdown, and PDF.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initializes the report generator and ensures the output directory exists.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _inject_metadata(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Injects tracking metadata and timestamps into the report data.
        """
        # Create a shallow copy to avoid mutating the original dict
        enriched_data = analysis_data.copy()
        
        enriched_data["_metadata"] = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "system_version": "VoxPulse v1.0",
        }
        return enriched_data

    def generate_json(self, base_filename: str, analysis_data: Dict[str, Any]) -> str:
        """
        Generates the raw JSON analysis file.
        """
        enriched_data = self._inject_metadata(analysis_data)
        output_path = os.path.join(self.output_dir, f"{base_filename}_analysis.json")
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enriched_data, f, indent=4, ensure_ascii=False)
            logger.info(f"JSON report generated: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            raise

    def generate_markdown(self, base_filename: str, analysis_data: Dict[str, Any]) -> str:
        """
        Generates a professionally formatted Markdown report.
        """
        enriched_data = self._inject_metadata(analysis_data)
        output_path = os.path.join(self.output_dir, f"{base_filename}_report.md")
        
        # Extract metadata
        meta = enriched_data.get("_metadata", {})
        
        # Build Markdown content
        md = []
        md.append("# 🎙️ VoxPulse: Call Quality Analysis Report")
        md.append("---")
        md.append(f"**Report ID:** `{meta.get('report_id')}`  ")
        md.append(f"**Generated:** {meta.get('generated_at')}  ")
        md.append(f"**System:** {meta.get('system_version')}  ")
        md.append("---\n")
        
        # Core Metrics Overview (Using a table for clean formatting)
        md.append("## 📊 Executive Overview\n")
        md.append("| Metric | Status |")
        md.append("|---|---|")
        md.append(f"| **Overall Sentiment** | {enriched_data.get('overall_sentiment', 'N/A')} |")
        md.append(f"| **Customer Sentiment** | {enriched_data.get('customer_sentiment', 'N/A')} |")
        md.append(f"| **Agent Score** | {enriched_data.get('agent_score', 'N/A')}/10 |")
        md.append(f"| **Resolution Status** | {enriched_data.get('resolution_status', 'N/A')} |")
        md.append(f"| **Risk Level** | {enriched_data.get('customer_risk_level', 'N/A')} |\n")
        
        md.append("## 📝 Call Summary")
        md.append(f"{enriched_data.get('call_summary', 'No summary provided.')}\n")
        
        md.append("## ⭐ Agent Performance")
        
        md.append("### Strengths")
        strengths = enriched_data.get('strengths', [])
        if strengths:
            for s in strengths:
                md.append(f"- {s}")
        else:
            md.append("- None identified.")
        md.append("")
            
        md.append("### Improvement Areas")
        improvements = enriched_data.get('improvement_areas', [])
        if improvements:
            for i in improvements:
                md.append(f"- {i}")
        else:
            md.append("- None identified.")
        md.append("")
            
        md.append("## ⚠️ Compliance Notes")
        compliance = enriched_data.get('compliance_issues', [])
        if compliance:
            for c in compliance:
                md.append(f"- **WARNING:** {c}")
        else:
            md.append("- No compliance issues detected. ✅")
        md.append("")
            
        md.append("## 🎯 Recommended Next Steps")
        steps = enriched_data.get('recommended_next_steps', [])
        if steps:
            for idx, step in enumerate(steps, 1):
                md.append(f"{idx}. {step}")
        else:
            md.append("No immediate action required.")
        md.append("")
            
        md.append("## 💡 Final Recommendation")
        score = enriched_data.get('agent_score', 0)
        if score >= 9:
            md.append("> **Excellent Interaction.** Consider using this call as a training example for new agents.")
        elif score >= 7:
            md.append("> **Good Interaction.** Standard quality. Address minor improvement areas in the next 1-on-1.")
        else:
            md.append("> **Requires Review.** Significant issues identified. Manager review recommended.")
            
        md_content = "\n".join(md)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Markdown report generated: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate Markdown report: {e}")
            raise

    def generate_pdf(self, base_filename: str, analysis_data: Dict[str, Any]) -> str:
        """
        Generates a professional PDF report by converting the Markdown output to HTML, 
        styling it with CSS, and using WeasyPrint to render the PDF.
        """
        output_path = os.path.join(self.output_dir, f"{base_filename}_report.pdf")
        
        # Ensure we have the markdown content
        md_filepath = self.generate_markdown(base_filename, analysis_data)
        
        try:
            with open(md_filepath, "r", encoding="utf-8") as f:
                md_content = f.read()
                
            # Convert Markdown to HTML. Extensions add support for tables and blockquotes.
            html_content = markdown.markdown(md_content, extensions=['tables', 'sane_lists'])
            
            # Wrap the HTML in a basic document structure
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>VoxPulse Report</title>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Define professional enterprise CSS styling for the PDF
            css = CSS(string='''
                @page {
                    size: A4;
                    margin: 2cm;
                    @bottom-right {
                        content: "Page " counter(page) " of " counter(pages);
                        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                        font-size: 9pt;
                        color: #666;
                    }
                }
                body {
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    color: #333;
                    line-height: 1.6;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #2980b9;
                    margin-top: 30px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 5px;
                }
                h3 {
                    color: #34495e;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: bold;
                }
                blockquote {
                    background: #f9f9f9;
                    border-left: 5px solid #3498db;
                    margin: 1.5em 10px;
                    padding: 1em 10px;
                    quotes: "\\201C""\\201D""\\2018""\\2019";
                }
                hr {
                    border: 0;
                    border-top: 1px solid #eee;
                    margin: 20px 0;
                }
            ''')
            
            # Render PDF
            logger.info("Rendering PDF via WeasyPrint...")
            HTML(string=full_html).write_pdf(output_path, stylesheets=[css])
            
            logger.info(f"PDF report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}", exc_info=True)
            raise

    def generate_all(self, base_filename: str, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Convenience method to generate JSON, Markdown, and PDF all at once.
        """
        paths = {}
        paths['json'] = self.generate_json(base_filename, analysis_data)
        paths['markdown'] = self.generate_markdown(base_filename, analysis_data)
        paths['pdf'] = self.generate_pdf(base_filename, analysis_data)
        return paths

if __name__ == "__main__":
    # =========================================================================
    # Sample Usage
    # =========================================================================
    print("--- Report Generator Sample Usage ---\n")
    
    # Dummy data simulating the validated output from analyze.py
    sample_analysis = {
        "call_summary": "The customer called to dispute a $15 late fee on their billing statement. The agent investigated, confirmed it was a system error, and proactively removed the fee.",
        "overall_sentiment": "Positive",
        "customer_sentiment": "Satisfied",
        "agent_score": 9,
        "strengths": [
            "Active listening",
            "Quick problem resolution",
            "Professional tone"
        ],
        "improvement_areas": [
            "Could have offered auto-pay to prevent future issues"
        ],
        "compliance_issues": [],
        "recommended_next_steps": [
            "Ensure the $15 credit is applied immediately to the account"
        ],
        "customer_risk_level": "Low",
        "call_category": "Billing",
        "resolution_status": "Resolved",
        "confidence_score": 0.98
    }
    
    generator = ReportGenerator(output_dir="../outputs")
    try:
        paths = generator.generate_all("sample_call_123", sample_analysis)
        print("Generated files:")
        for fmt, filepath in paths.items():
            print(f"[{fmt.upper()}] {filepath}")
    except Exception as e:
        print(f"Failed to generate reports: {e}")
