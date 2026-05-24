"""
analyze.py

Production-ready module for analyzing customer support calls using Microsoft Phi-3 Mini.

Best Practices Included:
1. **Memory Optimization:** Uses float16 on GPU/MPS to reduce VRAM footprint and 'low_cpu_mem_usage'.
2. **Device Management:** Automatically maps to CUDA, MPS, or CPU.
3. **Structured Output:** Forces the model to generate JSON via system prompting and validates the output.
4. **Resilience:** Implements retry logic if the model fails to generate valid JSON.
5. **Generation Config:** Uses optimal parameters for analytical tasks (low temperature, controlled max tokens).
"""

import json
import logging
import time
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CallAnalyzer:
    """
    A robust class to analyze transcripts using the Phi-3 Mini instruct model.
    """
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initializes the model and tokenizer, optimizing for the available hardware.
        """
        self.model_id = model_id
        
        # Determine appropriate device and data type
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        logger.info(f"Loading tokenizer and model '{model_id}' on {self.device}...")
        
        try:
            # Load tokenizer (trust_remote_code is required for Phi-3)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            # Load model with memory optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            
            # Setup Hugging Face generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.model.device
            )
            logger.info("Phi-3 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Phi-3 model: {e}", exc_info=True)
            raise

    def analyze_transcript(self, transcript: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Analyzes a customer support call transcript and returns structured JSON feedback.
        Includes retry logic for robustness against malformed JSON outputs.
        
        Args:
            transcript (str): The formatted conversation text.
            max_retries (int): Number of times to retry if JSON parsing fails.
            
        Returns:
            Dict: Parsed JSON containing the analysis, or None if it fails.
        """
        # System prompt explicitly asking for JSON. Phi-3 uses <|system|>, <|user|>, <|assistant|> tags
        prompt = f"""<|system|>
You are an elite Quality Assurance AI for a customer support center. Your sole function is to analyze the provided call transcript and evaluate the interaction based on predefined criteria. 

CRITICAL INSTRUCTIONS:
1. OUTPUT FORMAT: You must output ONLY a raw, perfectly formatted JSON object. Do not wrap the JSON in markdown blocks (e.g., no ```json ... ```). Do not include any conversational filler, pleasantries, or explanations before or after the JSON.
2. NO HALLUCINATION: Base your analysis STRICTLY on the text provided in the transcript. Do not invent facts, assume outcomes that did not happen on the call, or reference external knowledge.
3. SCORING FAIRNESS: The `agent_score` must be an integer from 1 to 10. Start at 10 and deduct points for rudeness, lack of empathy, failing to provide solutions, or placing the customer on unnecessary hold.
4. UNRESOLVED ISSUES: If the transcript ends without a clear resolution or next steps, mark `resolution_status` as "Unresolved".
5. EMPTY ARRAYS: If there are no strengths, improvement areas, or compliance issues found, return an empty array `[]`. Do not invent items to fill the arrays.

JSON SCHEMA REQUIREMENT:
Your output must exactly match the keys and data types of this JSON structure:
{{
  "call_summary": "A concise 2-3 sentence summary of the customer's issue and the outcome.",
  "overall_sentiment": "Must be one of: Positive, Neutral, Negative, Mixed.",
  "customer_sentiment": "Must be one of: Delighted, Satisfied, Neutral, Frustrated, Angry.",
  "agent_score": 0,
  "strengths": ["string", "string"],
  "improvement_areas": ["string", "string"],
  "compliance_issues": ["string"],
  "recommended_next_steps": ["string"],
  "customer_risk_level": "Must be one of: Low, Medium, High (High risk means threat to cancel or escalate).",
  "call_category": "e.g., Billing, Technical Support, Sales, General Inquiry.",
  "resolution_status": "Must be one of: Resolved, Partially Resolved, Unresolved, Escalated.",
  "confidence_score": 0.0
}}

FEW-SHOT EXAMPLE:
[Input Transcript]
Agent: Hello, thank you for calling. How can I help?
Customer: Hi, my internet has been down for three days. I'm paying for a service I can't use.
Agent: I'm sorry to hear that. I can see an outage in your area. It should be fixed by tomorrow.
Customer: That's ridiculous.
Agent: I understand. I will issue a $20 credit to your account for the downtime.
Customer: Okay, thank you. That helps.
Agent: Have a good day.

[Expected Output]
{{
  "call_summary": "The customer called regarding a 3-day internet outage. The agent identified a local outage, informed the customer of the ETA for a fix, and proactively issued a $20 credit to appease the customer.",
  "overall_sentiment": "Mixed",
  "customer_sentiment": "Frustrated",
  "agent_score": 9,
  "strengths": ["Proactive compensation", "Clear communication", "Empathy"],
  "improvement_areas": ["Could have asked if the customer needed help setting up a mobile hotspot"],
  "compliance_issues": [],
  "recommended_next_steps": ["Ensure the $20 credit is successfully applied to the next billing cycle"],
  "customer_risk_level": "Medium",
  "call_category": "Technical Support",
  "resolution_status": "Partially Resolved",
  "confidence_score": 0.95
}}
<|end|>
<|user|>
Analyze the following call transcript and generate the required JSON evaluation:

[Input Transcript]
{transcript}
<|end|>
<|assistant|>
"""
        
        generation_args = {
            "max_new_tokens": 512,
            "temperature": 0.1,      # Low temperature for analytical consistency (less hallucination)
            "do_sample": True,
            "return_full_text": False
        }

        for attempt in range(1, max_retries + 1):
            logger.info(f"Analysis attempt {attempt}/{max_retries}...")
            try:
                # Generate response
                output = self.pipe(prompt, **generation_args)
                raw_text = output[0]['generated_text'].strip()
                
                # Cleanup: Sometimes models wrap JSON in markdown blocks (```json ... ```)
                if raw_text.startswith("```json"):
                    raw_text = raw_text.replace("```json", "", 1)
                if raw_text.endswith("```"):
                    # Remove trailing ```
                    raw_text = raw_text[::-1].replace("```", "", 1)[::-1]
                raw_text = raw_text.strip()
                
                # JSON Validation
                parsed_json = json.loads(raw_text)
                logger.info("Successfully generated and validated JSON analysis.")
                return parsed_json
                
            except json.JSONDecodeError as je:
                logger.warning(f"Attempt {attempt} failed: Output was not valid JSON. Error: {je}")
                # Dynamically increase temperature to try and "shake" the model out of a bad generation loop
                generation_args["temperature"] = min(0.5, generation_args["temperature"] + 0.1)
                time.sleep(1) # Brief pause before retry
            except Exception as e:
                logger.error(f"Attempt {attempt} failed due to unexpected error: {e}", exc_info=True)
                time.sleep(1)
                
        logger.error("All analysis attempts failed.")
        return None


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Sample Usage
    # ---------------------------------------------------------
    print("--- Analyzer Sample Usage ---")
    
    # 1. Initialize analyzer
    analyzer = CallAnalyzer()
    
    # 2. Provide a sample transcript
    sample_transcript = """
    Agent: Hello, thank you for calling support. How can I help you today?
    Customer: Hi, I ordered a package three weeks ago and it still hasn't arrived. I'm very frustrated!
    Agent: I completely understand your frustration, and I apologize for the delay. Let me check your tracking number.
    Customer: It's 123456789.
    Agent: Thank you. It looks like it was stuck at customs but cleared yesterday. It should arrive by Tuesday. I will refund your shipping costs for the inconvenience.
    Customer: Oh, that's great. Thank you so much for your help.
    Agent: You're welcome! Is there anything else I can assist you with?
    Customer: No, that's all. Bye.
    """
    
    print("\nAnalyzing transcript...\n")
    
    # 3. Analyze
    result = analyzer.analyze_transcript(sample_transcript)
    
    # 4. View Validated Output
    if result:
        print("Validated JSON Output:")
        print(json.dumps(result, indent=2))
    else:
        print("Analysis failed.")
