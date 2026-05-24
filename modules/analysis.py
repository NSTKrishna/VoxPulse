import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class CallAnalyzerService:
    def __init__(self, model_id="microsoft/Phi-3-mini-4k-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading SLM {model_id} on {self.device}...")
        
        # Determine appropriate dtype based on device
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "mps":
             torch_dtype = torch.float16
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        self.model.to(self.device)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model.device
        )

    def analyze(self, transcript_text: str) -> str:
        """
        Uses Phi-3 to analyze the transcript and extract insights.
        """
        prompt = f"""<|user|>
You are an expert call quality analyst. Read the following call transcript and provide structured feedback.
Analyze the following points:
1. Summary of the call
2. Customer Sentiment
3. Agent Performance (strengths and areas for improvement)
4. Key Action Items

Transcript:
{transcript_text}
<|end|>
<|assistant|>
"""
        print("Generating analysis with Phi-3 Mini...")
        outputs = self.pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            return_full_text=False
        )
        
        return outputs[0]["generated_text"].strip()
