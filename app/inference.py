"""
Clover Cancer — Inference Module

Handles loading the fine-tuned Gemma 4 model and running inference
for pancreatic cancer symptom triage.

Supports two backends:
- Unsloth (NVIDIA/AMD/Intel GPU) — fast inference with 4-bit quantization
- Transformers + PEFT (Mac/CPU/MPS) — fallback for Apple Silicon

Author: Adhyaay Karnwal
"""

import json
import os
import re
import sys
from typing import Optional

# System prompt for the model
SYSTEM_PROMPT = """You are a medical triage AI specialized in pancreatic cancer early detection. Analyze the patient presentation and provide a structured assessment.

Your response MUST be valid JSON with this structure:
{
  "risk_assessment": "low|medium|high|critical",
  "conditions": [
    {"name": "condition name", "likelihood": "low|medium|high", "reasoning": "why this condition"}
  ],
  "urgency": "self-care|routine|urgent|emergency",
  "recommended_actions": ["action 1", "action 2"],
  "reasoning_chain": "Detailed clinical reasoning explaining your assessment"
}

Key guidelines:
- Always consider pancreatic cancer patterns, especially in patients over 50
- New-onset diabetes after 50 with weight loss or back pain = high suspicion
- Painless jaundice = emergency until proven otherwise
- BRCA2/Lynch syndrome carriers have elevated PC risk
- Match urgency to clinical severity — don't over-alarm for low-risk presentations
- Be specific about imaging recommendations (CT, MRI, EUS)
- Include CA 19-9 when PC is suspected"""


def _detect_device():
    """Detect the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def _has_unsloth():
    """Check if Unsloth is available."""
    try:
        import unsloth
        import torch
        return torch.cuda.is_available()
    except (ImportError, AttributeError):
        return False


class PancreaticCancerTriage:
    """Inference engine for pancreatic cancer triage."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self._loaded = False
        self._device = _detect_device()

    def load_model(self, model_path: Optional[str] = None):
        """Load the fine-tuned model with LoRA adapters."""
        path = model_path or self.model_path
        if not path:
            raise ValueError("No model path specified")

        print(f"Loading model from {path}...")
        print(f"Device: {self._device}")

        if _has_unsloth():
            self._load_with_unsloth(path)
        else:
            self._load_with_transformers(path)

    def _load_with_unsloth(self, path: str):
        """Load using Unsloth (NVIDIA/AMD/Intel GPU only)."""
        try:
            from unsloth import FastLanguageModel
            from peft import PeftModel

            base_model_name = "unsloth/gemma-4-e2b-it"
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                dtype=None,
                max_seq_length=2048,
                load_in_4bit=True,
            )

            self.model = PeftModel.from_pretrained(self.model, path)
            self.model = self.model.merge_and_unload()

            self._loaded = True
            print("Model loaded with Unsloth backend.")

        except Exception as e:
            print(f"Unsloth loading failed: {e}")
            self._loaded = False

    def _load_with_transformers(self, path: str):
        """Load using transformers + PEFT (works on Mac/CPU/MPS)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Try loading the LoRA adapter config to find the base model
            config_path = os.path.join(path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "google/gemma-4-e2b-it")
            else:
                base_model_name = "google/gemma-4-e2b-it"

            print(f"Base model: {base_model_name}")

            # Determine dtype based on device
            if self._device == "mps":
                dtype = torch.float16
            elif self._device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=True,
            )

            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=dtype,
                device_map="auto" if self._device != "mps" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            if self._device == "mps":
                self.model = self.model.to("mps")

            # Apply LoRA adapters
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, path)
            self.model = self.model.merge_and_unload()

            self.model.eval()
            self._loaded = True
            print("Model loaded with transformers backend.")

        except Exception as e:
            print(f"Transformers loading failed: {e}")
            print("Falling back to mock mode.")
            self._loaded = False

    def assess(self, patient_description: str) -> dict:
        """Run triage assessment on a patient description."""
        if not self._loaded:
            return self._mock_assessment(patient_description)

        # Merge system into user for Gemma 4 (no native system role)
        user_content = f"{SYSTEM_PROMPT}\n\nPlease assess my symptoms and tell me what might be going on:\n\n{patient_description}"
        messages = [
            {"role": "user", "content": user_content},
        ]

        try:
            import torch

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move to correct device
            if self._device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            elif self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    use_cache=True,
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            return self._parse_response(response)

        except Exception as e:
            print(f"Inference error: {e}")
            return {"error": str(e)}

    def _parse_response(self, response: str) -> dict:
        """Parse model response into structured output."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {
            "raw_response": response,
            "risk_assessment": "unknown",
            "urgency": "unknown"
        }

    def _mock_assessment(self, description: str) -> dict:
        """Return a mock assessment for testing without model."""
        desc_lower = description.lower()

        risk_factors = 0
        if "jaundice" in desc_lower or "yellow" in desc_lower:
            risk_factors += 3
        if "weight loss" in desc_lower or ("lost" in desc_lower and "pound" in desc_lower):
            risk_factors += 2
        if "diabetes" in desc_lower and "new" in desc_lower:
            risk_factors += 2
        if "back pain" in desc_lower:
            risk_factors += 1
        if "brca" in desc_lower:
            risk_factors += 2

        if risk_factors >= 4:
            risk = "high"
            urgency = "urgent"
        elif risk_factors >= 2:
            risk = "medium"
            urgency = "routine"
        else:
            risk = "low"
            urgency = "self-care"

        return {
            "risk_assessment": risk,
            "conditions": [
                {"name": "Requires model for accurate assessment", "likelihood": "unknown", "reasoning": "Mock mode — load model for real assessment"}
            ],
            "urgency": urgency,
            "recommended_actions": ["Consult healthcare provider", "Provide detailed symptom history"],
            "reasoning_chain": f"Mock assessment based on keyword analysis. Risk factors detected: {risk_factors}. Load the fine-tuned model for accurate clinical assessment.",
            "_mock": True
        }


def format_assessment(assessment: dict) -> str:
    """Format assessment as readable text."""
    if "error" in assessment:
        return f"Error: {assessment['error']}"

    lines = []
    lines.append(f"Risk Level: {assessment.get('risk_assessment', 'unknown').upper()}")
    lines.append(f"Urgency: {assessment.get('urgency', 'unknown').upper()}")
    lines.append("")

    lines.append("Possible Conditions:")
    for cond in assessment.get("conditions", []):
        lines.append(f"  - {cond['name']} ({cond.get('likelihood', 'unknown')} likelihood)")
        lines.append(f"    {cond.get('reasoning', '')}")

    lines.append("")
    lines.append("Recommended Actions:")
    for action in assessment.get("recommended_actions", []):
        lines.append(f"  - {action}")

    lines.append("")
    lines.append("Reasoning:")
    lines.append(f"  {assessment.get('reasoning_chain', 'No reasoning provided')}")

    if assessment.get("_mock"):
        lines.append("")
        lines.append("[MOCK ASSESSMENT — load model for real results]")

    return "\n".join(lines)


if __name__ == "__main__":
    triage = PancreaticCancerTriage()
    result = triage.assess(
        "I'm a 58-year-old male. I was just diagnosed with diabetes. "
        "I've lost 15 pounds without trying. I have back pain that won't go away."
    )
    print(format_assessment(result))
