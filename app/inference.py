"""
Clover Cancer — Inference Module

Handles loading the fine-tuned Gemma 4 model and running inference
for pancreatic cancer symptom triage.

Author: Adhyaay Karnwal
"""

import json
import os
import re
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


class PancreaticCancerTriage:
    """Inference engine for pancreatic cancer triage."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self._loaded = False

    def load_model(self, model_path: Optional[str] = None):
        """Load the fine-tuned model."""
        path = model_path or self.model_path
        if not path:
            raise ValueError("No model path specified")

        print(f"Loading model from {path}...")

        try:
            from unsloth import FastModel
            from unsloth.chat_templates import get_chat_template

            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=path,
                dtype=None,
                max_seq_length=2048,
                load_in_4bit=True,
            )

            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="gemma-4",
            )

            self._loaded = True
            print("Model loaded successfully.")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to base Gemma 4 model...")
            self._load_base_model()

    def _load_base_model(self):
        """Load base Gemma 4 model as fallback."""
        try:
            from unsloth import FastModel
            from unsloth.chat_templates import get_chat_template

            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name="unsloth/gemma-4-e2b-it",
                dtype=None,
                max_seq_length=2048,
                load_in_4bit=True,
            )

            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="gemma-4",
            )

            self._loaded = True
            print("Base model loaded.")

        except Exception as e:
            print(f"Failed to load base model: {e}")
            self._loaded = False

    def assess(self, patient_description: str) -> dict:
        """Run triage assessment on a patient description."""
        if not self._loaded:
            return self._mock_assessment(patient_description)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please assess my symptoms and tell me what might be going on:\n\n{patient_description}"}
        ]

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

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
            # Try to extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # If JSON parsing fails, return raw response
        return {
            "raw_response": response,
            "risk_assessment": "unknown",
            "urgency": "unknown"
        }

    def _mock_assessment(self, description: str) -> dict:
        """Return a mock assessment for testing without model."""
        desc_lower = description.lower()

        # Simple keyword-based mock
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
    # Quick test
    triage = PancreaticCancerTriage()
    result = triage.assess(
        "I'm a 58-year-old male. I was just diagnosed with diabetes. "
        "I've lost 15 pounds without trying. I have back pain that won't go away."
    )
    print(format_assessment(result))
