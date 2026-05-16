"""
Clover Cancer — Model Evaluation Script

Evaluates the fine-tuned Gemma 4 model on pancreatic cancer triage tasks.
Compares base model vs fine-tuned model on:
- Risk assessment accuracy
- Condition identification
- Urgency classification
- Reasoning quality

Author: Adhyaay Karnwal
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# Test scenarios for evaluation
TEST_SCENARIOS = [
    {
        "name": "Classic high-risk pattern",
        "input": "I'm a 58-year-old male. I was just diagnosed with diabetes last month. I've lost about 15 pounds without trying. I have this deep ache in my mid-back that won't go away. This has been going on for about 3 months.",
        "expected_risk": "high",
        "expected_urgency": "urgent",
        "must_mention": ["pancreatic", "diabetes", "imaging"],
        "description": "New-onset diabetes + weight loss + back pain — classic early PC pattern"
    },
    {
        "name": "Emergency jaundice",
        "input": "I'm a 67-year-old female. My skin turned yellow and the whites of my eyes look yellow too. I've lost about 10 pounds in the last month. My urine is really dark, almost brown. I haven't had any pain really. This started about 2 weeks ago.",
        "expected_risk": "critical",
        "expected_urgency": "emergency",
        "must_mention": ["jaundice", "pancreatic", "CT", "urgent"],
        "description": "Painless jaundice with weight loss — likely pancreatic head mass"
    },
    {
        "name": "BRCA2 carrier with symptoms",
        "input": "I'm a 52-year-old female. I have a BRCA2 mutation and my mother died of pancreatic cancer. I've been having abdominal pain and fatigue for about 6 weeks. I've lost some weight too, maybe 8 pounds.",
        "expected_risk": "high",
        "expected_urgency": "urgent",
        "must_mention": ["BRCA2", "pancreatic", "MRI", "genetic"],
        "description": "High-risk genetic background with concerning symptoms"
    },
    {
        "name": "Low-risk back pain",
        "input": "I'm a 42-year-old male. I've had back pain and some stomach discomfort for about 2 weeks. I sit at a desk all day and haven't been exercising much. No weight loss or other symptoms.",
        "expected_risk": "low",
        "expected_urgency": "routine",
        "must_mention": ["musculoskeletal"],
        "should_not_mention": ["emergency", "cancer"],
        "description": "Young patient with isolated back pain — low cancer risk"
    },
    {
        "name": "Steatorrhea pattern",
        "input": "I'm a 62-year-old female. My stools have been really greasy and foul-smelling. They float and are hard to flush. I've lost about 12 pounds in 3 months. I feel bloated after eating even small meals.",
        "expected_risk": "medium",
        "expected_urgency": "urgent",
        "must_mention": ["pancreatic", "enzyme", "CT"],
        "description": "Exocrine insufficiency with weight loss — warrants imaging"
    },
    {
        "name": "New-onset diabetes alone",
        "input": "I'm a 56-year-old male. My doctor just told me I have type 2 diabetes. My blood sugar has been running high. I don't have any other real symptoms, maybe a bit of stomach discomfort.",
        "expected_risk": "medium",
        "expected_urgency": "routine",
        "must_mention": ["diabetes", "monitor"],
        "description": "NOD after 50 — should flag PC consideration but not alarm"
    },
    {
        "name": "Recurrent pancreatitis",
        "input": "I'm a 58-year-old male. I've been to the ER three times this year for pancreatitis. Each time it's terrible pain in my upper abdomen that radiates to my back. I was also just diagnosed with diabetes. I've lost about 10 pounds.",
        "expected_risk": "high",
        "expected_urgency": "urgent",
        "must_mention": ["pancreatic", "cancer", "CT", "pancreatitis"],
        "description": "Recurrent pancreatitis with NOD — concerning for underlying malignancy"
    },
    {
        "name": "DVT with abdominal symptoms",
        "input": "I'm a 63-year-old female. I've had two blood clots in my legs in the past year. I've also been having abdominal pain and lost about 15 pounds without trying. My doctor can't figure out why I keep getting clots.",
        "expected_risk": "high",
        "expected_urgency": "urgent",
        "must_mention": ["pancreatic", "Trousseau", "CT", "clot"],
        "description": "Recurrent DVT with weight loss — classic Trousseau syndrome pattern"
    },
    {
        "name": "IBS in young patient",
        "input": "I'm a 34-year-old female. I've been having bloating, diarrhea, and some fatigue for about 2 months. It comes and goes. No weight loss, no family history of cancer.",
        "expected_risk": "low",
        "expected_urgency": "routine",
        "must_mention": ["IBS", "irritable"],
        "should_not_mention": ["pancreatic cancer", "emergency"],
        "description": "Young patient with IBS symptoms — PC extremely unlikely"
    },
    {
        "name": "Mild jaundice (subtle)",
        "input": "I'm a 64-year-old male. I've been itching all over for weeks with no rash. My wife noticed my skin looks slightly yellowish. I've been more tired than usual. No pain. This has been going on about 3 weeks.",
        "expected_risk": "high",
        "expected_urgency": "urgent",
        "must_mention": ["jaundice", "pancreatic", "bile", "imaging"],
        "description": "Mild jaundice with pruritus — early obstructive pattern"
    },
]


def extract_json_from_response(response: str) -> Optional[dict]:
    """Extract structured JSON from model response."""
    # Try to find JSON in the response
    try:
        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return None


def evaluate_response(response: str, scenario: dict) -> dict:
    """Evaluate a single model response against expected outcomes."""
    result = {
        "scenario": scenario["name"],
        "passed": True,
        "issues": [],
        "scores": {}
    }

    response_lower = response.lower()

    # Check risk assessment
    extracted = extract_json_from_response(response)
    if extracted:
        actual_risk = extracted.get("risk_assessment", "").lower()
        if actual_risk == scenario["expected_risk"]:
            result["scores"]["risk_accuracy"] = 1.0
        elif scenario["expected_risk"] == "high" and actual_risk in ["critical", "medium"]:
            result["scores"]["risk_accuracy"] = 0.5
        else:
            result["scores"]["risk_accuracy"] = 0.0
            result["issues"].append(f"Risk: expected '{scenario['expected_risk']}', got '{actual_risk}'")
            result["passed"] = False

        # Check urgency
        actual_urgency = extracted.get("urgency", "").lower()
        if actual_urgency == scenario["expected_urgency"]:
            result["scores"]["urgency_accuracy"] = 1.0
        elif scenario["expected_urgency"] == "urgent" and actual_urgency in ["emergency"]:
            result["scores"]["urgency_accuracy"] = 0.7
        else:
            result["scores"]["urgency_accuracy"] = 0.0
            result["issues"].append(f"Urgency: expected '{scenario['expected_urgency']}', got '{actual_urgency}'")
            result["passed"] = False

        # Check reasoning quality (based on clinical reasoning depth, not length)
        reasoning = extracted.get("reasoning_chain", "")
        clinical_indicators = ["risk", "symptom", "age", "imaging", "suggest", "indicate",
                               "recommend", "diagnosis", "presentation", "guideline", "evidence"]
        indicator_count = sum(1 for word in clinical_indicators if re.search(r'\b' + word + r'\b', reasoning.lower()))
        if indicator_count >= 4:
            result["scores"]["reasoning_quality"] = 1.0
        elif indicator_count >= 2:
            result["scores"]["reasoning_quality"] = 0.5
        else:
            result["scores"]["reasoning_quality"] = 0.0
            result["issues"].append("Reasoning lacks clinical depth")
    else:
        result["scores"]["risk_accuracy"] = 0.0
        result["scores"]["urgency_accuracy"] = 0.0
        result["scores"]["reasoning_quality"] = 0.0
        result["issues"].append("Could not extract JSON from response")
        result["passed"] = False

    # Check must-mention terms with word boundary matching
    must_mention_score = 0
    for term in scenario.get("must_mention", []):
        pattern = re.compile(r'\b' + re.escape(term.lower()) + r'\b')
        if pattern.search(response_lower):
            must_mention_score += 1
        else:
            result["issues"].append(f"Missing required term: '{term}'")

    if scenario.get("must_mention"):
        result["scores"]["must_mention"] = must_mention_score / len(scenario["must_mention"])
        if result["scores"]["must_mention"] < 0.5:
            result["passed"] = False
    else:
        result["scores"]["must_mention"] = 1.0

    # Check should-not-mention terms with word boundary matching
    for term in scenario.get("should_not_mention", []):
        pattern = re.compile(r'\b' + re.escape(term.lower()) + r'\b')
        if pattern.search(response_lower):
            result["issues"].append(f"Should not mention: '{term}'")
            result["scores"]["appropriateness"] = result.get("scores", {}).get("appropriateness", 1.0) * 0.5

    result["scores"].setdefault("appropriateness", 1.0)

    # Overall score
    result["overall_score"] = sum(result["scores"].values()) / len(result["scores"])

    return result


def run_evaluation(model_path: Optional[str] = None, use_base_model: bool = False, temperature: float = 0.3):
    """Run evaluation on test scenarios."""
    print("=" * 60)
    print("Clover Cancer — Model Evaluation")
    print("=" * 60)

    if use_base_model:
        print("Evaluating: BASE Gemma 4 model (no fine-tuning)")
        # Load base model without LoRA adapters by passing None path
        eval_model_path = None
    else:
        print(f"Evaluating: Fine-tuned model from {model_path}")
        eval_model_path = model_path

    # Load model (fine-tuned or base)
    from app.inference import PancreaticCancerTriage
    triage = PancreaticCancerTriage()
    if eval_model_path and os.path.exists(eval_model_path):
        triage.load_model(eval_model_path)
    else:
        print("WARNING: No model path provided or path doesn't exist. Using mock mode.")
        print("  Pass model path as argument: python train/evaluate.py <model_path>")

    results = []
    for scenario in TEST_SCENARIOS:
        print(f"\n--- {scenario['name']} ---")
        print(f"  {scenario['description']}")

        # Run inference
        response_raw = triage.assess(scenario["input"])
        response_text = response_raw.get("_raw_response", str(response_raw))

        # Evaluate
        result = evaluate_response(response_text, scenario)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"  Result: {status} (score: {result['overall_score']:.2f})")
        for issue in result.get("issues", []):
            print(f"    - {issue}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)")

    avg_scores = {}
    for r in results:
        for k, v in r.get("scores", {}).items():
            avg_scores.setdefault(k, []).append(v)

    print("\nAverage Scores:")
    for metric, scores in avg_scores.items():
        print(f"  {metric}: {sum(scores)/len(scores):.2f}")

    return results


def compare_models(base_results: list, ft_results: list):
    """Compare base model vs fine-tuned model results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: Base vs Fine-tuned")
    print("=" * 60)

    for base, ft in zip(base_results, ft_results):
        scenario = base["scenario"]
        base_score = base.get("overall_score", 0)
        ft_score = ft.get("overall_score", 0)
        improvement = ft_score - base_score

        indicator = "+" if improvement > 0 else "-" if improvement < 0 else "="
        print(f"  {scenario}: Base={base_score:.2f} → FT={ft_score:.2f} ({indicator}{abs(improvement):.2f})")


if __name__ == "__main__":
    results = run_evaluation(use_base_model=True)
