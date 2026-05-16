# Clover Cancer: Pancreatic Cancer Early Detection with Fine-tuned Gemma 4

## The Problem

Pancreatic cancer is the deadliest major cancer. The 5-year survival rate is just **12%** — the lowest of all major cancers. But when caught at Stage 1, survival jumps to **44%**. The difference between life and death is often just **timing**.

The tragedy? The symptoms are vague and routinely dismissed:
- Back pain → "probably your desk job"
- New-onset diabetes → "let's start metformin"
- Weight loss → "good for you"
- Fatigue → "get more sleep"

By the time symptoms become obvious — jaundice, severe pain, dramatic weight loss — it's usually Stage III or IV. The window for curative surgery has closed.

**What if we could catch the patterns that get missed?**

## The Solution

Clover Cancer is a fine-tuned Gemma 4 model trained to recognize pancreatic cancer symptom patterns and risk factors. Not a chatbot — a specialized triage system built to catch what gets missed.

### How It Works

1. **Patient describes symptoms** in natural language
2. **Model analyzes** the presentation against learned clinical patterns
3. **Structured output**: risk level, possible conditions with reasoning, urgency, recommended next steps

### Example

**Input:**
> "I'm a 58-year-old male. I was just diagnosed with diabetes last month. I've lost about 15 pounds without trying. I have this deep ache in my mid-back that won't go away."

**Output:**
```json
{
  "risk_assessment": "high",
  "urgency": "urgent",
  "conditions": [
    {
      "name": "Pancreatic ductal adenocarcinoma",
      "likelihood": "high",
      "reasoning": "New-onset diabetes after 50 with weight loss and back pain is a recognized early PC pattern. Studies show NOD can precede diagnosis by 1-3 years."
    }
  ],
  "recommended_actions": [
    "Abdominal CT with pancreatic protocol",
    "CA 19-9 tumor marker",
    "Endoscopic ultrasound if CT inconclusive"
  ]
}
```

## Technical Approach

### Why Fine-tuning, Not Just Prompting

A base language model can discuss pancreatic cancer, but it won't consistently:
- Recognize subtle symptom clusters as high-risk
- Provide structured, auditable assessments
- Match urgency to clinical severity appropriately
- Recommend specific diagnostic pathways

Fine-tuning teaches the model to **think like a clinician** — not just recite facts, but reason through differential diagnoses.

### Model Selection: Gemma 4 E2B

- **2B parameters** — small enough for 24GB unified memory on Apple Silicon
- **Apache 2.0 license** — fully open for research and modification
- **Strong reasoning base** — Gemma 4's pre-training provides solid medical knowledge foundation

### Training Pipeline

```
Medical Research (NCCN, PubMed, Frontiers)
        ↓
Dataset Curation (350 structured examples)
        ↓
LoRA Fine-tuning (PEFT, rank 16, 0.6% trainable)
        ↓
Evaluation (10 clinical test scenarios)
        ↓
Gradio Demo (interactive triage interface)
```

### Key Technical Decisions

1. **Unwrap Gemma4ClippableLinear**: The model uses custom wrapper modules that PEFT doesn't recognize. Solution: unwrap to inner Linear layers before applying LoRA.

2. **MPS Training**: Apple Silicon's unified memory is ideal for LLM training. No quantization needed — E2B fits in float16.

3. **Structured Output**: Training with JSON output format ensures consistent, parseable assessments rather than free-text responses.

4. **Synthetic + Real Patterns**: Training data combines real clinical patterns from NCCN guidelines and PubMed research with synthetic augmentation for coverage.

## Dataset

- **350 total examples** (280 train, 35 val, 35 test)
- **152 pancreatic cancer patterns**: symptom clusters, risk factors, differential diagnoses
- **198 general medical examples**: prevents catastrophic forgetting
- **Sources**: NCCN guidelines, Chari et al. (MD Anderson), Johns Hopkins CT study, Frontiers 2025 review

### Pattern Categories

| Category | Examples | Risk Level |
|----------|----------|------------|
| Classic early detection | NOD + weight loss + back pain | High |
| Emergency presentations | Painless jaundice | Critical |
| Genetic risk | BRCA2 + symptoms | High |
| Subtle patterns | Mild jaundice, recurrent DVT | Medium-High |
| Differential diagnosis | Low-risk, must distinguish | Low |

## Results

The fine-tuned model demonstrates improved accuracy over the base Gemma 4 model:

- **Risk assessment accuracy**: Correctly identifies high-risk patterns that the base model misses
- **Urgency classification**: Appropriate urgency matching (not over-alarming for low-risk)
- **Reasoning quality**: Clinically grounded explanations with specific diagnostic recommendations
- **Condition identification**: Correctly ranks pancreatic cancer higher in relevant presentations

## Impact

### The "2am Parent" Scenario

A parent wakes at 2am with a child who's been having stomach pain. They've also noticed the child has been more tired than usual. The model assesses: low risk, likely routine, recommends follow-up with pediatrician.

But imagine a 58-year-old with new diabetes, weight loss, and back pain. The model flags: **high risk, urgent imaging recommended**. That flag could be the difference between Stage 1 (44% survival) and Stage 4 (3%).

### Democratizing Clinical Pattern Recognition

Not everyone has access to a gastroenterologist who will connect "new diabetes + back pain + weight loss" to pancreatic cancer. This tool brings that pattern recognition to anyone with an internet connection.

## Future Work

1. **Expand training data**: Partner with medical institutions for real (anonymized) clinical data
2. **Multi-cancer support**: Extend to other cancers with similar early detection gaps
3. **Clinical validation**: Partner with oncologists for validation study
4. **Edge deployment**: Quantize for mobile/edge deployment in low-resource settings
5. **Multimodal**: Add imaging analysis (CT scans) for comprehensive assessment

## Author

**Adhyaay Karnwal** — 15-year-old developer, Morris Hills High School

## Links

- **Repository**: github.com/adhyaay-karnwal/clover-cancer
- **Live Demo**: [Coming soon]
- **Model Weights**: [Coming soon]
