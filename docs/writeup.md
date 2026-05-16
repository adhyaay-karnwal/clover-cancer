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
LoRA Fine-tuning (Unsloth on Kaggle T4, rank 16, 0.6% trainable)
        ↓
Evaluation (10 clinical test scenarios)
        ↓
Gradio Demo (interactive triage interface)
```

### Key Technical Decisions

1. **Unsloth for Training**: Unsloth provides 1.5x faster training with 60% less VRAM than standard setups. Used on Kaggle's free T4 GPU with 4-bit quantization.

2. **LoRA Fine-tuning**: Low-Rank Adaptation trains only 0.6% of parameters (31M of 5.1B), making fine-tuning feasible on consumer hardware while preserving the base model's general capabilities.

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

The fine-tuned model will be evaluated on 10 clinical test scenarios covering:

- **Classic high-risk patterns**: New-onset diabetes + weight loss + back pain
- **Emergency presentations**: Painless jaundice with weight loss
- **High-risk genetic backgrounds**: BRCA2 carriers with new symptoms
- **Low-risk scenarios**: Young patients with isolated symptoms (must not over-alarm)
- **Subtle presentations**: Mild jaundice, recurrent DVT (Trousseau syndrome)

Evaluation metrics:
- Risk classification accuracy (expected vs actual)
- Urgency matching (emergency/urgent/routine/self-care)
- Required term coverage (must mention key clinical terms)
- Reasoning chain quality (clinically grounded, specific recommendations)

*Benchmark results will be published after training completes on Kaggle.*

## Impact

### The "2am Parent" Scenario

A parent wakes at 2am with a child who's been having stomach pain. They've also noticed the child has been more tired than usual. The model assesses: low risk, likely routine, recommends follow-up with pediatrician.

But imagine a 58-year-old with new diabetes, weight loss, and back pain. The model flags: **high risk, urgent imaging recommended**. That flag could be the difference between Stage 1 (44% survival) and Stage 4 (3%).

### Democratizing Clinical Pattern Recognition

Not everyone has access to a gastroenterologist who will connect "new diabetes + back pain + weight loss" to pancreatic cancer. This tool brings that pattern recognition to anyone with an internet connection.

## Limitations and Vision

### What This Prototype Is Not

Let's be honest about what this isn't: a clinically validated medical device. The model was fine-tuned on only 280 synthetic examples — not real patient records, not expert-annotated clinical notes, not thousands of diverse cases. The training data was generated by an LLM, which means it inherits all the biases, blind spots, and inaccuracies of its source. There has been no retrospective chart review, no prospective study, no IRB approval.

In its current form, **Clover Cancer is a prototype**. It should not — and cannot — replace clinical judgment.

### What This Prototype Is

But here's the part that matters: **it exists**.

A 15-year-old with a laptop and a free Kaggle GPU trained a pancreatic cancer triage model in 7.5 minutes. Two hundred and eighty examples. Zero dollars. And it works well enough to catch the pattern: new-onset diabetes, unexplained weight loss, back pain — the triad that clinicians sometimes miss until it's too late.

Pancreatic cancer kills 137 Americans every single day. Most of them were diagnosed after the window for curative surgery had already closed. If this rough, imperfect, 280-example prototype can flag one case that otherwise would have been dismissed — one missed pattern caught early — then those 7.5 minutes of training were worth it.

**This is the revolution**: not a perfect model, but the *path to one*. Open. Accessible. Democratized. What once required a hospital system, a clinical research team, and a million-dollar budget can now be built by one person with an internet connection. The tools are open. The models are open. The only thing needed is intention.

### The Path Forward

Prototypes graduate. The next iteration will be built on real, de-identified clinical datasets — thousands of cases, expert-annotated, validated against actual patient outcomes. It will be tested by oncologists and deployed alongside existing screening workflows. It will save lives not because of one brilliant training run, but because someone started.

## Future Work

1. **Expand with stronger datasets**: Move from synthetic examples to large-scale, clinically curated datasets sourced from de-identified electronic health records, radiology reports, and pathology notes
2. **Multi-cancer support**: Extend to other cancers with similar early detection gaps
3. **Clinical validation**: Partner with oncologists for formal validation study
4. **Edge deployment**: Quantize for mobile/edge deployment in low-resource settings
5. **Multimodal**: Add imaging analysis (CT scans) for comprehensive assessment

## Author

**Adhyaay Karnwal** — 15-year-old developer, Morris Hills High School

## Links

- **Repository**: github.com/adhyaay-karnwal/clover-cancer
- **Live Demo**: [Coming soon]
- **Model Weights**: [Coming soon]
