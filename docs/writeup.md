# Clover Cancer: Pancreatic Cancer Early Detection with Fine-tuned Gemma 4

## The Problem

Pancreatic cancer is the deadliest major cancer. The 5-year survival rate is just 12% — the lowest of all major cancers. But when caught at Stage 1, survival jumps to 44%. The difference is timing.

Here's what happens in practice. Someone goes to their doctor with back pain. "Probably your desk job," they're told. Someone else gets diagnosed with diabetes at 58. "Let's start metformin." Weight loss? "Good for you." Fatigue? "Get more sleep."

By the time symptoms become obvious — jaundice, severe pain, dramatic weight loss — it's usually Stage III or IV. The window for surgery has closed.

I wanted to see if fine-tuning an LLM could help catch these patterns earlier.

## The Solution

Clover Cancer is a fine-tuned Gemma 4 model that recognizes pancreatic cancer symptom patterns and risk factors. It's not a chatbot — it's a specialized triage system that takes symptoms in natural language and returns a structured assessment: risk level, possible conditions, urgency, and recommended next steps.

### How It Works

Someone describes their symptoms, the model analyzes them against learned clinical patterns, and it outputs a structured assessment.

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

A general LLM can talk about pancreatic cancer, but it won't consistently recognize subtle symptom clusters as high-risk, or provide structured, auditable assessments with specific next steps. Fine-tuning teaches the model to reason through differential diagnoses the way a clinician would.

### Why Gemma 4

I picked Gemma 4 E2B because of its size — 2 billion parameters is small enough to run on a normal laptop, but big enough that it already understands a lot about the world. That's the thing with foundation models: they come pre-loaded with general medical knowledge and reasoning ability from pre-training. What fine-tuning does is specialize that knowledge toward a specific use case. Instead of knowing broadly about "cancer symptoms," the model learns to recognize the specific patterns that indicate pancreatic cancer — and more importantly, to distinguish them from benign conditions that look similar.

A model like Gemma 4 sits at a sweet spot. Too small and it wouldn't have enough reasoning capacity to handle the nuance. Too big and it wouldn't run on consumer hardware. This one works on a MacBook with 24GB of RAM. That matters because it means anyone can run it.

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

### The Setup

My resources were pretty minimal. I trained everything on a MacBook with 24GB of unified memory. For the actual GPU training, I used a few hours of free Kaggle T4 credits — the kind you get just for having a Kaggle account. No cloud bills, no rented GPUs. The entire project cost zero dollars to build.

The training used LoRA, which only trains 0.6% of the model's parameters (31 million out of 5.2 billion total). This is what makes the whole thing feasible on free hardware. With full fine-tuning, you'd need several high-end GPUs. With LoRA, a T4 GPU with 16GB VRAM handles it in about 8 minutes.

### What More Training Would Look Like

With 280 training examples, the model already catches every high-risk pattern I tested it on. But the validation loss doesn't drop much after the first epoch, which tells me the dataset is too small to teach the model anything deeper. More data would make a real difference here.

Think about it this way: with 280 examples, the model learns the broad pattern categories. With 2,800 examples — real clinical cases from electronic health records, curated by oncologists — the model could learn nuance, edge cases, and the statistical rarity that makes pancreatic cancer so hard to catch. It could learn to calibrate its confidence. It could reduce the false positives that currently happen on low-risk cases.

The pipeline scales. The approach scales. The limit right now isn't the model or the technique — it's having access to high-quality clinical data.

## Dataset

- 350 total examples (280 train, 35 val, 35 test)
- 152 pancreatic cancer patterns: symptom clusters, risk factors, differential diagnoses
- 198 general medical examples to prevent catastrophic forgetting
- Sources: NCCN guidelines, Chari et al. (MD Anderson), Johns Hopkins CT study, Frontiers 2025 review

### Pattern Categories

| Category | Examples | Risk Level |
|----------|----------|------------|
| Classic early detection | NOD + weight loss + back pain | High |
| Emergency presentations | Painless jaundice | Critical |
| Genetic risk | BRCA2 + symptoms | High |
| Subtle patterns | Mild jaundice, recurrent DVT | Medium-High |
| Differential diagnosis | Low-risk, must distinguish | Low |

## Results

Three epochs on a Kaggle T4, about 7.5 minutes of training. Evaluated on 10 clinical test scenarios.

### Training Metrics

| Epoch | Training Loss | Validation Loss |
|-------|--------------|----------------|
| 1 | 0.3829 | 2.3861 |
| 2 | 0.1442 | 2.2595 |
| 3 | 0.1096 | 2.2607 |

### Evaluation Results

| Metric | Score |
|--------|-------|
| Pass Rate | 8/10 (80%) |
| Risk Classification | 0.80 |
| Urgency Classification | 0.81 |
| Clinical Term Coverage | 1.00 |
| Reasoning Depth | 0.60 |

The model caught every high-risk pattern I threw at it — the classic new-diabetes-weight-loss-back-pain triad, painless jaundice, BRCA2 carriers, recurrent pancreatitis, even the more subtle Trousseau syndrome presentation with DVT. The two failures were both on low-risk cases where the model mentioned pancreatic cancer when it shouldn't have. For a triage tool, that's the safer direction to be wrong.

The reasoning depth score is the lowest at 0.60. That's because the model sometimes gives short reasoning chains — it says the right thing but doesn't always elaborate. More training data would help here, since longer, more detailed reasoning in the training examples would teach the model to go deeper.

## Impact

Imagine a 58-year-old man walks into a clinic with new diabetes, weight loss, and back pain. Each symptom on its own is easy to dismiss. Together, they're a pattern that warrants investigation. The model flags it as high risk, recommends an abdominal CT and CA 19-9 test. That's the difference between catching pancreatic cancer at Stage 1 and catching it at Stage 4.

Not everyone has access to a gastroenterologist who will connect those dots. This tool makes that pattern recognition available to anyone.

## Limitations

I want to be straightforward about what this is. The model was trained on 280 synthetic examples, not real patient records. Synthetic data inherits biases from the LLM that generated it. There's been no retrospective study, no clinical trial, no IRB approval. This is a prototype, not a medical device, and it should not replace clinical judgment.

But here's what matters: it works well enough to catch patterns that clinicians sometimes miss. I built a pancreatic cancer triage model in under 10 minutes on a free GPU with no budget at all. The technique is proven. What needs to happen now is scale it with real data and clinical validation.

The next version should train on thousands of real clinical cases, validated by oncologists, tested against actual patient outcomes. The model architecture, the LoRA approach, the reasoning framework — all of that transfers. What's missing is access to the data.

## The Bigger Picture

This is what open models enable. Five years ago, building something like this would have required a hospital system's IT department and a research budget. Now it takes a high schooler with a laptop, a Kaggle account, and a reason to try.

Gemma 4 is Apache 2.0 licensed. The training tools are open source. The Kaggle GPU is free. The barriers aren't technical anymore. The only thing you need is a problem worth solving.

## Future Work

1. Scale to thousands of real clinical cases
2. Expand to other cancers with similar early detection gaps
3. Clinical validation with oncologists
4. Mobile deployment for low-resource settings

## Author

**Adhyaay Karnwal** — 15, Morris Hills High School

## Links

- **Repository**: github.com/adhyaay-karnwal/clover-cancer
- **Training Notebook**: kaggle.com/code/adhyaaykarnwal/clover-cancer-gemma-4-fine-tuning
- **Dataset**: kaggle.com/datasets/adhyaaykarnwal/clover-cancer-data
- **LoRA Weights**: kaggle.com/datasets/adhyaaykarnwal/clover-cancer-lora
