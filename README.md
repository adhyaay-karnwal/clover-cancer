# Clover Cancer

**Pancreatic Cancer Early Detection with Fine-tuned Gemma 4**

## The Problem

Pancreatic cancer is the deadliest major cancer. The 5-year survival rate is ~12% — the lowest of all major cancers. But when caught at Stage 1, survival jumps to ~44%. The problem? Symptoms are vague and dismissed: back pain, bloating, new-onset diabetes, weight loss. By the time it's obvious, it's usually too late.

## The Solution

A fine-tuned Gemma 4 model trained on pancreatic cancer symptom patterns, risk factors, and clinical guidelines. Not a chatbot — a specialized triage system built to catch what gets missed.

## How It Works

1. **Input**: Symptoms described in plain language + risk factors (age, family history, smoking, diabetes)
2. **Processing**: Fine-tuned Gemma 4 E2B with LoRA adapters via Unsloth
3. **Output**: Risk assessment, possible conditions with reasoning, urgency level, recommended next steps

## Architecture

```
User Input (symptoms + risk factors)
    ↓
Fine-tuned Gemma 4 E2B (Unsloth + LoRA)
    ↓
Structured Output
├── Risk level (low/medium/high/critical)
├── Possible conditions (ranked)
├── Reasoning chain
└── Recommended actions (imaging, tests, referrals)
```

## Prize Tracks

| Track | Prize | Why |
|-------|-------|-----|
| **Unsloth** | $10K | Best fine-tuned Gemma 4 for a specific task |
| **Health & Sciences** | $10K | Democratizing cancer early detection |
| **Safety & Trust** | $10K | Explainable reasoning, grounded in clinical guidelines |
| **Main Track** | Up to $50K | Compelling story + real technical depth |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python app/main.py
```

## Repository Structure

```
clover-cancer/
├── data/           # Training data and documentation
├── train/          # Training scripts and configuration
├── app/            # Gradio demo application
├── docs/           # Documentation (medical reference, model card)
├── notebooks/      # Analysis and evaluation notebooks
└── tests/          # Smoke tests
```

## Author

**Adhyaay Karnwal** - Morris Hills High School (Sophomore)

## License

MIT
