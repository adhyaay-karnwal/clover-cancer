# Clover Cancer — Model Card

## Model Details

- **Model name:** Clover Cancer Pancreatic Triage
- **Base model:** google/gemma-4-e2b-it (2B parameters)
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) via PEFT
- **LoRA rank:** 16
- **Trainable parameters:** 31,039,488 (0.60% of total)
- **Hardware:** Apple Silicon Mac (MPS), 24GB RAM
- **Training framework:** PyTorch + transformers + PEFT + TRL

## Intended Use

This model is designed for **research and educational purposes** to demonstrate how fine-tuning can specialize a general language model for medical triage tasks, specifically pancreatic cancer early detection.

**NOT intended for:**
- Clinical decision-making
- Self-diagnosis
- Replacing professional medical advice

## Training Data

- **Source:** Synthetically generated from clinical research and NCCN guidelines
- **Size:** 280 training examples, 35 validation, 35 test
- **Composition:**
  - 152 pancreatic cancer triage patterns (high-risk symptom clusters, genetic risk factors, differential diagnoses)
  - 198 general medical examples (to prevent catastrophic forgetting)
- **Format:** Structured conversations with symptom descriptions → risk assessments

## Model Output

The model produces structured JSON output:
```json
{
  "risk_assessment": "low|medium|high|critical",
  "conditions": [
    {"name": "condition", "likelihood": "...", "reasoning": "..."}
  ],
  "urgency": "self-care|routine|urgent|emergency",
  "recommended_actions": ["action1", "action2"],
  "reasoning_chain": "Detailed clinical reasoning"
}
```

## Evaluation

Tested on 10 clinical scenarios covering classic high-risk patterns, emergency presentations, genetic risk factors, subtle presentations, and low-risk cases.

### Results

| Metric | Score |
|--------|-------|
| Pass Rate | 8/10 (80%) |
| Risk Classification | 0.80 |
| Urgency Classification | 0.81 |
| Clinical Term Coverage | 1.00 |
| Reasoning Depth | 0.60 |

## Limitations

- Trained on synthetic data, not real clinical records
- Small training set (280 examples) — production models need thousands
- Not validated by medical professionals
- May produce inaccurate assessments for edge cases
- Should not be used for actual medical decisions

## Ethical Considerations

- Includes clear disclaimers that this is not a medical device
- Risk assessments are conservative (better to over-refer than miss cancer)
- Reasoning chains are transparent and auditable
- No patient data used — all synthetic

## Citation

```
@software{clover_cancer2026,
  author = {Adhyaay Karnwal},
  title = {Clover Cancer: Pancreatic Cancer Early Detection with Fine-tuned Gemma 4},
  year = {2026},
  url = {https://github.com/adhyaay-karnwal/clover-cancer}
}
```
