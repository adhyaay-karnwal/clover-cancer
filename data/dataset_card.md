# Pancreatic Cancer Triage Dataset

## Overview

A structured dataset for training AI models on pancreatic cancer symptom triage and early detection.

## Dataset Description

This dataset contains structured training examples that map patient presentations (symptoms, demographics, risk factors) to clinical assessments including risk levels, differential diagnoses, urgency classifications, and recommended actions.

## Source Data

- NCCN Clinical Practice Guidelines for Pancreatic Cancer
- PubMed case reports and clinical presentations
- SEER epidemiological patterns
- Clinical symptom checklists from oncology guidelines

## Data Format

Each example follows this structure:

```json
{
  "input": {
    "symptoms": [...],
    "age": int,
    "risk_factors": [...],
    "duration": "string"
  },
  "output": {
    "risk_assessment": "low|medium|high|critical",
    "conditions": [...],
    "urgency": "self-care|routine|urgent|emergency",
    "recommended_actions": [...],
    "reasoning_chain": "string"
  }
}
```

## Risk Levels

- **Low**: No concerning patterns, routine screening appropriate
- **Medium**: Some risk factors present, consider screening
- **High**: Pattern consistent with early presentation, imaging recommended
- **Critical**: Urgent referral needed, immediate evaluation required

## Limitations

- This dataset is for research and educational purposes only
- Not intended for clinical decision-making
- Synthetic examples generated from medical knowledge should be validated by clinicians
- Does not replace professional medical advice

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/processed/train.json")
```

## License

MIT
