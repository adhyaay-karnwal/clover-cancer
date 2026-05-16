# Pancreatic Cancer Triage Dataset

## Overview

A structured dataset for training AI models on pancreatic cancer symptom triage and early detection.

## Dataset Description

This dataset contains structured training examples that map patient presentations (symptoms, demographics, risk factors) to clinical assessments including risk levels, differential diagnoses, urgency classifications, and recommended actions.

### Statistics

| Split | Count |
|-------|-------|
| Training | 280 |
| Validation | 35 |
| Test | 35 |
| **Total** | **350** |
| PC-specific patterns | 152 |
| General medical examples | 198 |

## Source Data

- NCCN Clinical Practice Guidelines for Pancreatic Cancer
- PubMed case reports and clinical presentations
- SEER epidemiological patterns
- Clinical symptom checklists from oncology guidelines

## Data Format

Each example is a conversation with system/user/assistant roles, formatted for chat-based fine-tuning:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a medical triage AI specialized in pancreatic cancer early detection..."
    },
    {
      "role": "user",
      "content": "Please assess my symptoms and tell me what might be going on:\n\nI'm a 58-year-old male..."
    },
    {
      "role": "assistant",
      "content": "{\"risk_assessment\": \"high\", \"conditions\": [...], \"urgency\": \"urgent\", ...}"
    }
  ]
}
```

The assistant response is structured JSON containing:
- `risk_assessment`: low | medium | high | critical
- `conditions`: ranked differential diagnoses with reasoning
- `urgency`: self-care | routine | urgent | emergency
- `recommended_actions`: specific diagnostic steps
- `reasoning_chain`: clinical reasoning explanation

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
