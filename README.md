# Clover Cancer

**Pancreatic Cancer Early Detection with Fine-tuned Gemma 4**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Gemma%204%20Good%20Hackathon-blue)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Training Data](https://img.shields.io/badge/Kaggle-Training%20Dataset-green)](https://www.kaggle.com/datasets/adhyaaykarnwal/clover-cancer-data)
[![LoRA Weights](https://img.shields.io/badge/Kaggle-LoRA%20Weights-orange)](https://www.kaggle.com/datasets/adhyaaykarnwal/clover-cancer-lora)
[![Training Notebook](https://img.shields.io/badge/Kaggle-Training%20Notebook-red)](https://www.kaggle.com/code/adhyaaykarnwal/clover-cancer-gemma-4-fine-tuning)
[![Submission Writeup](https://img.shields.io/badge/Kaggle-Submission%20Writeup-purple)](https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/clover-cancer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Clover Cancer](clover-banner.png)

## The Problem

Pancreatic cancer is the deadliest major cancer. The 5-year survival rate is ~12% — the lowest of all major cancers. But when caught at Stage 1, survival jumps to ~44%. The problem? Symptoms are vague and dismissed: back pain, bloating, new-onset diabetes, weight loss. By the time it's obvious, it's usually too late.

**The story**: A 58-year-old with new-onset diabetes and unexplained back pain. The model flags this pattern as high-risk for pancreatic cancer. Stage 1 is found. Survival jumps from 3% to 44%.

## The Solution

A fine-tuned Gemma 4 E2B model trained on pancreatic cancer symptom patterns, risk factors, and clinical guidelines. Not a chatbot — a specialized triage system built to catch what gets missed.

### Key Features

- Identifies high-risk symptom clusters (new-onset diabetes + weight loss + back pain)
- Considers genetic risk factors (BRCA2, Lynch syndrome, family history)
- Provides structured reasoning chains for explainability
- Recommends specific diagnostic pathways (CT, MRI, EUS, CA 19-9)
- Grounded in NCCN clinical guidelines

## Training Results

| Metric | Value |
|--------|-------|
| Base Model | Gemma 4 E2B (2B params) |
| Training Method | LoRA (r=16, alpha=16) via Unsloth |
| Trainable Parameters | 31M (0.60% of total, 5.2B including embeddings) |
| Training Data | 280 training (of 350 total) |
| Training Time | 7.5 min (3 epochs on Kaggle T4) |
| Final Training Loss | 0.1096 |
| Final Validation Loss | 2.261 |

Training log by epoch:

| Epoch | Training Loss | Validation Loss |
|-------|--------------|----------------|
| 1 | 0.3829 | 2.3861 |
| 2 | 0.1442 | 2.2595 |
| 3 | 0.1096 | 2.2607 |

## Evaluation Results

Tested on 10 clinical scenarios covering high-risk classic patterns, emergency presentations, genetic risk, subtle presentations, and low-risk cases:

| Metric | Score |
|--------|-------|
| Pass Rate | 8/10 (80%) |
| Risk Classification Accuracy | 0.80 |
| Urgency Classification | 0.81 |
| Clinical Term Coverage | 1.00 |
| Reasoning Depth | 0.60 |

### Prerequisites

- Python 3.10+
- 16GB+ RAM (for CPU/MPS inference)
- NVIDIA GPU with 16GB+ VRAM (for Unsloth/fast inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/adhyaay-karnwal/clover-cancer.git
cd clover-cancer

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

The LoRA adapter weights are hosted on Kaggle. Download them:

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (get key from kaggle.com/settings)
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download LoRA adapters
kaggle datasets download adhyaaykarnwal/clover-cancer-lora -p outputs/clover-cancer-lora --unzip
```

Or download manually from: [kaggle.com/datasets/adhyaaykarnwal/clover-cancer-lora](https://www.kaggle.com/datasets/adhyaaykarnwal/clover-cancer-lora)

### Training Data

The training data is available at: [kaggle.com/datasets/adhyaaykarnwal/clover-cancer-data](https://www.kaggle.com/datasets/adhyaaykarnwal/clover-cancer-data)

### Run the Demo

```bash
# Start the Gradio demo
python app/main.py
```

Open http://localhost:7860 in your browser.

### Run Without Model (Mock Mode)

The demo works without the model in mock mode for testing:

```bash
python app/main.py
```

## Architecture

```
User Input (symptoms + risk factors)
    ↓
Fine-tuned Gemma 4 E2B (Unsloth + LoRA)
    ↓
Structured Output (JSON)
├── Risk level (low/medium/high/critical)
├── Possible conditions (ranked with likelihood)
├── Reasoning chain (clinical reasoning)
└── Recommended actions (imaging, tests, referrals)
```

## Kaggle Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Training Dataset** | 350 structured conversation examples (train/val/test) | [kaggle.com/datasets/adhyaaykarnwal/clover-cancer-data](https://www.kaggle.com/datasets/adhyaaykarnwal/clover-cancer-data) |
| **LoRA Adapter Weights** | Fine-tuned Gemma 4 E2B weights (119MB) | [kaggle.com/datasets/adhyaaykarnwal/clover-cancer-lora](https://www.kaggle.com/datasets/adhyaaykarnwal/clover-cancer-lora) |
| **Training Notebook** | Kaggle T4 GPU training pipeline with Unsloth | [kaggle.com/code/adhyaaykarnwal/clover-cancer-gemma-4-fine-tuning](https://www.kaggle.com/code/adhyaaykarnwal/clover-cancer-gemma-4-fine-tuning) |
| **Submission Writeup** | Official hackathon submission writeup | [kaggle.com/competitions/gemma-4-good-hackathon/writeups/clover-cancer](https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/clover-cancer) |

## Repository Structure

```
clover-cancer/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── app/
│   ├── inference.py           # Model loading and inference
│   └── main.py                # Gradio web demo
├── data/
│   ├── processed/             # Training data (train/val/test)
│   └── dataset_card.md        # Dataset documentation
├── train/
│   ├── config.yaml            # Training hyperparameters
│   ├── prepare_data.py        # Dataset generation script
│   ├── fine_tune.py           # Local training script
│   └── evaluate.py            # Evaluation script
├── notebooks/
│   └── train_clover_cancer.ipynb  # Kaggle training notebook
├── docs/
│   ├── writeup.md             # Project writeup
│   ├── model_card.md          # Model documentation
│   ├── medical_reference.md   # Pancreatic cancer knowledge base
│   └── video_script.md        # Video script
└── outputs/
    └── clover-cancer-lora/    # LoRA adapter weights (download separately)
```

## Training on Kaggle

To reproduce the training:

1. Open [kaggle.com/code](https://www.kaggle.com/code)
2. Create new notebook, enable **GPU T4** accelerator
3. Add dataset: `adhyaaykarnwal/clover-cancer-data`
4. Upload `notebooks/train_clover_cancer.ipynb`
5. Run all (~7.5 minutes)

## Medical References

- [NCCN Guidelines v2.2026](https://www.nccn.org/guidelines/guidelines-detail?category=2&id=1508) — Pancreatic cancer screening and management
- Chari et al. (MD Anderson) — New-onset diabetes as early PC predictor
- Ahmed et al. (PMID 38782784) — CT missed PC signs
- Frontiers 2025 (DOI 10.3389/fgstr.2025.1645459) — Comprehensive PC review

## Prize Tracks

| Track | Prize | Why We Qualify |
|-------|-------|----------------|
| **Unsloth** | $10K | Fine-tuned Gemma 4 E2B using Unsloth on Kaggle T4 |
| **Health & Sciences** | $10K | Democratizing pancreatic cancer early detection |
| **Safety & Trust** | $10K | Explainable reasoning chains grounded in NCCN guidelines |
| **Main Track** | Up to $50K | Compelling story + real technical execution |

## What This Really Means

I want to be upfront: this is a prototype. I trained it on 280 synthetic examples in 7.5 minutes on a free Kaggle GPU. That is not enough data to be clinically reliable, and I haven't validated it against real patient outcomes. It makes mistakes. It has blind spots. What the evaluation shows is promising: 80% pass rate across 10 clinical scenarios, 100% term coverage, and it caught every high-risk cancer pattern correctly. But it also sometimes over-flags low-risk cases -- which is the safer direction for a triage tool, but not perfect.

Pancreatic cancer kills 137 Americans every day. Most are diagnosed too late because the symptoms are easy to dismiss. If this rough prototype helps even one person catch a pattern that would have been missed, then it matters.

This is what open models make possible. A high schooler with a laptop can now build something that once required a hospital system and a million dollar budget. The next version will train on thousands of real clinical cases, validated by people who save lives every day. But you have to start somewhere. This is where I started.

## Author

**Adhyaay Karnwal** - Morris Hills Highschool (Class of 2028)

- GitHub: [@adhyaay-karnwal](https://github.com/adhyaay-karnwal)
- X: [@AdhyaayK](https://x.com/AdhyaayK)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Disclaimer

This is an AI research tool for educational purposes. It is NOT a medical device. Always consult a healthcare provider for medical decisions.
