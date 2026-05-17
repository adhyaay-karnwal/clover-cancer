# Clover Cancer: Pancreatic Cancer Early Detection with Fine-tuned Gemma 4

## Motivation

Pancreatic cancer is the deadliest major cancer. The 5-year survival rate is just 12% — the lowest of all major cancers. But when caught at Stage 1, survival jumps to 44%. The difference is timing.

The challenge is that early symptoms are vague: back pain, new-onset diabetes, weight loss, fatigue. Each on its own is easily dismissed. By the time symptoms become obvious — jaundice, severe pain, dramatic weight loss — the window for curative surgery has usually closed.

A general-purpose LLM can talk about pancreatic cancer, but it won't consistently recognize subtle symptom clusters as high-risk, or provide structured, auditable assessments. I wanted to see if fine-tuning could close that gap — teaching a model to reason through differential diagnoses the way a clinician would, rather than just reciting facts.

## Solution Approach

Gemma 4 E2B represents a sweet spot for this kind of task. At 2 billion parameters, it's small enough to run on consumer hardware — my MacBook with 24GB of unified memory handles inference without issue. But it's large enough that its pre-training provides a strong foundation in medical knowledge and general reasoning. The key insight is that foundation models already understand a lot about the world. What fine-tuning does is specialize that knowledge toward a specific use case. Instead of knowing broadly about "cancer symptoms," the model learns to recognize the specific patterns that warrant investigation — and more importantly, to distinguish them from benign conditions that look similar.

My goal extended beyond creating a demo: I wanted to build something that could actually catch patterns that get missed in clinical practice. Convenience matters here — if the tool isn't easy to use, it doesn't matter how accurate it is. That's why the output is structured JSON with specific next steps rather than free text. The user doesn't need to interpret a paragraph — they get risk level, urgency, possible conditions, and recommended actions.

### Why Fine-tuning, Not Just Prompting

A base Gemma 4 can discuss pancreatic cancer knowledgeably. But without fine-tuning, it won't consistently:

- Recognize that new-onset diabetes at 58 + unexplained weight loss + back pain is a high-risk pattern
- Provide structured, parseable assessments rather than narrative text
- Match urgency to clinical severity appropriately
- Recommend specific diagnostic pathways (CT with pancreatic protocol, CA 19-9, EUS)

Fine-tuning teaches the model to think in a structured way — not just what to say, but how to say it, and when to escalate.

## Development Process

### Model Selection

I chose Gemma 4 E2B over larger models (like 7B or 27B) for a few reasons:

- **Hardware constraints**: At 2B parameters, the 4-bit quantized version fits in about 1.5GB of VRAM. This means it runs on a free Kaggle T4 (16GB) with plenty of headroom, and also runs on my MacBook's MPS backend for local inference.
- **Apache 2.0 license**: Fully open for research and modification.
- **Reasoning capability**: Despite its size, Gemma 4's pre-training provides solid medical knowledge. The model already understands concepts like differential diagnosis, risk stratification, and clinical guidelines — it just needs to be taught how to apply them consistently.

### Training Pipeline

The training was done on Kaggle's free T4 GPU using Unsloth for efficient LoRA fine-tuning:

```
Medical Research (NCCN, PubMed, Frontiers)
        ↓
Dataset Curation (350 structured examples)
        ↓
LoRA Fine-tuning (Unsloth on Kaggle T4, rank 16, 0.6% trainable)
        ↓
Challenges: tokenizer alignment, multimodal format, MPS compatibility
        ↓
Evaluation (10 clinical test scenarios)
        ↓
Gradio Demo (interactive triage interface)
```

I had minimal resources: a MacBook for development and testing, and a few hours of free Kaggle GPU credits for training. The entire project cost nothing to build.

### Challenges I Faced

**Token alignment during fine-tuning**

When I first ran training on the notebook, the tokenizer's PAD/BOS/EOS tokens didn't match what the model expected. The tokenizer had custom token values that differed from the model config and generation config. I had to sync all three — updating the model config and generation config with the tokenizer's values (including setting bos_token_id to 2). Without this alignment, the model would generate incoherent output because it was decoding with mismatched special tokens.

**Gemma 4 multimodal content format**

The Gemma 4 processor expects content in a specific multimodal format: `[{"type": "text", "text": "..."}]` rather than plain strings. This wasn't immediately obvious from the documentation. The initial training code passed plain strings to `apply_chat_template`, which worked for the template but caused subtle issues during generation. Switching to explicit content blocks resolved this.

**MPS compatibility for local inference**

Training happened on Kaggle's CUDA GPU, but I wanted to run inference locally on my MacBook's MPS backend. The fine-tuning script initially used `device_map="mps"`, which newer versions of transformers don't fully support for Gemma 4. I had to remove the device_map parameter and explicitly move the model to MPS after loading. Similarly, the adamw_8bit optimizer used by Unsloth isn't available on MPS, so I had to fall back to adamw_torch for local training runs.

**Format verification during training**

Unsloth's `train_on_responses_only` needs to match the correct instruction and response markers. For Gemma 4, the markers are `<|turn|>user` and `<|turn|>model` rather than the `<start_of_turn>` format used by other Gemma versions. Getting this wrong means the model trains on the user input instead of just the responses, which defeats the purpose of supervised fine-tuning.

**Inference parsing**

The model generates JSON wrapped in natural language, so parsing the structured output required robust extraction. The initial approach used simple JSON parsing, but the model sometimes prepends or appends explanatory text around the JSON block. I implemented balanced-brace extraction to find the outermost JSON object regardless of surrounding text.

### Dataset

The training data consists of 350 structured conversation examples (280 train, 35 val, 35 test):

- **152 pancreatic cancer patterns**: symptom clusters, risk factors, and differential diagnoses covering a range of presentations from early-stage to emergency
- **198 general medical examples**: common conditions that could be confused with pancreatic cancer, preventing the model from developing a "when in doubt, say cancer" bias
- **Sources**: NCCN clinical practice guidelines, Chari et al. (MD Anderson) on new-onset diabetes as early predictor, Johns Hopkins CT study on missed PC signs, and the Frontiers 2025 comprehensive review

| Category | Examples | Risk Level |
|----------|----------|------------|
| Classic early detection | NOD + weight loss + back pain | High |
| Emergency presentations | Painless jaundice | Critical |
| Genetic risk | BRCA2 + symptoms | High |
| Subtle patterns | Mild jaundice, recurrent DVT | Medium-High |
| Differential diagnosis | Low-risk, must distinguish | Low |

## Results

Three epochs on a Kaggle T4, about 7.5 minutes of training. Evaluated on 10 held-out clinical test scenarios.

### Training Metrics

| Epoch | Training Loss | Validation Loss |
|-------|--------------|----------------|
| 1 | 0.3829 | 2.3861 |
| 2 | 0.1442 | 2.2595 |
| 3 | 0.1096 | 2.2607 |

The training loss drops cleanly through all three epochs, but the validation loss plateaus after epoch 1. This is the expected pattern for a small dataset — the model memorizes the training patterns rather than learning deeper generalizations. More data would directly address this.

### Evaluation Results

| Metric | Score |
|--------|-------|
| Pass Rate | 8/10 (80%) |
| Risk Classification | 0.80 |
| Urgency Classification | 0.81 |
| Clinical Term Coverage | 1.00 |
| Reasoning Depth | 0.60 |

The model caught every high-risk pattern I tested: the classic new-diabetes-weight-loss-back-pain triad, painless jaundice, BRCA2 carriers with symptoms, recurrent pancreatitis, and Trousseau syndrome (DVT with abdominal symptoms). The two failures were both on low-risk cases where the model mentioned pancreatic cancer when it shouldn't have. For a triage tool, this is the safer direction to be wrong.

The reasoning depth score is the weakest at 0.60. The model gives accurate assessments but often with short reasoning chains. This reflects the training data — 280 examples doesn't give the model enough varied examples of detailed clinical reasoning. Expanding the dataset with longer, more thorough reasoning examples would likely improve this.

### What More Training Would Enable

With 280 examples, the model learns broad pattern categories. With 2,800 examples — real clinical cases from electronic health records, curated by oncologists — the model could learn:

- **Calibration**: Better confidence estimation for borderline cases
- **Edge cases**: Rarer presentations that currently aren't covered
- **Nuance**: Distinguishing between similar presentations with different underlying causes
- **Reduced false positives**: Learning finer-grained distinctions so low-risk cases don't get over-escalated

The approach scales. The pipeline is the same regardless of dataset size. The limiting factor isn't the model or the technique — it's access to high-quality clinical data.

## Limitations

This is not a clinically validated medical device. The model was trained on 280 synthetic examples, not real patient records. Synthetic data inherits biases from the LLM that generated it. There has been no retrospective study, no clinical trial, no IRB approval.

But the approach is proven. A pancreatic cancer triage model was built in 7.5 minutes on a free GPU with zero budget. It catches the patterns it should catch. The next step is scale — thousands of real clinical cases, validation against patient outcomes, and eventually, clinical deployment.

## The Bigger Picture

Five years ago, building a specialized medical triage model would have required a hospital IT department and a significant research budget. Today, it took a high schooler with a laptop, a Kaggle account, and a willingness to try.

Gemma 4 is Apache 2.0 licensed. Unsloth is open source. Kaggle provides free GPU time. The barriers aren't technical anymore. What's needed is the will to apply these tools to problems that matter.

## Author

**Adhyaay Karnwal** — 15, Morris Hills High School

## Links

- **Repository**: github.com/adhyaay-karnwal/clover-cancer
- **Training Notebook**: kaggle.com/code/adhyaaykarnwal/clover-cancer-gemma-4-fine-tuning
- **Dataset**: kaggle.com/datasets/adhyaaykarnwal/clover-cancer-data
- **LoRA Weights**: kaggle.com/datasets/adhyaaykarnwal/clover-cancer-lora
