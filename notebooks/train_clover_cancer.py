"""
Clover Cancer — Kaggle Training Notebook

Fine-tunes Gemma 4 E2B on the pancreatic cancer triage dataset using Unsloth.
Runs on Kaggle's free T4 GPU (16GB VRAM).

Upload this as a Kaggle notebook, add the dataset files, and run.

Author: Adhyaay Karnwal
"""

# %% [markdown]
# # Clover Cancer — Gemma 4 Fine-tuning
#
# Fine-tunes Google's Gemma 4 E2B model on pancreatic cancer triage data
# using Unsloth for efficient LoRA training on T4 GPU.
#
# ## Setup

# %%
%%capture
import torch
major_version, minor_version = torch.cuda.get_device_capability()
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
if major_version >= 8:
    !pip install --no-deps packaging ninja einops "flash-attn>=2.6.3"

# %%
from unsloth import FastLanguageModel
import torch
import json
import os

# %%
# Configuration
MODEL_NAME = "unsloth/gemma-4-e2b-it"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # T4 supports 4-bit quantization

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8  # Effective batch = 16

# %%
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Load Model

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=LOAD_IN_4BIT,
)

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# %% [markdown]
# ## Load Dataset

# %%
# Load training data (upload data/processed/train.json to Kaggle)
with open("/kaggle/input/clover-cancer-data/train.json") as f:
    train_data = json.load(f)

with open("/kaggle/input/clover-cancer-data/val.json") as f:
    val_data = json.load(f)

print(f"Train: {len(train_data)} examples")
print(f"Val: {len(val_data)} examples")

# %%
# Format for chat template
from datasets import Dataset

def format_conversations(examples):
    """Format conversations for the model's chat template."""
    texts = []
    for convo in examples["conversations"]:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        texts.append(text)
    return {"text": texts}

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

train_dataset = train_dataset.map(format_conversations, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(format_conversations, batched=True, remove_columns=val_dataset.column_names)

print(f"Train formatted: {len(train_dataset)} examples")
print(f"Val formatted: {len(val_dataset)} examples")

# %% [markdown]
# ## Train

# %%
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.05,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
    ),
)

# Train on assistant responses only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

# %%
# Check memory usage
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU: {gpu_stats.name}")
print(f"Max memory: {max_memory} GB")
print(f"Starting memory: {start_gpu_memory} GB")

# %%
stats = trainer.train()

# %%
# Print training stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"\nTraining complete!")
print(f"  Steps: {stats.global_step}")
print(f"  Loss: {stats.training_loss:.4f}")
print(f"  Time: {stats.metrics['train_runtime']:.0f}s")
print(f"  Peak memory: {used_memory}/{max_memory} GB ({100*used_memory/max_memory:.1f}%)")

# %% [markdown]
# ## Save Model

# %%
# Save LoRA adapters
model.save_pretrained("clover-cancer-lora")
tokenizer.save_pretrained("clover-cancer-lora")
print("LoRA adapters saved to clover-cancer-lora/")

# %%
# Save merged model (16-bit) for deployment
model.save_pretrained_merged(
    "clover-cancer-merged",
    tokenizer,
    save_method="merged_16bit",
)
print("Merged model saved to clover-cancer-merged/")

# %%
# Export to GGUF for Ollama deployment
model.save_pretrained_gguf(
    "clover-cancer-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF model saved to clover-cancer-gguf/")

# %% [markdown]
# ## Evaluate

# %%
# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "Classic high-risk",
        "input": "I'm a 58-year-old male. I was just diagnosed with diabetes last month. I've lost about 15 pounds without trying. I have this deep ache in my mid-back that won't go away.",
        "expected": "high"
    },
    {
        "name": "Emergency jaundice",
        "input": "I'm a 67-year-old female. My skin turned yellow. I've lost about 10 pounds. My urine is really dark.",
        "expected": "critical"
    },
    {
        "name": "Low-risk back pain",
        "input": "I'm a 42-year-old male. I've had back pain for 2 weeks. I sit at a desk all day. No other symptoms.",
        "expected": "low"
    },
]

# %%
FastLanguageModel.for_inference(model)

SYSTEM_PROMPT = """You are a medical triage AI specialized in pancreatic cancer early detection.
Analyze symptoms and respond with JSON:
{"risk_assessment": "low|medium|high|critical", "conditions": [...], "urgency": "...", "recommended_actions": [...], "reasoning_chain": "..."}"""

for scenario in TEST_SCENARIOS:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Assess my symptoms:\n\n{scenario['input']}"}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario['name']}")
    print(f"Expected risk: {scenario['expected']}")
    print(f"Response:\n{response}")
