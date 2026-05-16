"""
Clover Cancer — Gemma 4 Fine-tuning with PyTorch MPS

Fine-tunes Google's Gemma 4 E2B model on the pancreatic cancer triage dataset
using transformers + PEFT for LoRA training on Apple Silicon.

Hardware: Apple Silicon Mac, 24GB RAM
Model: Gemma 4 E2B (2B params) — fits in 24GB in float16

Author: Adhyaay Karnwal
"""

import os
import json
import yaml
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def load_config(config_path: str = "train/config.yaml") -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(data_path: str) -> Dataset:
    """Load training data from JSON file."""
    with open(data_path) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_conversations(example, tokenizer):
    """Format a conversation example for the model."""
    messages = example["conversations"]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def unwrap_clippable_linear(model):
    """Replace Gemma4ClippableLinear wrappers with their inner Linear modules.

    PEFT doesn't recognize Gemma4ClippableLinear as a valid target.
    This unwraps them so LoRA can be applied to the inner Linear layers.
    """
    replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, 'linear') and not isinstance(module, torch.nn.Linear):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], module.linear)
            replaced += 1
    print(f"  Unwrapped {replaced} Gemma4ClippableLinear modules")
    return model


def setup_model(config: dict):
    """Load Gemma 4 model in float16 and configure LoRA."""
    model_name = config["model"]["name"]
    print(f"Loading model: {model_name}...")
    print(f"Device: MPS (Apple Silicon)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in float16 (no quantization — MPS doesn't support bitsandbytes)
    # E2B is ~4GB in float16, fits easily in 24GB unified memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="mps",
    )

    # Unwrap Gemma4ClippableLinear for PEFT compatibility
    model = unwrap_clippable_linear(model)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Configure LoRA
    print("Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        target_modules=config["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_dataset(tokenizer, data_path: str) -> Dataset:
    """Format dataset for SFT training."""
    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset(data_path)

    # Format conversations
    def format_fn(example):
        return format_conversations(example, tokenizer)

    formatted_dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
    print(f"Prepared {len(formatted_dataset)} training examples")

    return formatted_dataset


def create_trainer(model, tokenizer, train_dataset, val_dataset, config: dict):
    """Create SFT trainer with configured parameters."""
    training_config = config["training"]

    training_args = SFTConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        warmup_ratio=training_config["warmup_ratio"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        logging_steps=training_config["logging_steps"],
        save_strategy=training_config["save_strategy"],
        eval_strategy=training_config["eval_strategy"],
        optim=training_config["optim"],
        max_grad_norm=training_config["max_grad_norm"],
        seed=training_config["seed"],
        bf16=False,
        fp16=True,  # Use float16 for MPS
        report_to="none",
        max_length=config["model"]["max_seq_length"],
        dataset_text_field="text",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    return trainer


def train(trainer):
    """Run training and return stats."""
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"  Batch size: {trainer.args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {trainer.args.gradient_accumulation_steps}")
    print(f"  Effective batch: {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")
    print(f"  Learning rate: {trainer.args.learning_rate}")
    print(f"  Epochs: {trainer.args.num_train_epochs}")
    print(f"  Device: MPS (Apple Silicon)")
    print()

    stats = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Total steps: {stats.global_step}")
    print(f"  Training loss: {stats.training_loss:.4f}")

    return stats


def save_model(model, tokenizer, config: dict):
    """Save the fine-tuned model."""
    output_dir = config["training"]["output_dir"]

    print(f"\nSaving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config
    with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
        yaml.dump(config, f)

    print("Model saved successfully.")
    print(f"  LoRA adapters: {output_dir}/adapter_model.safetensors")
    print(f"  Tokenizer: {output_dir}/tokenizer.json")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Clover Cancer — Gemma 4 Fine-tuning Pipeline")
    print("=" * 60)

    # Check device
    if torch.backends.mps.is_available():
        print(f"Device: Apple Silicon (MPS)")
        print(f"PyTorch: {torch.__version__}")
    else:
        print("WARNING: MPS not available, using CPU (will be slow)")

    # Load config
    config = load_config()
    print(f"Model: {config['model']['name']}")
    print(f"Max sequence length: {config['model']['max_seq_length']}")

    # Setup model
    model, tokenizer = setup_model(config)

    # Prepare datasets
    train_dataset = prepare_dataset(tokenizer, config["data"]["train_file"])
    val_dataset = prepare_dataset(tokenizer, config["data"]["val_file"])

    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, config)

    # Train
    stats = train(trainer)

    # Save
    save_model(model, tokenizer, config)

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Evaluate the model: python train/evaluate.py")
    print("  2. Run the demo: python app/main.py")
    print("  3. Export to GGUF for Ollama deployment")


if __name__ == "__main__":
    main()
