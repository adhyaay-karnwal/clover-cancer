"""
Clover Cancer — Gemma 4 Fine-tuning with Unsloth

Fine-tunes Google's Gemma 4 E2B model on the pancreatic cancer triage dataset
using Unsloth for efficient LoRA training.

Based on Unsloth's official Gemma 4 fine-tuning guide:
https://unsloth.ai/docs/models/gemma-4/train

Hardware: Apple Silicon Mac, 24GB RAM
Model: Gemma 4 E2B (2B params) — trains on 8GB VRAM with Unsloth

Author: Adhyaay Karnwal
"""

import os
import json
import yaml
import torch
from pathlib import Path

# Set environment variables for MPS (Apple Silicon)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def load_config(config_path: str = "train/config.yaml") -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(data_path: str) -> list:
    """Load training data from JSON file."""
    with open(data_path) as f:
        return json.load(f)


def setup_model(config: dict):
    """Load Gemma 4 model with Unsloth and configure LoRA."""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    print(f"Loading model: {config['model']['name']}...")

    model, tokenizer = FastModel.from_pretrained(
        model_name=config["model"]["name"],
        dtype=None,  # Auto-detect
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        full_finetuning=False,
    )

    # Set up chat template for Gemma 4
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4",  # Non-thinking mode for E2B
    )

    # Configure LoRA
    print("Configuring LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Text-only
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,

        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        random_state=3407,
    )

    return model, tokenizer


def prepare_dataset(tokenizer, data_path: str):
    """Format dataset for Unsloth SFT training."""
    from unsloth.chat_templates import standardize_data_formats

    print(f"Loading dataset from {data_path}...")
    data = load_dataset(data_path)

    # Convert to standard format
    dataset = standardize_data_formats({"conversations": data})

    # Apply chat template
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            ).removeprefix('<bos>')
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"Prepared {len(formatted_dataset)} training examples")

    return formatted_dataset


def create_trainer(model, tokenizer, dataset, config: dict):
    """Create SFT trainer with configured parameters."""
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    training_config = config["training"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,  # Can add validation later
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            warmup_ratio=training_config["warmup_ratio"],
            num_train_epochs=training_config["num_train_epochs"],
            learning_rate=training_config["learning_rate"],
            logging_steps=training_config["logging_steps"],
            save_strategy=training_config["save_strategy"],
            optim=training_config["optim"],
            weight_decay=0.001,
            lr_scheduler_type=training_config["lr_scheduler_type"],
            seed=training_config["seed"],
            max_grad_norm=training_config["max_grad_norm"],
            output_dir=training_config["output_dir"],
            report_to="none",  # No wandb/logging
            bf16=training_config["bf16"],
            fp16=training_config["fp16"],
        ),
    )

    # Train on assistant responses only (not system/user prompts)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn|>user\n",
        response_part="<|turn|>model\n",
    )

    return trainer


def train(trainer):
    """Run training and return stats."""
    print("\n=== Starting Training ===")
    print(f"  Batch size: {trainer.args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {trainer.args.gradient_accumulation_steps}")
    print(f"  Effective batch: {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")
    print(f"  Learning rate: {trainer.args.learning_rate}")
    print(f"  Epochs: {trainer.args.num_train_epochs}")
    print()

    stats = trainer.train()

    print("\n=== Training Complete ===")
    print(f"  Total steps: {stats.global_step}")
    print(f"  Training loss: {stats.training_loss:.4f}")

    return stats


def save_model(model, tokenizer, config: dict):
    """Save the fine-tuned model."""
    output_dir = config["training"]["output_dir"]

    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully.")

    # Optionally push to HuggingFace Hub
    if config["output"].get("push_to_hub"):
        hub_model_id = config["output"]["hub_model_id"]
        print(f"Pushing to HuggingFace Hub: {hub_model_id}...")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print("Pushed to Hub successfully.")


def export_gguf(model, tokenizer, output_dir: str = "models/gguf"):
    """Export model to GGUF format for Ollama/llama.cpp deployment."""
    print(f"\nExporting to GGUF format at {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method="q4_k_m"
    )
    print(f"GGUF model saved to {output_dir}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Clover Cancer — Gemma 4 Fine-tuning Pipeline")
    print("=" * 60)

    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Device: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Device: CUDA ({torch.cuda.get_device_name()})")
    else:
        device = "cpu"
        print(f"Device: CPU (training will be slow)")

    # Load config
    config = load_config()
    print(f"Model: {config['model']['name']}")
    print(f"Max sequence length: {config['model']['max_seq_length']}")

    # Setup model
    model, tokenizer = setup_model(config)

    # Prepare dataset
    dataset = prepare_dataset(tokenizer, config["data"]["train_file"])

    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset, config)

    # Train
    stats = train(trainer)

    # Save
    save_model(model, tokenizer, config)

    # Export GGUF for deployment
    try:
        export_gguf(model, tokenizer)
    except Exception as e:
        print(f"GGUF export failed (non-critical): {e}")

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
