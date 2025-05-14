"""Training utilities for fine-tuning models."""

import json
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from ..utils.logging import get_logger

logger = get_logger(__name__)

def prepare_model(
    model_name: str,
    lora_config: Optional[dict] = None,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> tuple:
    """Prepare model and tokenizer for training.
    
    Args:
        model_name: HuggingFace model name/path
        lora_config: LoRA configuration dict
        load_in_4bit: Whether to quantize model to 4-bit
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Quantization config
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Apply LoRA if configured
    if lora_config:
        logger.info("Applying LoRA configuration")
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
            
        lora = LoraConfig(
            r=lora_config.get("r", 16),
            alpha=lora_config.get("alpha", 32),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            dropout=lora_config.get("dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    
    return model, tokenizer

def train(
    model_name: str,
    train_data_path: Path,
    output_dir: Path,
    training_config: dict,
    lora_config: Optional[dict] = None,
) -> None:
    """Train or fine-tune a model.
    
    Args:
        model_name: HuggingFace model name/path
        train_data_path: Path to training data (JSONL)
        output_dir: Directory to save outputs
        training_config: Training hyperparameters
        lora_config: Optional LoRA configuration
    """
    logger.info("Starting training")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = prepare_model(
        model_name,
        lora_config=lora_config,
        load_in_4bit=training_config.get("load_in_4bit", False),
    )
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(train_data_path))
    
    # Tokenization function
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=training_config.get("max_length", 2048),
        )
    
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get("epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 2e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 100),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 200),
        eval_steps=training_config.get("eval_steps", 200),
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to=training_config.get("report_to", "wandb"),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "training_config": training_config,
            "lora_config": lora_config,
        }, f, indent=2)
        
    logger.info(f"Training complete. Model saved to {output_dir}/final") 