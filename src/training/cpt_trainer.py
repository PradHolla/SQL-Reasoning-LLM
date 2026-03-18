import os
import torch
from model_loader import load_base_model

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

def train_cpt():
    print("Initializing CPT Training Run...")
    
    # Load the model and tokenizer with LoRA adapters
    model, tokenizer = load_base_model("Qwen/Qwen2.5-0.5B", max_seq_length=2048)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(
        "json", 
        data_files="data/processed/cpt_training_data.jsonl", 
        split="train"
    )
    
    # Training Args
    training_args = SFTConfig(
        output_dir = "outputs/cpt_checkpoints",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4, # Simulates a batch size of 16 to save VRAM
        warmup_steps = 50,
        max_steps = 500, # A short run to verify it works before doing a full epoch
        learning_rate = 2e-4, # Standard learning rate for LoRA
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit", # Uses 8-bit optimizer to drastically reduce memory
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "wandb", # Automatically logs your loss curves to Weights & Biases
        dataset_text_field = "text",
        max_length = 2048,
        dataset_num_proc = 4,
        packing = True, # Enables packing multiple samples into one sequence
        # eos_token = tokenizer.eos_token,
    )
    
    # 4. Initialize the Trainer
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = training_args,
    )
    
    # 5. Start Training
    print("Starting training...")
    trainer.train()
    
    # 6. Save only the LoRA adapters (saves massive amounts of disk space)
    final_save_path = "models/qwen-0.5b-cpt-lora"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Training complete! Adapters saved to {final_save_path}")

if __name__ == "__main__":
    train_cpt()