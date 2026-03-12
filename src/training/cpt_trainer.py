import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Import the loader you built earlier
from model_loader import load_base_model

def train_cpt():
    print("Initializing CPT Training Run...")
    
    # Load the model and tokenizer with LoRA adapters
    model, tokenizer = load_base_model("Qwen/Qwen2.5-0.5B", max_seq_length=2048)
    
    # Load dataset
    dataset = load_dataset(
        "json", 
        data_files="data/processed/cpt_training_data.jsonl", 
        split="train"
    )
    
    # Training Args
    training_args = TrainingArguments(
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
    )
    
    # 4. Initialize the Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 4,
        packing = True, # Packs multiple short SQL schemas into one 2048-token chunk
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
    # Ensure Weights & Biases has your API key loaded in your environment variables
    # export WANDB_API_KEY="your_key_here"
    train_cpt()