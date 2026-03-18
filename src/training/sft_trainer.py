import os
import torch

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def train_sft():
    print("Initializing SFT Training Run...")
    
    # Load YOUR newly trained CPT model, not the base Qwen
    model_path = "models/qwen-0.5b-cpt-lora"
    print(f"Loading CPT adapters from {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    
    # Fix the EOS token issue natively
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- ADD THIS BLOCK ---
    print("Injecting ChatML template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml", # Forces Qwen's standard format
    )

    # Unlock the loaded LoRA adapters so they can be trained further
    FastLanguageModel.for_training(model)
    
    # Load the formatted ChatML SFT dataset
    dataset = load_dataset(
        "json", 
        data_files="data/processed/sft_training_data.jsonl", 
        split="train"
    )
    
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        # Flattens the {"role": ..., "content": ...} dicts into a raw string
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    print("Applying Qwen Chat Template...")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # SFT Config
    training_args = SFTConfig(
        output_dir = "outputs/sft_checkpoints",
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 50,
        max_steps = 500, # Validation run for SFT
        learning_rate = 2e-5, # Notice the learning rate is 10x lower than CPT!
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "wandb",
        max_length = 2048,
        packing = False, # We turn packing OFF for strict Q&A formatting
    )
    
    # Initialize the Trainer
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = training_args,
    )
    
    print("Starting SFT training...")
    trainer.train()
    
    # Save the final SFT adapters
    final_save_path = "models/qwen-0.5b-sft-lora"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"SFT complete! Adapters saved to {final_save_path}")

if __name__ == "__main__":
    train_sft()