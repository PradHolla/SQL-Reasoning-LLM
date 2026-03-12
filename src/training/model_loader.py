import os
from unsloth import FastLanguageModel
import torch

def load_base_model(model_name: str = "Qwen/Qwen2.5-0.5B", max_seq_length: int = 2048):
    """
    Loads the Qwen model using Unsloth for memory-efficient training.
    """
    print(f"Loading {model_name}...")
    
    # Unsloth handles 4-bit quantization natively to save VRAM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None, # Auto-detects FP16/BF16 based on your AWS GPU
        load_in_4bit = True, 
    )

    # Attach LoRA adapters. We only train 1-2% of the parameters.
    # This turns full fine-tuning into Parameter-Efficient Fine-Tuning (PEFT).
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank of the adapter
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Unsloth optimizes dropout to 0
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    
    print("Model and Tokenizer loaded successfully with LoRA adapters.")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_base_model()
    model.print_trainable_parameters()