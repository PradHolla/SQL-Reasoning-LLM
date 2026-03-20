from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def merge_and_export():
    # Load your final RL-trained model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "models/qwen-0.5b-reasoning-final",
        max_seq_length = 2048,
        dtype = None, # Auto-detects bf16 or fp16
        load_in_4bit = False, # 16-bit to ensure a high-quality merge
    )
    
    # Apply the same ChatML template to the tokenizer before exporting, so it's ready for inference without extra setup
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )    
    export_path = "models/qwen-0.5b-production-vllm"
    
    # save_method="merged_16bit" permanently fuses the LoRA matrices into the base weights
    # This will also save the updated tokenizer with the ChatML template included
    model.save_pretrained_merged(export_path, tokenizer, save_method = "merged_16bit")
    
if __name__ == "__main__":
    merge_and_export()