import os
import re
import torch

# 1. UNSLOTH FIRST (Recent Unsloth versions auto-patch GRPO, no manual patching needed)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
# 2. TRL AND DATASETS
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

# REWARD FUNCTIONS

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Rewards the model +1.0 if it strictly follows the <think> and <answer> format.
    Penalizes it 0.0 if it hallucinates other tags or forgets them.
    """
    rewards = []
    for completion in completions:
        # TRL passes the generated text in a specific list format
        text = completion[0]["content"] 
        # Regex checks for the exact XML structure we taught it in SFT
        if re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", text, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    """
    Rewards the model +2.0 if the SQL inside the <answer> tags perfectly matches 
    the ground truth SQL query.
    """
    rewards = []
    # 'answer' is passed from our dataset column mapped below
    for completion, ground_truth in zip(completions, answer):
        text = completion[0]["content"]
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        
        if match:
            # Extract the SQL, strip whitespace, and normalize to lowercase for comparison
            extracted_sql = match.group(1).strip().lower()
            expected_sql = ground_truth.strip().lower()
            
            if extracted_sql == expected_sql:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

# --- MAIN TRAINING LOOP ---

def train_grpo():
    print("Initializing GRPO Reinforcement Learning...")
    
    # 1. Load YOUR SFT Model (The one that knows how to think)
    model_path = "models/qwen-0.5b-sft-lora"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- ADD THIS BLOCK ---
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )
    # ----------------------

    FastLanguageModel.for_training(model)
    
    # 2. Load and format the raw dataset directly in memory
    raw_dataset = load_dataset("b-mc2/sql-create-context", split="train")
    
    # RL is compute-heavy. We only take a 5,000-row subset to keep the training time realistic.
    dataset = raw_dataset.select(range(5000))
    
    def format_grpo_prompts(row):
        # GRPO strictly requires a 'prompt' column (which can be ChatML)
        prompt = [
            {"role": "system", "content": "You are a database expert. You must think step-by-step inside <think></think> tags, and output ONLY the final SQL query inside <answer></answer> tags."},
            {"role": "user", "content": f"Schema: {row['context']}\nQuestion: {row['question']}"}
        ]
        return {"prompt": prompt, "answer": row["answer"]}
        
    dataset = dataset.map(format_grpo_prompts)
    
    # 3. Configure GRPO
    training_args = GRPOConfig(
        output_dir = "outputs/grpo_checkpoints",
        learning_rate = 5e-6, # Extremely low LR to prevent destroying the SFT weights
        per_device_train_batch_size = 1, # Must be 1 for GRPO to manage multiple generations
        gradient_accumulation_steps = 4,
        max_steps = 300, 
        num_generations = 4, # The model will generate 4 different answers per question to compete against each other
        max_prompt_length = 512,
        max_completion_length = 512, # Gives the model enough room to "think" before answering
        logging_steps = 10,
        report_to = "wandb",
        optim = "adamw_8bit",
        seed = 3407,
    )
    
    # 4. Initialize the GRPO Trainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [format_reward_func, correctness_reward_func],
        args = training_args,
        train_dataset = dataset,
    )
    
    # 5. Execute
    print("Starting GRPO Training...")
    trainer.train()
    
    # 6. Save the final, reasoning-capable model
    final_save_path = "models/qwen-0.5b-reasoning-final"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"GRPO complete! Final model saved to {final_save_path}")

if __name__ == "__main__":
    train_grpo()