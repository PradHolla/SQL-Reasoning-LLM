import os
from datasets import load_dataset

def format_row_for_sft(row):
    """
    Formats the SQL data into a strict instruction-response ChatML format 
    with <think> and <answer> tags.
    """
    # We provide a generic "thought" process just to teach the model the structure.
    # The actual deep reasoning will be learned during the GRPO phase.
    thought_process = "I need to analyze the schema to find the correct tables and columns, then construct a valid SQL query."
    
    chat = [
        {"role": "system", "content": "You are a database expert. You must think step-by-step inside <think></think> tags, and output ONLY the final SQL query inside <answer></answer> tags."},
        {"role": "user", "content": f"Schema: {row['context']}\nQuestion: {row['question']}"},
        {"role": "assistant", "content": f"<think>\n{thought_process}\n</think>\n<answer>\n{row['answer']}\n</answer>"}
    ]
    return {"messages": chat}

def build_sft_dataset(output_dir: str = "data/processed"):
    print("Loading dataset for SFT formatting...")
    raw_dataset = load_dataset("b-mc2/sql-create-context", split="train")
    
    print("Formatting rows into ChatML <think> format...")
    sft_dataset = raw_dataset.map(
        format_row_for_sft,
        remove_columns=raw_dataset.column_names,
        num_proc=4
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sft_training_data.jsonl")
    sft_dataset.to_json(output_path)
    print(f"Success! SFT records saved to {output_path}")

if __name__ == "__main__":
    build_sft_dataset()