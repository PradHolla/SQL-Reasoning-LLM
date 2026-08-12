import os
from datasets import load_dataset

# Define the exact text format for Continual Pre-Training
CPT_TEMPLATE = """
-- Database Schema --
{context}

-- Executed SQL Query --
{question}
{answer}
"""

def format_row_for_cpt(row):
    """
    Takes a row from the HF dataset and formats it into a raw text block.
    """
    text = CPT_TEMPLATE.format(
        context=row["context"],
        question=f"-- Intent: {row['question']}",
        answer=row["answer"]
    )
    return {"text": text}

def build_cpt_dataset(output_dir: str = "data/processed"):
    """
    Downloads, formats, and saves the CPT dataset.
    """
    print("Downloading raw SQL dataset from Hugging Face...")
    # This dataset contains ~78k highly quality Text-to-SQL pairs
    raw_dataset = load_dataset("b-mc2/sql-create-context", split="train")
    
    print("Formatting rows for Continual Pre-Training...")
    # Map the formatting function across the dataset across multiple CPU cores
    cpt_dataset = raw_dataset.map(
        format_row_for_cpt,
        remove_columns=raw_dataset.column_names, # Drop old columns, keep only 'text'
        num_proc=4 
    )
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cpt_training_data.jsonl")
    cpt_dataset.to_json(output_path)
    
    print(f"Success! {len(cpt_dataset)} CPT records saved to {output_path}")
    print("\nSample Record:")
    print(cpt_dataset[0]["text"])

if __name__ == "__main__":
    build_cpt_dataset()