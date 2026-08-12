import torch
from unsloth import FastLanguageModel

from sqlrl.tokenizer import assert_stops_on, build_tokenizer

def generate_sql():
    # Load your final reasoning model
    model_path = "models/qwen-0.5b-reasoning-final"
    print(f"Loading reasoning engine from {model_path}...")
    
    model, _ = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # Built from the checkpoint, never from the base model name: v0's saved
    # vocab has <|im_end|> at a different id than stock Qwen2.5. See
    # sqlrl.tokenizer. The assert turns a silent non-terminating generation
    # into an immediate, readable failure.
    tokenizer = build_tokenizer(model_path, chat=True)
    assert_stops_on(tokenizer, model_path)

    # Unsloth optimization for 2x faster inference
    FastLanguageModel.for_inference(model)

    # A sample schema and question it hasn't seen before
    schema = "CREATE TABLE employees (id INT, name VARCHAR, department_id INT, salary DECIMAL); CREATE TABLE departments (id INT, dept_name VARCHAR);"
    question = "What is the average salary of employees in the Engineering department?"

    # Format the prompt using the exact ChatML structure we trained it on
    messages = [
        {"role": "system", "content": "You are a database expert. You must think step-by-step inside <think></think> tags, and output ONLY the final SQL query inside <answer></answer> tags."},
        {"role": "user", "content": f"Schema: {schema}\nQuestion: {question}"}
    ]

    # Convert the messages into the raw ChatML string format
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize = False, 
        add_generation_prompt = True
    )

    # Tokenize and push to GPU
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    print("\n--- Generating Reasoning Trace ---")
    
    # Generate the response
    outputs = model.generate(
        **inputs,
        max_new_tokens = 512, # Gives it room to think
        use_cache = True,     # Standard inference optimization
        do_sample = True,
        temperature = 0.6,    # A little bit of randomness for reasoning
    )

    # Decode the output tokens back into readable text
    response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    
    # We slice off the prompt so it only prints the model's generated response
    generated_text = response.split("user\nSchema:")[-1].split("\nassistant\n")[-1]
    print(generated_text)

if __name__ == "__main__":
    generate_sql()