import sqlite3
import re
from openai import OpenAI

# Connect to your local vLLM instance (it mimics the OpenAI API perfectly)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

def extract_sql(text: str) -> str:
    """Extracts the SQL query from the model's <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        # Strip out markdown SQL blocks if the model accidentally added them inside the tags
        sql = match.group(1).replace("```sql", "").replace("```", "").strip()
        return sql
    return ""

def run_agentic_loop(schema: str, data_inserts: str, question: str, max_retries: int = 3):
    print("INITIATING AGENTIC SQL LOOP")

    # 1. Setup the deterministic environment (The Database)
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.executescript(schema)
    if data_inserts:
        cursor.executescript(data_inserts)

    # 2. Setup the agent's memory (Message History)
    messages = [
        {"role": "system", "content": "You are an elite database engineer. You must think step-by-step inside <think></think> tags, and output ONLY the final SQL query inside <answer></answer> tags. If you receive an error, analyze it inside the think tags and output the fixed query."},
        {"role": "user", "content": f"Schema: {schema}\nQuestion: {question}"}
    ]

    attempt = 1
    success = False

    while attempt <= max_retries and not success:
        print(f"Attempt {attempt} of {max_retries}...")
        
        # 3. Generate the reasoning and SQL
        response = client.chat.completions.create(
            model="models/qwen-0.5b-production-vllm",
            messages=messages,
            max_tokens=512,
            temperature=0.2 # Low temperature for more deterministic coding
        )
        
        agent_reply = response.choices[0].message.content
        print(f"Model's Thought Process:\n{agent_reply}\n")
        
        sql_query = extract_sql(agent_reply)
        
        if not sql_query:
            error_msg = "Format Error: Could not find SQL inside <answer> tags."
            print(f"{error_msg}")
            messages.append({"role": "assistant", "content": agent_reply})
            messages.append({"role": "user", "content": error_msg})
            attempt += 1
            continue

        # 4. The Execution Step (Testing reality)
        try:
            print(f"Executing Query: {sql_query}")
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            # If we get here, the SQL was syntactically perfect
            print(f"Success! Data Retrieved: {results}\n")
            success = True
            
        except sqlite3.Error as e:
            # 5. The Self-Correction Trigger
            error_msg = str(e)
            print(f"Database Error: {error_msg}")
            print("Feeding error back to the model for self-correction...\n")
            
            # Append the failed attempt and the compiler error to the chat history
            messages.append({"role": "assistant", "content": agent_reply})
            messages.append({
                "role": "user", 
                "content": f"The database threw this error: '{error_msg}'. Please write a new <think> block analyzing why this failed, and provide the corrected SQL query inside <answer> tags."
            })
            
            attempt += 1

    conn.close()
    
    if not success:
        print("Agent failed to solve the query within the retry limit.")

if __name__ == "__main__":
    # THE FINAL BOSS: The Self-Join Trap
    # Notice the column is strictly named 'manager_emp_id', not 'manager_id'
    test_schema = """
    CREATE TABLE staff (emp_id INT, employee_name VARCHAR, manager_emp_id INT, salary DECIMAL);
    """
    
    test_data = """
    INSERT INTO staff VALUES (1, 'The Boss', NULL, 250000);
    INSERT INTO staff VALUES (2, 'Alice', 1, 120000);
    INSERT INTO staff VALUES (3, 'Bob', 1, 110000);
    INSERT INTO staff VALUES (4, 'Charlie', 2, 130000); -- Charlie earns 130k, his manager Alice earns 120k
    """
    
    # THE TRICK QUESTION: Requires comparing two rows in the exact same table
    test_question = "Find the employee_name of any staff member who earns a strictly higher salary than their direct manager. You must also return their manager's employee_name."
    
    run_agentic_loop(test_schema, test_data, test_question)