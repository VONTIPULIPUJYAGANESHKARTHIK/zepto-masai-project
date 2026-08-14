import os
from transformers import pipeline

def load_policies(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def get_answer(question, context):
    print("Loading QA model (this might take a few seconds on first run)...")
    # Using a small, fast model for question answering without needing an API key
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    result = qa_pipeline(question=question, context=context)
    return result['answer']

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.join(base_dir, "zepto_policies.txt")
    
    context = load_policies(policy_path)
    
    print("Welcome to Zepto Support Assistant!")
    print("Ask me any policy question based on Zepto's documentation.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue
                
            answer = get_answer(user_input, context)
            print(f"Zepto Assistant: {answer}\n")
            
        except KeyboardInterrupt:
            break
