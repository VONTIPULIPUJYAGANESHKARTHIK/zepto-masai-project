import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_chunk_policies(filepath):
    """Loads policies and splits them into logical chunks (paragraphs)."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by double newline to get distinct policy sections
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    return chunks

def get_answer_rag(question, chunks, embedder, tokenizer, model):
    """Retrieves the most relevant chunk and generates an answer (True RAG)."""
    # 1. Retrieval
    # Embed the chunks and the question
    chunk_embeddings = embedder.encode(chunks)
    question_embedding = embedder.encode([question])
    
    # Calculate cosine similarity to find the most relevant chunk
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    best_chunk_idx = np.argmax(similarities)
    best_chunk = chunks[best_chunk_idx]
    best_score = similarities[best_chunk_idx]
    
    # Fallback if no relevant chunk is found
    if best_score < 0.2:
        return "I'm sorry, I couldn't find an answer to that in the policy documents."
    
    # 2. Generation (QA based on the retrieved context)
    inputs = tokenizer(question, best_chunk, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.join(base_dir, "zepto_policies.txt")
    
    print("Loading models (this might take a few seconds on first run)...")
    # Load sentence transformer for semantic retrieval
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Load QA pipeline for generation
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    chunks = load_and_chunk_policies(policy_path)
    
    print("\nWelcome to Zepto Support Assistant (RAG Enabled)!")
    print("Ask me any policy question based on Zepto's documentation.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue
                
            answer = get_answer_rag(user_input, chunks, embedder, tokenizer, model)
            print(f"Zepto Assistant: {answer}\n")
            
        except KeyboardInterrupt:
            break
