import os
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_chunk_policies(filepath):
    """Loads policies and splits them into logical chunks (paragraphs)."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by double newline to get distinct policy sections
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    return chunks

def get_answer_rag(question, chunks, embedder, qa_pipeline):
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
    result = qa_pipeline(question=question, context=best_chunk)
    return result['answer']

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.join(base_dir, "zepto_policies.txt")
    
    print("Loading models (this might take a few seconds on first run)...")
    # Load sentence transformer for semantic retrieval
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Load QA pipeline for generation
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
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
                
            answer = get_answer_rag(user_input, chunks, embedder, qa_pipeline)
            print(f"Zepto Assistant: {answer}\n")
            
        except KeyboardInterrupt:
            break
