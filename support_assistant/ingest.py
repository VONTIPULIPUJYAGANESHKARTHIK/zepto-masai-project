import os
import chromadb
from sentence_transformers import SentenceTransformer

def ingest_docs():
    # Initialize embedding model (runs locally)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB in local persistent mode
    # Store DB inside support_assistant folder
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    
    # Recreate collection to ensure clean state
    try:
        client.delete_collection(name="zepto_policies")
    except:
        pass
    collection = client.create_collection(name="zepto_policies")

    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    
    documents = []
    ids = []
    
    print("Loading documents...")
    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                documents.append(content)
                ids.append(filename)
                
    if not documents:
        print("No documents found in docs/ directory!")
        return
        
    print(f"Generating embeddings for {len(documents)} documents...")
    # The simple per-document chunking scheme is used as allowed by rubric
    embeddings = model.encode(documents).tolist()
    
    print("Saving to ChromaDB...")
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=ids
    )
    
    print("Ingestion complete. ChromaDB ready.")

if __name__ == "__main__":
    ingest_docs()
