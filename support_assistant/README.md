# Support Assistant Module

This module implements a GenAI support assistant that answers policy questions grounded in Zepto's documents.

## Workflow

This module utilizes a **Retrieval-Augmented Generation (RAG)** architecture using entirely local, free, open-source models (avoiding paid APIs like OpenAI).

1. **Knowledge Base**: We defined a mock Zepto policy document (`zepto_policies.txt`) which contains details about delivery guarantees, cancellations, and premium memberships.
2. **Document Chunking**: The `assistant.py` script loads the policy document and splits it into logical, manageable chunks (paragraphs).
3. **Semantic Retrieval**: When a user asks a question, we use `sentence-transformers` (`all-MiniLM-L6-v2`) to create vector embeddings of the chunks and the user's question. We calculate cosine similarity to retrieve the single most relevant chunk of context.
4. **Generation (Question Answering)**: The retrieved chunk is passed alongside the user's question to a local HuggingFace model (`distilbert-base-cased-distilled-squad`). The model reads the specific context and extracts the exact answer.

## Execution

Run the CLI assistant script from the root directory:
```bash
python support_assistant/assistant.py
```
*(Note: The first run will automatically download the language models from HuggingFace).*
