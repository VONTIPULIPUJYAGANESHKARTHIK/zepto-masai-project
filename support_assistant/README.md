# Module 3: Support Assistant

This module implements a complete GenAI Retrieval-Augmented Generation (RAG) service for Zepto's policy corpus, orchestrated via LangGraph and exposed through a FastAPI endpoint.

## Architecture Description

The RAG pipeline flows through four distinct stages:

1. **Ingestion & Embedding (`ingest.py`)**:
   - The 8 raw text documents in `/docs` are loaded.
   - Embeddings are generated locally using `sentence-transformers/all-MiniLM-L6-v2`.
   - The embedded documents are stored persistently in a local ChromaDB collection (`/chroma_db`).

2. **Classification (Intent Routing) (`assistant.py: classify_intent`)**:
   - When a query hits the `/ask` endpoint, LangGraph first routes it to the `classify_intent` node.
   - **`MOCK_LLM` Branching**: If `MOCK_LLM=1` (or unset), a deterministic keyword heuristic is used to classify the query as `policy_question` or `general_question` (no LLM call is made). If `MOCK_LLM=0`, a real LLM (Groq) is prompted to classify the intent.
   - A conditional edge routes `policy_question` to `retrieve_and_answer` and `general_question` to `direct_answer`.

3. **Retrieval (`assistant.py: retrieve_and_answer`)**:
   - For `policy_question` queries, the query is embedded locally and the top 3 most similar chunks are retrieved from ChromaDB via cosine similarity. 
   - *Note: This retrieval step runs for real regardless of the `MOCK_LLM` toggle, as it requires no external API.*

4. **Generation (`assistant.py: retrieve_and_answer` & `direct_answer`)**:
   - **`MOCK_LLM` Branching**: 
     - If `MOCK_LLM=1` (default), generation is entirely mocked. `retrieve_and_answer` returns a deterministic string: `"Based on the retrieved context: <snippet>"`, and `direct_answer` returns a fixed string: `"I can only answer questions about Zepto policies right now."`.
     - If `MOCK_LLM=0`, the Groq LLM is called using a strict Role-Context-Task prompt (including negative constraints and a few-shot example). If the Pydantic schema validation fails, the node automatically retries up to 2 times with corrective instructions.

## Example Transcripts (Mock Baseline)

These examples were generated locally using the default mock baseline (`MOCK_LLM=1`).

### Policy Question (Triggers Retrieval)
**Query:** "How do I return groceries?"

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop...",
  "sources": [
    "doc_02.txt",
    "doc_06.txt",
    "doc_05.txt"
  ],
  "confidence": 1.0
}
```

### General Question (Direct Answer)
**Query:** "What is the weather today?"

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Running Locally

1. **Setup**: Run `python support_assistant/ingest.py` to initialize ChromaDB.
2. **Server**: Run `uvicorn support_assistant.assistant:app --host 0.0.0.0 --port 7860`.
3. **Docker**:
   ```bash
   docker build -t zepto-support .
   docker run -p 7860:7860 zepto-support
   ```
