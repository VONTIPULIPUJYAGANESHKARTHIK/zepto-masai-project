import os
import json
from typing import List, Dict, Any, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# --- Schema & State ---
class AskRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

import operator
from typing import Annotated

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    intent: str
    retrieved_docs: List[Dict[str, str]]
    final_response: dict
    retries: int

# --- Setup Shared Resources ---
# We initialize the DB and Embedding model globally for efficiency
db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
try:
    collection = chroma_client.get_collection(name="zepto_policies")
except Exception:
    collection = None

# Initialize embedding model lazily inside functions if needed, but since retrieval happens
# on every policy request, doing it globally is better.
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Mock Toggle
def is_mock():
    val = os.environ.get("MOCK_LLM", "1")
    return val != "0"

# --- Prompt Template (Requirement 2) ---
# Role: You are a customer support AI for Zepto.
# Context: Provided chunks.
# Task: Answer the user's question.
# Format: JSON matching the Pydantic schema.
# Length: Keep answers concise and direct (under 3 sentences).
# Negative Constraint: Do not answer using information not present in the provided context.
# Few-Shot: Embedded example.
PROMPT_TEMPLATE = """You are a helpful customer support AI for Zepto.
Your task is to answer the user's query based ONLY on the provided context.
Keep your answers concise and direct, no more than 3 sentences.

IMPORTANT CONSTRAINTS:
- DO NOT answer using information not present in the provided context. If the context doesn't contain the answer, say so.
- You must output VALID JSON strictly matching this schema:
{
  "answer": "string",
  "sources": ["doc_01.txt", "doc_02.txt"],
  "confidence": 0.0 to 1.0
}

CONTEXT:
{context}

USER QUERY: {query}

FEW-SHOT EXAMPLE:
Context: [doc_09.txt] "Zepto bags cost INR 5 each."
User: "How much are bags?"
Output: {"answer": "Zepto bags cost INR 5 each.", "sources": ["doc_09.txt"], "confidence": 1.0}

YOUR OUTPUT:
"""

# --- Real LLM Helper (Optional Extension) ---
def call_groq_llm(prompt: str) -> str:
    # Requires groq package and API key
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content

# --- Nodes ---
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()
    if is_mock():
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        intent = "policy_question" if any(k in query for k in keywords) else "general_question"
    else:
        # Optional real LLM classification
        prompt = f"Classify this query as 'policy_question' (about Zepto rules, delivery, refunds) or 'general_question' (everything else). Reply with ONLY the exact intent string.\\nQuery: {query}"
        try:
            resp = call_groq_llm(prompt).strip().strip('"').strip("'")
            intent = "policy_question" if "policy_question" in resp.lower() else "general_question"
        except Exception:
            intent = "general_question"
            
    state["intent"] = intent
    return state

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    
    # 1. Retrieval (runs for real in both modes)
    if collection:
        q_emb = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        docs = results['documents'][0]
        ids = results['ids'][0]
        retrieved = [{"id": i, "content": c} for i, c in zip(ids, docs)]
    else:
        retrieved = []
    
    state["retrieved_docs"] = retrieved
    
    # 2. Generation (branches on MOCK)
    if is_mock():
        if retrieved:
            top_chunk = retrieved[0]["content"]
            snippet = top_chunk[:200] + ("..." if len(top_chunk) > 200 else "")
            ans = f"Based on the retrieved context: {snippet}"
            sources = [r["id"] for r in retrieved]
        else:
            ans = "Based on the retrieved context: No context found."
            sources = []
            
        state["final_response"] = {
            "answer": ans,
            "sources": sources,
            "confidence": 1.0
        }
    else:
        # Real LLM logic with 2-retry loop
        context_str = "\\n".join([f"[{r['id']}] {r['content']}" for r in retrieved])
        prompt = PROMPT_TEMPLATE.format(context=context_str, query=query)
        
        max_retries = 2
        attempts = 0
        success = False
        final_dict = None
        
        while attempts <= max_retries and not success:
            try:
                raw_out = call_groq_llm(prompt)
                parsed = json.loads(raw_out)
                # Validate with Pydantic
                valid_resp = AskResponse(**parsed)
                final_dict = valid_resp.model_dump()
                success = True
            except Exception as e:
                attempts += 1
                prompt += f"\\n\\nWARNING: Your previous output failed validation. Error: {str(e)}. Please try again and ONLY output valid JSON."
                
        if success:
            state["final_response"] = final_dict
        else:
            state["final_response"] = {
                "answer": "Error generating structured response from LLM.",
                "sources": [],
                "confidence": 0.0
            }

    return state

def direct_answer(state: GraphState) -> GraphState:
    if is_mock():
        state["final_response"] = {
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0
        }
    else:
        # Real LLM direct answer without context
        query = state["query"]
        prompt = f"Answer the user's general query concisely. Format as JSON: {{'answer': '...', 'sources': [], 'confidence': 1.0}}. Query: {query}"
        try:
            raw_out = call_groq_llm(prompt)
            parsed = json.loads(raw_out)
            AskResponse(**parsed) # validate
            state["final_response"] = parsed
        except Exception:
            state["final_response"] = {
                "answer": "I can only answer questions about Zepto policies right now.",
                "sources": [],
                "confidence": 1.0
            }
    return state

def route_intent(state: GraphState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

# --- Build LangGraph ---
workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges("classify_intent", route_intent)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# --- FastAPI App ---
app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    # Pass user message into state
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "intent": "",
        "retrieved_docs": [],
        "final_response": {},
        "retries": 0
    }
    
    config = {"configurable": {"thread_id": request.session_id}}
    result = app_graph.invoke(initial_state, config=config)
    
    # Store AI response in messages array for next time
    # Though LangGraph nodes technically should do this, we can just let it persist state.
    # The Pydantic output is stored in final_response.
    return result["final_response"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
