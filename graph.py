"""
Document Q&A LangGraph Agent
Graph: retrieve → generate → reflect → (retry | done)
"""

from typing import TypedDict, Annotated, List
import operator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    documents: Annotated[List[Document], operator.add]   # retrieved chunks
    answer: str                                           # current answer draft
    reflection: str                                       # critic's feedback
    confidence: float                                     # 0.0 – 1.0
    iterations: int                                       # retry counter
    chat_history: Annotated[List, operator.add]          # full turn history


# ─── Nodes ────────────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState, retriever) -> dict:
    """Fetch top-k relevant chunks from the vector store."""
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "iterations": state.get("iterations", 0)}


def generate_node(state: AgentState, llm) -> dict:
    """Generate an answer grounded in retrieved documents."""
    context = "\n\n---\n\n".join(d.page_content for d in state["documents"])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context doesn't contain enough information, say so explicitly.

Context:
{context}

Question: {state['question']}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


def reflect_node(state: AgentState, llm) -> dict:
    """Critic: score the answer and decide if retrieval should retry."""
    prompt = f"""You are a quality critic. Evaluate the answer below.

Question: {state['question']}
Answer: {state['answer']}

Reply in this exact format (no extra text):
CONFIDENCE: <float 0.0-1.0>
FEEDBACK: <one sentence — what's missing or weak, or 'Answer is complete and accurate.'>"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    confidence = 0.5
    feedback = "Unable to parse reflection."
    for line in text.splitlines():
        if line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {"confidence": confidence, "reflection": feedback}


def refine_query_node(state: AgentState, llm) -> dict:
    """Rewrite the question to be more specific based on critic feedback."""
    prompt = f"""The original question didn't get a satisfactory answer.
Rewrite it to be more specific, incorporating the feedback.

Original question: {state['question']}
Critic feedback: {state['reflection']}

Rewritten question (one sentence only):"""

    response = llm.invoke([HumanMessage(content=prompt)])
    new_q = response.content.strip()
    return {
        "question": new_q,
        "iterations": state["iterations"] + 1,
        "documents": [],   # clear stale docs before re-retrieval
    }


def finalize_node(state: AgentState) -> dict:
    """Package the final answer into chat history."""
    turn = {
        "question": state["question"],
        "answer": state["answer"],
        "confidence": state["confidence"],
        "iterations": state["iterations"],
    }
    return {"chat_history": [turn]}


# ─── Routing ──────────────────────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    """Return 'retry' if confidence is low and we haven't hit the limit."""
    MAX_RETRIES = 2
    if state["confidence"] < 0.7 and state["iterations"] < MAX_RETRIES:
        return "retry"
    return "done"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph(retriever, llm):
    """Assemble and compile the LangGraph StateGraph."""

    # Bind retriever/llm into each node via closures
    def _retrieve(s): return retrieve_node(s, retriever)
    def _generate(s): return generate_node(s, llm)
    def _reflect(s):  return reflect_node(s, llm)
    def _refine(s):   return refine_query_node(s, llm)

    g = StateGraph(AgentState)

    g.add_node("retrieve",    _retrieve)
    g.add_node("generate",    _generate)
    g.add_node("reflect",     _reflect)
    g.add_node("refine_query",_refine)
    g.add_node("finalize",    finalize_node)

    # Linear flow
    g.set_entry_point("retrieve")
    g.add_edge("retrieve",     "generate")
    g.add_edge("generate",     "reflect")

    # Conditional branch after reflection
    g.add_conditional_edges(
        "reflect",
        should_retry,
        {"retry": "refine_query", "done": "finalize"},
    )

    # Retry loop: refine → re-retrieve → generate → reflect
    g.add_edge("refine_query", "retrieve")
    g.add_edge("finalize",     END)

    return g.compile()
