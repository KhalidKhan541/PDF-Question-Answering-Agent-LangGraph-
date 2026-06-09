"""
Document Q&A Agent — CLI entry point

Usage:
  # 1. Ingest documents (first run)
  python main.py ingest --docs ./data

  # 2. Ask a single question
  python main.py ask "What is the main argument of the paper?"

  # 3. Interactive chat loop
  python main.py chat
"""

import argparse
import json
from pathlib import Path

from app.graph import build_graph
from app.llm import get_llm
from app.vectorstore import build_vectorstore, load_vectorstore

# ── Config ────────────────────────────────────────────────────────────────────
PERSIST_DIR = "./chroma_db"
DOCS_DIR    = "./data"
LLM_PROVIDER    = "openai"       # openai | anthropic | ollama | groq
EMBED_PROVIDER  = "openai"       # openai | ollama | huggingface
TOP_K           = 4


def get_graph():
    retriever = load_vectorstore(PERSIST_DIR, EMBED_PROVIDER, TOP_K)
    llm       = get_llm(LLM_PROVIDER)
    return build_graph(retriever, llm)


def run_question(graph, question: str, verbose: bool = True) -> dict:
    """Run the graph for one question and return the final state."""
    initial_state = {
        "question":    question,
        "documents":   [],
        "answer":      "",
        "reflection":  "",
        "confidence":  0.0,
        "iterations":  0,
        "chat_history": [],
    }

    final_state = graph.invoke(initial_state)

    if verbose:
        print("\n" + "═" * 60)
        print(f"Q: {question}")
        print(f"\nA: {final_state['answer']}")
        print(f"\n📊 Confidence : {final_state['confidence']:.2f}")
        print(f"🔁 Iterations : {final_state['iterations']}")
        print(f"💬 Reflection : {final_state['reflection']}")
        print(f"📄 Chunks used: {len(final_state['documents'])}")
        print("═" * 60)

    return final_state


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_ingest(args):
    print(f"Ingesting documents from '{args.docs}'...")
    build_vectorstore(args.docs, PERSIST_DIR, EMBED_PROVIDER, TOP_K)
    print("✅ Ingestion complete.")


def cmd_ask(args):
    graph = get_graph()
    state = run_question(graph, args.question)
    if args.json:
        out = {
            "question":   state["question"],
            "answer":     state["answer"],
            "confidence": state["confidence"],
            "iterations": state["iterations"],
            "reflection": state["reflection"],
        }
        print(json.dumps(out, indent=2))


def cmd_chat(args):
    print("\n🔍 Document Q&A Agent  (type 'exit' to quit)\n")
    graph   = get_graph()
    history = []

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q or q.lower() in ("exit", "quit"):
            print("Bye!")
            break

        state = run_question(graph, q)
        history.append({"q": q, "a": state["answer"]})


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Document Q&A LangGraph Agent")
    sub    = parser.add_subparsers(dest="cmd")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the vector store")
    p_ingest.add_argument("--docs", default=DOCS_DIR, help="Path to documents directory")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", help="Question string")
    p_ask.add_argument("--json", action="store_true", help="Output as JSON")

    # chat
    sub.add_parser("chat", help="Interactive chat loop")

    args = parser.parse_args()

    if args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "chat":
        cmd_chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
