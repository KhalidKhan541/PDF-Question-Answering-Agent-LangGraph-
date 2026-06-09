# Document Q&A Agent — LangGraph RAG + Reflection Loop

A production-ready document Q&A pipeline built with LangGraph. The agent retrieves
relevant chunks, generates an answer, then **critiques its own answer** and retries
with a refined query if confidence is low.

---

## Graph Architecture

```
         ┌─────────┐
  START ─►  retrieve │
         └────┬────┘
              │
         ┌────▼────┐
         │ generate │
         └────┬────┘
              │
         ┌────▼────┐
         │  reflect │  ◄─────────────────────┐
         └────┬────┘                         │
              │                              │
       ┌──────▼──────┐    confidence < 0.7   │
       │  should_retry│ ──── & iter < 2 ─────┤
       └──────┬──────┘           ▲           │
              │ done             │      ┌────┴──────────┐
         ┌────▼────┐             └──────┤ refine_query  │
         │finalize │                    └───────────────┘
         └────┬────┘
              │
             END
```

### Nodes

| Node | Role |
|------|------|
| `retrieve` | Semantic search → top-k document chunks |
| `generate` | LLM answers question using retrieved context |
| `reflect` | Critic LLM scores answer (0–1) and gives feedback |
| `refine_query` | Rewrites question using critic feedback |
| `finalize` | Packages result into chat history |

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY (or see provider options below)
```

### 3. Add your documents

Drop `.pdf`, `.txt`, or `.md` files into the `data/` folder.

### 4. Ingest

```bash
python main.py ingest --docs ./data
```

### 5. Ask questions

```bash
# Single question
python main.py ask "What is the main argument of the paper?"

# Interactive loop
python main.py chat

# Output as JSON
python main.py ask "Summarize the key findings" --json
```

---

## LLM / Embedding Providers

Edit `main.py` to switch providers:

```python
LLM_PROVIDER   = "openai"      # openai | anthropic | ollama | groq
EMBED_PROVIDER = "openai"      # openai | ollama | huggingface
```

### Free / Local Option (Ollama)

```bash
ollama serve
ollama pull qwen2.5:7b           # or qwen2.5:72b for better quality
ollama pull nomic-embed-text
```

Then set both providers to `"ollama"` in `main.py`.

---

## Project Structure

```
doc_qa_agent/
├── app/
│   ├── graph.py          # LangGraph state + all nodes + routing
│   ├── vectorstore.py    # Ingestion, chunking, ChromaDB
│   └── llm.py            # LLM provider factory
├── data/                 # Drop your documents here
├── tests/
│   └── test_graph.py     # Unit tests (mock LLM)
├── main.py               # CLI: ingest / ask / chat
├── requirements.txt
└── .env.example
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | 4 | Chunks retrieved per query |
| `MAX_RETRIES` | 2 | Max reflection→retry loops |
| `confidence threshold` | 0.7 | Below this → retry |
| `chunk_size` | 800 | Characters per chunk |
| `chunk_overlap` | 150 | Overlap between chunks |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use mock LLMs — no API key needed.

---

## Extending

- **Add web search**: add a `web_search_node` that calls Tavily/SerpAPI when
  the vectorstore confidence is low.
- **Streaming**: use `graph.stream()` instead of `graph.invoke()` to stream
  node outputs to a frontend.
- **Persist chat history**: pass a LangGraph `MemorySaver` checkpointer to
  `g.compile(checkpointer=...)` for cross-session memory.
- **NEXUS integration**: expose `build_graph()` via FastAPI and stream
  node-level events to your LangGraph pipeline visualizer panel.
