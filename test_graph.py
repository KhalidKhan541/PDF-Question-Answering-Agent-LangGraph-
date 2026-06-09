"""
Tests for graph nodes using mock LLM and retriever.
Run with: pytest tests/
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.graph import (
    generate_node,
    reflect_node,
    refine_query_node,
    finalize_node,
    should_retry,
    AgentState,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def mock_llm(response: str):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=response)
    return llm


def base_state(**overrides) -> AgentState:
    state = AgentState(
        question="What is self-attention?",
        documents=[Document(page_content="Self-attention maps queries to keys and values.")],
        answer="",
        reflection="",
        confidence=0.0,
        iterations=0,
        chat_history=[],
    )
    state.update(overrides)
    return state


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_generate_node_returns_answer():
    llm   = mock_llm("Self-attention computes dot products of Q, K, V.")
    state = base_state()
    out   = generate_node(state, llm)
    assert "answer" in out
    assert len(out["answer"]) > 0


def test_reflect_node_parses_confidence():
    llm   = mock_llm("CONFIDENCE: 0.85\nFEEDBACK: Answer is complete and accurate.")
    state = base_state(answer="Self-attention uses Q, K, V matrices.")
    out   = reflect_node(state, llm)
    assert abs(out["confidence"] - 0.85) < 0.01
    assert "complete" in out["reflection"]


def test_reflect_node_handles_bad_format():
    llm   = mock_llm("I think it's pretty good, maybe 80% confident.")
    state = base_state(answer="Some answer")
    out   = reflect_node(state, llm)
    assert 0.0 <= out["confidence"] <= 1.0  # fallback


def test_should_retry_low_confidence():
    state = base_state(confidence=0.4, iterations=0)
    assert should_retry(state) == "retry"


def test_should_retry_high_confidence():
    state = base_state(confidence=0.9, iterations=0)
    assert should_retry(state) == "done"


def test_should_retry_max_iterations():
    state = base_state(confidence=0.3, iterations=2)
    assert should_retry(state) == "done"


def test_refine_query_increments_iterations():
    llm   = mock_llm("How does the self-attention mechanism compute attention scores?")
    state = base_state(reflection="Too vague. Ask about the computation.", iterations=0)
    out   = refine_query_node(state, llm)
    assert out["iterations"] == 1
    assert out["documents"] == []   # stale docs cleared


def test_finalize_node_adds_to_history():
    state = base_state(
        answer="Self-attention uses Q, K, V.",
        confidence=0.9,
        iterations=1,
    )
    out = finalize_node(state)
    assert len(out["chat_history"]) == 1
    entry = out["chat_history"][0]
    assert entry["answer"] == state["answer"]
    assert entry["confidence"] == state["confidence"]
