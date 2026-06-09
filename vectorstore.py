"""
Vector store utilities — ingest documents and create retriever.
Uses ChromaDB (local, no server needed) + OpenAI or Ollama embeddings.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma


# ─── Loader factory ───────────────────────────────────────────────────────────

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md":  UnstructuredMarkdownLoader,
}


def load_documents(source_dir: str) -> List[Document]:
    """Load all supported files from a directory."""
    docs = []
    for path in Path(source_dir).rglob("*"):
        loader_cls = LOADER_MAP.get(path.suffix.lower())
        if loader_cls:
            print(f"  Loading: {path.name}")
            docs.extend(loader_cls(str(path)).load())
    if not docs:
        raise ValueError(f"No supported documents found in '{source_dir}'.")
    return docs


def chunk_documents(docs: List[Document], chunk_size=800, chunk_overlap=150) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ─── Embedding selector ───────────────────────────────────────────────────────

def get_embeddings(provider: str = "openai"):
    """
    Return an embedding model.
    provider: 'openai' | 'ollama' | 'huggingface'
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# ─── Vector store ─────────────────────────────────────────────────────────────

def build_vectorstore(
    source_dir: str,
    persist_dir: str = "./chroma_db",
    embedding_provider: str = "openai",
    top_k: int = 4,
):
    """
    Ingest documents → chunk → embed → persist to ChromaDB.
    Returns a configured retriever.
    """
    print(f"\n[VectorStore] Ingesting from '{source_dir}'...")
    docs   = load_documents(source_dir)
    chunks = chunk_documents(docs)
    embed  = get_embeddings(embedding_provider)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        persist_directory=persist_dir,
        collection_name="doc_qa",
    )
    print(f"  Stored {vectorstore._collection.count()} vectors in '{persist_dir}'")
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def load_vectorstore(
    persist_dir: str = "./chroma_db",
    embedding_provider: str = "openai",
    top_k: int = 4,
):
    """Load an already-persisted ChromaDB vectorstore."""
    embed = get_embeddings(embedding_provider)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embed,
        collection_name="doc_qa",
    )
    count = vectorstore._collection.count()
    if count == 0:
        raise RuntimeError(f"No vectors found in '{persist_dir}'. Run ingest first.")
    print(f"[VectorStore] Loaded {count} vectors from '{persist_dir}'")
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
