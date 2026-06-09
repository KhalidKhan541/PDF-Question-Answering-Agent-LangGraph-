"""
LLM factory — swap providers without touching agent code.
Supported: openai | anthropic | ollama | groq
"""


def get_llm(provider: str = "openai", model: str | None = None, temperature: float = 0.0):
    """
    Return a LangChain chat model.

    provider: 'openai' | 'anthropic' | 'ollama' | 'groq'
    model:    override the default model for the provider
    temperature: 0.0 = deterministic, higher = more creative
    """

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-3-5-haiku-20241022",
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or "qwen2.5:7b",   # swap to qwen2.5:72b for better quality
            temperature=temperature,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "llama-3.1-8b-instant",
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Choose from: openai, anthropic, ollama, groq"
        )
