import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GROQ_API_KEY:    str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY:  str = os.getenv("GEMINI_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    REDIS_URL:       str = os.getenv("REDIS_URL", "redis://localhost:6379")
    LOG_LEVEL:       str = os.getenv("LOG_LEVEL", "INFO")
    MAX_RETRIES:     int = int(os.getenv("MAX_RETRIES", "3"))
    FAISS_STORE_PATH: str = os.getenv("FAISS_STORE_PATH", "faiss_store")
    LOG_FILE:        str = os.getenv("LOG_FILE", "logs/orchestrator.jsonl")

    # ── Model identifiers ───────────────────────────────────────────────────
    # Ollama local models
    OLLAMA_CODER_MODEL:   str = "qwen2.5-coder:7b"   # Code & debugging
    OLLAMA_GENERAL_MODEL: str = "mistral:7b"          # General tasks
    OLLAMA_LLAMA_MODEL:   str = "llama3.2:3b"         # General / code fallback

    # Cloud models (updated to current non-deprecated names)
    GROQ_MODEL:   str = "llama-3.3-70b-versatile"    # Fast inference
    GEMINI_MODEL: str = "gemini-2.0-flash"            # Complex reasoning

    # ── Routing thresholds ──────────────────────────────────────────────────
    COMPLEXITY_HIGH_THRESHOLD: float = 0.7
    COMPLEXITY_LOW_THRESHOLD:  float = 0.3


config = Config()
