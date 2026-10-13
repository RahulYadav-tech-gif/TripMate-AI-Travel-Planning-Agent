"""Configuration module for TripMate."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Application settings and runtime parameters."""

    # Gemini settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    # RAG / ChromaDB settings
    chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db"))
    data_dir: str = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "2"))

    # Orchestrator limits
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", str(BASE_DIR / "tripmate.log"))

    def validate_api_key(self) -> bool:
        """Checks whether GEMINI_API_KEY is provided and not a placeholder."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip() != "your_gemini_api_key_here")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns singleton Settings instance."""
    return Settings()
