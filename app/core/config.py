"""
Configuration settings for KASPARRO Claim Checker service.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable fallbacks."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "KASPARRO AI Claim Extraction & Fact Verification Service"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"

    # Data paths (relative to workspace root)
    DATA_DIR: Path = Path("data")
    ANSWERS_FILE: Path = Path("data/answers.csv")
    BRAND_FACTS_FILE: Path = Path("data/brand_facts.csv")
    DEVELOPMENT_GOLD_FILE: Path = Path("data/development_gold.csv")

    # Expected SHA-256 digests for original data files
    EXPECTED_ANSWERS_SHA256: str = (
        "fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724"
    )
    EXPECTED_BRAND_FACTS_SHA256: str = (
        "2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2"
    )
    EXPECTED_DEVELOPMENT_GOLD_SHA256: str = (
        "afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850"
    )

    # Database
    DATABASE_URL: str = "postgresql://postgres:12345@localhost:5432/kasparro_db"

    # LLM Provider credentials (loaded securely via environment, never hardcoded)
    GEMINI_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None


settings = Settings()
