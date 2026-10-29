from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_pipeline_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1],
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "corpus" / "entities" / "company.yaml").exists():
            return candidate
    raise RuntimeError("Cannot find corpus/entities/company.yaml from the app or the working directory.")


PIPELINE_ROOT = find_pipeline_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PIPELINE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr = SecretStr("")
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    top_k: int = 5
    index_path: Path = Field(default=PIPELINE_ROOT / "data" / "chroma")
    docs_path: Path = Field(default=PIPELINE_ROOT / "corpus" / "docs")
    entities_path: Path = Field(default=PIPELINE_ROOT / "corpus" / "entities")
    table_name: str = "northline"

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_key_quotes(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().strip('"').strip("'")
        return value

    @field_validator("index_path", "docs_path", "entities_path", mode="after")
    @classmethod
    def resolve_under_repo(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (PIPELINE_ROOT / value).resolve()


settings = Settings()
