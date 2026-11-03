from pathlib import Path
from typing import Literal

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
    preprocessed_path: Path = Field(default=PIPELINE_ROOT / "data" / "preprocessed")
    table_name: str = "northline"
    chunker: Literal["markdown", "llm"] = "markdown"
    preprocess_model: str = ""
    rewrite_query: bool = False
    expand_query: bool = False
    expand_n: int = 3
    rerank: bool = False
    candidate_k: int = 20

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_key_quotes(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().strip('"').strip("'")
        return value

    @field_validator("index_path", "docs_path", "entities_path", "preprocessed_path", mode="after")
    @classmethod
    def resolve_under_repo(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (PIPELINE_ROOT / value).resolve()

    @property
    def collection_name(self) -> str:
        """Chroma collection the app reads: one per chunker, so both indexes coexist."""
        return collection_for(self.table_name, self.chunker)


def collection_for(table_name: str, chunker: str) -> str:
    return table_name if chunker == "markdown" else f"{table_name}_{chunker}"


settings = Settings()
