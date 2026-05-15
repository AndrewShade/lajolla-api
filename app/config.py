from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_dir: Path = Path(__file__).parent.parent / "models"
    go_threshold: float = 0.5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
