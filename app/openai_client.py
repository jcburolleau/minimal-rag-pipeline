from app.config import settings


def api_key() -> str:
    return settings.openai_api_key.get_secret_value()
