from pydantic_settings import BaseSettings
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')
print(ENV_PATH)

class Settings(BaseSettings):
    openai_api_key: str
    google_api_key: str
    nvidia_nim_endpoint: str
    nvidia_nim_api_key: str
    validator_api_key: str = os.getenv("VALIDATOR_API_KEY", "your-secret-key")
    vlm_url: str = os.getenv("VLM_URL", "https://qwen3-vl-service-bqvwxg2cvq-uk.a.run.app/v1/chat/completions")
    vlm_api_key: str = os.getenv("VLM_API_KEY", "")  # optional, can also use VALIDATOR_API_KEY

    

    class Config:
        env_file = '.env'

settings = Settings()