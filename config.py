from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    LANGSMITH_TRACING: bool = Field(False, env="LANGSMITH_TRACING")
    LANGSMITH_API_KEY: str = Field(..., env="LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = Field(..., env="LANGSMITH_PROJECT")
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    PINECONE_INDEX: str = Field(..., env="PINECONE_INDEX")
    COHERE_API_KEY: str = Field(..., env="COHERE_API_KEY")
    BM25_ENCODER_FILE: str = Field(..., env="BM25_ENCODER_FILE")
    S3_BUCKET_NAME: str = Field(..., env="S3_BUCKET_NAME")
    AWS_ACCESS_KEY_ID: str = Field(..., env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    AWS_URL: str = Field(..., env="AWS_URL")
    LOCAL_BM25_PATH: str = Field(..., env="LOCAL_BM25_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"


settings = Settings()