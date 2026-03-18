"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://nexus:nexus@localhost:5432/nexus"

    @property
    def async_database_url(self) -> str:
        """Ensure the URL uses the asyncpg driver.

        Railway (and most providers) give postgresql:// URLs.
        SQLAlchemy async requires postgresql+asyncpg://.
        """
        url = self.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    # S3 / MinIO
    s3_endpoint_url: str = "http://localhost:9000"
    s3_public_url: str = ""  # Public URL for presigned downloads (e.g. https://minio.example.com). If empty, downloads are proxied through the API.
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "nexus"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Seed
    seed_org_name: str = "nexus-admin"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
