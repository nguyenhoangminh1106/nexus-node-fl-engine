"""Run Alembic migrations programmatically at application startup."""

import subprocess
import sys
from pathlib import Path

from nexus.config import settings


def run_migrations() -> None:
    """Apply all pending Alembic migrations via subprocess.

    We shell out instead of using alembic.command directly because the async
    Alembic env calls asyncio.run(), which cannot be nested inside uvicorn's
    already-running event loop.
    """
    alembic_ini = str(Path(__file__).resolve().parents[2] / "alembic.ini")

    # Convert async URL to sync for Alembic CLI (asyncpg -> psycopg2-style).
    db_url = settings.async_database_url
    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", alembic_ini, "upgrade", "head"],
        env={**__import__("os").environ, "DATABASE_URL": sync_url},
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Alembic migration failed:\n{result.stderr}")
