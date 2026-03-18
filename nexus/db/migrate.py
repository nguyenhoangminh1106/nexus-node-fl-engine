"""Run Alembic migrations programmatically at application startup."""

import os
from pathlib import Path

from alembic import command
from alembic.config import Config

from nexus.config import settings


def run_migrations() -> None:
    """Apply all pending Alembic migrations."""
    alembic_cfg = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))

    # Override the DB URL with the app's configured value so the migration
    # always targets the same database the application connects to.
    db_url = settings.async_database_url
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(alembic_cfg, "head")
