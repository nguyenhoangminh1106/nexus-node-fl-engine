FROM python:3.11-slim

WORKDIR /app

# Install system deps for asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy everything needed for install
COPY pyproject.toml README.md ./
COPY nexus/ nexus/
COPY alembic/ alembic/
COPY alembic.ini .

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "nexus.main:app", "--host", "0.0.0.0", "--port", "8000"]
