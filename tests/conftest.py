from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import psycopg
import pytest
import pytest_asyncio

import sys

from config import settings


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def mock_openai():
    client = SimpleNamespace()
    client.chat = SimpleNamespace()
    client.chat.completions = SimpleNamespace()
    client.chat.completions.create = AsyncMock()
    client.embeddings = SimpleNamespace()
    client.embeddings.create = AsyncMock()
    return client


@pytest.fixture
def settings_stub():
    return SimpleNamespace(
        chat_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        summary_refresh_every_k=2,
    )


def unit_vector(seed: int = 1, dim: int = 1536) -> list[float]:
    values = [((seed * (i + 7)) % 997) / 997.0 for i in range(dim)]
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


@pytest_asyncio.fixture
async def db_conn():
    try:
        conn = await psycopg.AsyncConnection.connect(settings.database_url)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Database unavailable for integration-style tests: {exc}")
    try:
        yield conn
    finally:
        await conn.close()
