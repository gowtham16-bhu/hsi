from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def mini_musique():
    from hsi_harness.datasets import load_musique
    return load_musique(str(FIXTURES / "musique_mini.jsonl"), None, 0)


@pytest.fixture
def pg_dsn() -> str:
    dsn = os.environ.get("HSI_PG_DSN")
    if not dsn:
        pytest.skip("HSI_PG_DSN not set; start PostgreSQL with scripts/dev-postgres.sh")
    return dsn
