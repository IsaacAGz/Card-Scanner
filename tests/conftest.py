"""Shared fixtures. API tests skip model loading."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    @asynccontextmanager
    async def _skip_model_load(_app):
        yield

    original = main.app.router.lifespan_context
    main.app.router.lifespan_context = _skip_model_load
    try:
        with TestClient(main.app) as test_client:
            yield test_client
    finally:
        main.app.router.lifespan_context = original
