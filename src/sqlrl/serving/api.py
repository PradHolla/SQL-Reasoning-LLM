"""FastAPI wrapper around ``SqlService``.

Replaces the old ``api.py``, which streamed from a vLLM server that was never
running, at a model path that never existed, and never executed a single SQL
query -- none of it worked. This one loads the same weights ``eval.run_eval``
scores and actually runs the generated SQL.

**Endpoints are sync ``def``, not ``async def``.** The model is one GPU-bound
resource shared by every request; ``async`` would not make two ``generate``
calls run any faster, it would only let two of them race for the same GPU at
once. FastAPI already runs sync endpoints in a threadpool, which serialises
naturally enough for a single-model service without inventing a queue.

**The service is built in the lifespan handler, not at import time.**
Importing this module must never construct a model -- that is what lets
``tests/test_serving.py`` import it, override the ``get_service`` dependency
with a fake, and exercise every route on a laptop with no GPU and no
checkpoint on disk.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from sqlrl.serving.service import SqlService

__all__ = ["app", "get_service"]

#: Env vars read at startup, not at import time -- see the module docstring.
DEFAULT_MODEL = "grpo-coder15"
DEFAULT_DATABASES = Path("data/spider/spider_data/test_database")

MIN_SAMPLES, MAX_SAMPLES = 1, 16
MIN_ATTEMPTS, MAX_ATTEMPTS = 1, 5


def _build_service() -> SqlService:
    model = os.environ.get("SQLRL_MODEL", DEFAULT_MODEL)
    databases = Path(os.environ.get("SQLRL_DATABASES", str(DEFAULT_DATABASES)))
    return SqlService(model, databases=databases)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.service = _build_service()
    yield


app = FastAPI(title="SQL Reasoning Service", lifespan=_lifespan)


def get_service() -> SqlService:
    """The one model instance, built at startup. Overridden with a fake in
    tests via ``app.dependency_overrides`` -- see ``tests/test_serving.py``.
    """
    return app.state.service


class QueryRequest(BaseModel):
    question: str
    db_id: str
    samples: int = Field(default=1, ge=MIN_SAMPLES, le=MAX_SAMPLES)
    max_attempts: int = Field(default=1, ge=MIN_ATTEMPTS, le=MAX_ATTEMPTS)


@app.get("/health")
def health(service: SqlService = Depends(get_service)) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": service.model_name,
        "device": service.backend.device,
        "databases": len(service.databases),
    }


@app.get("/databases")
def databases(service: SqlService = Depends(get_service)) -> list[str]:
    return service.databases


@app.post("/query")
def query(request: QueryRequest, service: SqlService = Depends(get_service)) -> dict[str, Any]:
    if request.db_id not in service.databases:
        raise HTTPException(
            status_code=404,
            detail=f"unknown db_id {request.db_id!r}; valid ids: {service.databases}",
        )
    answer = service.answer(
        request.question,
        request.db_id,
        samples=request.samples,
        max_attempts=request.max_attempts,
    )
    return asdict(answer)
