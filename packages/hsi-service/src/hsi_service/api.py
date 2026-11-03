"""REST surface (A5 §3). Run behind the company's auth gateway, which sets `principals`."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from hsi_service.service import RetrievalService


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    k: int = Field(ge=1)
    principals: list[str]


def create_app(service: RetrievalService) -> FastAPI:
    app = FastAPI(title="HSI retrieval", version="0.1.0")

    @app.post("/v1/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        try:
            return service.retrieve(req.question, req.k, req.principals)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/v1/health")
    def health() -> dict:
        return service.health()

    return app
