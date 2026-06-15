from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class KIEBackend(str, Enum):
    layoutlmv3 = "layoutlmv3"
    qwen = "qwen"


class KIEFields(BaseModel):
    company: Optional[str] = None
    date: Optional[str] = None
    address: Optional[str] = None
    total: Optional[str] = None


class KIEResponse(BaseModel):
    backend: KIEBackend
    fields: KIEFields
    latency_ms: float = Field(..., description="End-to-end latency including OCR + KIE")


class ModelStatus(BaseModel):
    name: str
    backend: str
    ready: bool
    version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
