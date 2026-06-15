from fastapi import APIRouter

from src.api.schemas import ModelStatus

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/status", response_model=list[ModelStatus])
async def status() -> list[ModelStatus]:
    # TODO (S8): probe Triton + vLLM and return real readiness.
    return [
        ModelStatus(name="layoutlmv3", backend="triton", ready=False),
        ModelStatus(name="qwen2.5-3b", backend="vllm", ready=False),
    ]
