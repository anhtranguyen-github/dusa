from fastapi import FastAPI

from src.api.routers import health, kie, models

app = FastAPI(
    title="dusa — Document Understanding API",
    description="Dual-backend KIE on SROIE: LayoutLMv3 (Triton) + Qwen2.5-3B (vLLM).",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(kie.router)
