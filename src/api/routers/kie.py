from fastapi import APIRouter, File, UploadFile

from src.api.schemas import KIEBackend, KIEFields, KIEResponse

router = APIRouter(tags=["kie"])


@router.post("/kie", response_model=KIEResponse)
async def extract(
    backend: KIEBackend = KIEBackend.layoutlmv3,
    image: UploadFile = File(...),
) -> KIEResponse:
    # TODO (S8): wire pipeline → OCR (PARSeq via Triton) → KIE backend dispatch.
    _ = await image.read()
    return KIEResponse(
        backend=backend,
        fields=KIEFields(),
        latency_ms=0.0,
    )
