from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from backend.app.services.gradcam_service import gradcam_service


router = APIRouter()


@router.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    suffix = Path(file.filename or "").suffix

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        class_name, image_bytes = gradcam_service.generate(temp_path)

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "X-GradCAM-Class": class_name,
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {str(exc)}",
        )

    finally:
        if "temp_path" in locals():
            Path(temp_path).unlink(missing_ok=True)