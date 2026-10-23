from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.services.prediction_service import prediction_service


router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
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

        predictions = prediction_service.predict(temp_path)

        return {
            "filename": file.filename,
            "predictions": predictions,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )

    finally:
        if "temp_path" in locals():
            Path(temp_path).unlink(missing_ok=True)