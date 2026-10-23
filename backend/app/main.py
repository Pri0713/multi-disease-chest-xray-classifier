from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.predict import router as predict_router
from backend.app.api.gradcam import router as gradcam_router


app = FastAPI(
    title="Multi-Disease Chest X-ray Classifier API",
    description="Backend API for chest X-ray disease prediction and Grad-CAM visualization.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(predict_router)
app.include_router(gradcam_router)


@app.get("/")
def root():
    return {"message": "Chest X-ray Classifier API is running"}