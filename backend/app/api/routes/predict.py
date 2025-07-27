from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from app.services.predictor import predict_image_from_bytes
from app.schemas.prediction import PredictionResponse

router = APIRouter()

@router.get("/")
def health_check():
    return {"message": "Bengali Food Calorie Estimator API"}

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_bytes = BytesIO(contents)
        food, calories, confidence = predict_image_from_bytes(image_bytes)
        return PredictionResponse(
            food=food,
            estimated_calories=calories,
            confidence=round(confidence * 100, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
