from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from io import BytesIO
from app.services.predictor import predict_image_from_bytes
from app.schemas.prediction import PredictionResponse

router = APIRouter()

@router.get("/")
def health_check():
    return {"message": "Bengali Food Calorie Estimator API"}

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    amount_in_grams: float = Form(...)
):
    """
    Predict food type and calories from image and calculate final calories based on user-input amount.
    """
    if amount_in_grams <= 0:
        raise HTTPException(status_code=400, detail="Amount in grams must be greater than 0")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        image_bytes = BytesIO(contents)
        food, calories_per_100g, confidence = predict_image_from_bytes(image_bytes)

        # Calculate calories for the specified amount
        final_calories = (calories_per_100g / 100) * amount_in_grams

        return PredictionResponse(
            food=food,
            estimated_calories=round(final_calories, 2),
            confidence=round(confidence * 100, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
