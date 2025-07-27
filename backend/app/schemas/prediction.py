from pydantic import BaseModel

class PredictionResponse(BaseModel):
    food: str
    estimated_calories: float
    confidence: float
