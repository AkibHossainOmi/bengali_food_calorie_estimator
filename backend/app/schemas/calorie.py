from pydantic import BaseModel
from typing import List
from datetime import datetime

class AddCaloriesRequest(BaseModel):
    calories: int
    food_name: str

class FoodPredictionResponse(BaseModel):
    food_name: str
    calories: int
    predicted_at: datetime

class DailyProgressResponse(BaseModel):
    consumed: int
    goal: int
    progress: float
    recent_foods: List[FoodPredictionResponse] = []
