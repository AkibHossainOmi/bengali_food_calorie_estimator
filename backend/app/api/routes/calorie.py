from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import date
from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.calorie_log import CalorieLog
from app.models.food_prediction import FoodPrediction
from app.schemas.calorie import AddCaloriesRequest, DailyProgressResponse, FoodPredictionResponse

router = APIRouter()

# --- Helper function to format food names ---
def format_food_name(raw_name: str) -> str:
    """
    Convert raw food names like 'Meat Curry_Gosht Bhuna' or 'Boiled_egg'
    into a readable format: 'Meat curry gosht bhuna' or 'Boiled egg'
    """
    name = raw_name.replace("_", " ")
    return name[:1].upper() + name[1:].lower()

# --- Add calories endpoint ---
@router.post("/add")
def add_calories(
    payload: AddCaloriesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    calories = payload.calories
    food_name = payload.food_name  # make sure your AddCaloriesRequest includes this
    today = date.today()

    # 1️⃣ Update CalorieLog
    log = db.query(CalorieLog).filter_by(user_id=current_user.id, date=today).first()
    if log:
        log.calories += calories
    else:
        log = CalorieLog(user_id=current_user.id, calories=calories, date=today)
        db.add(log)

    # 2️⃣ Save FoodPrediction
    food_pred = FoodPrediction(
        user_id=current_user.id,
        food_name=food_name,
        calories=calories
    )
    db.add(food_pred)

    db.commit()
    return {"msg": "Calories added successfully"}

# --- Get daily progress endpoint ---
@router.get("/progress", response_model=DailyProgressResponse)
def get_daily_progress(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    today = date.today()
    log = db.query(CalorieLog).filter_by(user_id=current_user.id, date=today).first()
    consumed = log.calories if log else 0
    goal = current_user.daily_calorie_goal
    progress = min(consumed / goal * 100, 100) if goal else 0

    # Fetch recent 5 food predictions
    recent_predictions = (
        db.query(FoodPrediction)
        .filter_by(user_id=current_user.id)
        .order_by(FoodPrediction.predicted_at.desc())
        .limit(5)
        .all()
    )

    recent_foods = [
        FoodPredictionResponse(
            food_name=format_food_name(fp.food_name),  # formatted here
            calories=fp.calories,
            predicted_at=fp.predicted_at
        )
        for fp in recent_predictions
    ]

    return DailyProgressResponse(
        consumed=consumed,
        goal=goal,
        progress=progress,
        recent_foods=recent_foods
    )
