from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.schemas.user import UserResponse, UserUpdateRequest

router = APIRouter(
    prefix="/profile",
    tags=["Profile"]
)

# Mapping of activity levels to factors
ACTIVITY_FACTORS = {
    "sedentary": 1.2,            # little or no exercise
    "lightly_active": 1.375,     # light exercise 1–3 days/week
    "moderately_active": 1.55,   # moderate exercise 3–5 days/week
    "very_active": 1.725,        # hard exercise 6–7 days/week
    "extra_active": 1.9          # very hard exercise or physical job
}

def calculate_daily_calorie_goal(
    weight: float,
    height: float,
    age: int,
    gender: str,
    activity_level: str = "sedentary"
) -> int:
    """Calculate daily calorie goal using BMR × activity factor."""
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factor = ACTIVITY_FACTORS.get(activity_level, 1.2)  # default sedentary
    return int(bmr * activity_factor)

def user_to_response(user: User) -> UserResponse:
    """Convert SQLAlchemy User model to UserResponse dict."""
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name or "",
        age=user.age,
        gender=user.gender or "",
        weight=user.weight,
        height=user.height,
        activity_level=user.activity_level or "sedentary",
        daily_calorie_goal=user.daily_calorie_goal
    )

@router.get("/", response_model=UserResponse)
def get_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return user_to_response(current_user)

@router.put("/update", response_model=UserResponse)
def update_profile(
    payload: UserUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Update provided fields
    for field in ["name", "age", "gender", "weight", "height", "activity_level"]:
        value = getattr(payload, field, None)
        if value is not None:
            setattr(current_user, field, value)

    # Recalculate daily calorie goal if all required fields are present
    if all([
        current_user.weight,
        current_user.height,
        current_user.age,
        current_user.gender,
        current_user.activity_level
    ]):
        current_user.daily_calorie_goal = calculate_daily_calorie_goal(
            weight=current_user.weight,
            height=current_user.height,
            age=current_user.age,
            gender=current_user.gender,
            activity_level=current_user.activity_level
        )

    db.commit()
    db.refresh(current_user)
    return user_to_response(current_user)
