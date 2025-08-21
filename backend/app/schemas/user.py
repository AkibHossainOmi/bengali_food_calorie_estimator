# app/schemas/user.py
from pydantic import BaseModel, EmailStr
from typing import Optional

class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    weight: Optional[float]
    height: Optional[float]
    activity_level: Optional[str]
    daily_calorie_goal: Optional[int]

    class Config:
        from_attributes = True

class UserUpdateRequest(BaseModel):
    name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    weight: Optional[float]
    height: Optional[float]

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    activity_level: Optional[str] = "sedentary"
    daily_calorie_goal: Optional[int] = None
