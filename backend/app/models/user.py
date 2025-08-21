from sqlalchemy import Boolean, Column, Integer, String, Float
from sqlalchemy.orm import relationship
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=False)

 
    # BMI & calorie tracking
    name = Column(String(255), nullable=True)
    weight = Column(Float, nullable=True)
    height = Column(Float, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    activity_level = Column(String(20), nullable=True, default="sedentary") 
    daily_calorie_goal = Column(Integer, nullable=True)
    
    # Relationships
    calorie_logs = relationship("CalorieLog", back_populates="user", cascade="all, delete-orphan")
    food_suggestions = relationship("FoodSuggestion", back_populates="user", cascade="all, delete-orphan")
    food_predictions = relationship("FoodPrediction", back_populates="user", cascade="all, delete-orphan")
