from sqlalchemy import Column, Integer, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class FoodPrediction(Base):
    __tablename__ = "food_predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    food_name = Column(String(255), nullable=False)
    calories = Column(Integer, nullable=False)
    image_url = Column(String(255), nullable=True)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="food_predictions")
