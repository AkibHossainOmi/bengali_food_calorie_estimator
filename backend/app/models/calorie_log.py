# app/models/calorie_log.py
from sqlalchemy import Column, Integer, ForeignKey, Date
from sqlalchemy.orm import relationship
from app.core.database import Base

class CalorieLog(Base):
    __tablename__ = "calorie_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date, nullable=False)
    calories = Column(Integer, nullable=False)

    user = relationship("User", back_populates="calorie_logs")
