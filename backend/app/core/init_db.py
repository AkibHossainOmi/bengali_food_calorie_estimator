from app.core.database import engine, Base
from app.models import user, food_prediction, food_suggestion, calorie_log

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
