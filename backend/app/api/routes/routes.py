from fastapi import APIRouter
from app.api.routes.predict import router as predict_router
from app.api.routes.auth import router as auth_router
from app.api.routes.calorie import router as calorie_router
from app.api.routes.profile import router as profile_router

router = APIRouter()

router.include_router(predict_router, prefix="/api")
router.include_router(auth_router, prefix="/api/auth")
router.include_router(calorie_router, prefix="/api/calorie")
router.include_router(profile_router, prefix="/api")
