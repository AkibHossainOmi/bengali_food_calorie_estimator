from fastapi import APIRouter
from app.api.routes.predict import router as predict_router
from app.api.routes.auth import router as auth_router

router = APIRouter()

router.include_router(predict_router, prefix="/api")
router.include_router(auth_router, prefix="/api/auth")
