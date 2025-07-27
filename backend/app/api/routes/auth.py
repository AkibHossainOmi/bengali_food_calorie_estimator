from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User
from app.core.security import hash_password
from app.core.tokens import generate_verification_token, confirm_verification_token, generate_password_reset_token, confirm_password_reset_token
from app.core.email import send_email
from fastapi import Query
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from app.core.security import verify_password
from app.core.jwt import create_access_token
from fastapi import BackgroundTasks

router = APIRouter()

class UserCreate(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
async def register_user(user: UserCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = hash_password(user.password)
    new_user = User(email=user.email, hashed_password=hashed_pw, is_active=False)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = generate_verification_token(new_user.email)
    verify_url = f"http://localhost:3000/verify-email?token={token}"  # Adjust frontend URL as needed

    background_tasks.add_task(
        send_email,
        to_email=new_user.email,
        subject="Please verify your email",
        body=f"Click to verify your account: {verify_url}"
    )

    return {"msg": "Registration successful, please check your email to verify your account"}

@router.get("/verify-email")
async def verify_email(token: str = Query(...), db: Session = Depends(get_db)):
    email = confirm_verification_token(token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.is_active:
        return {"msg": "Account already verified"}

    user.is_active = True
    db.commit()
    return {"msg": "Email verified successfully, you can now login"}

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Email not verified")

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

class PasswordResetRequest(BaseModel):
    email: EmailStr

@router.post("/password-reset-request")
async def password_reset_request(data: PasswordResetRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if user:
        token = generate_password_reset_token(user.email)
        reset_url = f"http://localhost:3000/reset-password?token={token}"

        background_tasks.add_task(
            send_email,
            to_email=user.email,
            subject="Password Reset Request",
            body=f"Reset your password by clicking here: {reset_url}"
        )
    # Always return success to avoid revealing if email exists
    return {"msg": "If your email is registered, you will receive a password reset email shortly."}

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

@router.post("/password-reset-confirm")
async def password_reset_confirm(data: PasswordResetConfirm, db: Session = Depends(get_db)):
    email = confirm_password_reset_token(data.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.hashed_password = hash_password(data.new_password)
    db.commit()
    return {"msg": "Password reset successful"}
