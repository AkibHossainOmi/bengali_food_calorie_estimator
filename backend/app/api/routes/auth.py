import http
import os
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
import datetime
from app.schemas.user import UserCreate 

router = APIRouter()

FRONTEND_SITE_URL = os.getenv("FRONTEND_SITE_URL")

@router.post("/register")
async def register_user(user: UserCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = hash_password(user.password)
    new_user = User(
        email=user.email,
        hashed_password=hashed_pw,
        is_active=False,
        name=user.name,
        weight=user.weight,
        height=user.height,
        age=user.age,
        gender=user.gender,
        activity_level=user.activity_level,
        daily_calorie_goal=user.daily_calorie_goal,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = generate_verification_token(new_user.email)
    verify_url = f"{FRONTEND_SITE_URL}/verify-email?token={token}"

    background_tasks.add_task(
        send_email,
        to_email=new_user.email,
        subject="Please verify your email",
        body=build_verification_email_body(verify_url),
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
        reset_url = f"{FRONTEND_SITE_URL}/reset-password?token={token}"

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

def build_verification_email_body(verify_url: str) -> str:
    return f"""
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: 'Arial', sans-serif; background-color: #f8fafc; margin: 0; padding: 0;">
            <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
            <tr>
                <td align="center" style="padding: 40px 20px;">
                <table width="100%" max-width="600" cellpadding="0" cellspacing="0" role="presentation" style="background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); overflow: hidden; border: 1px solid #e2e8f0;">
                    <!-- Header -->
                    <tr>
                    <td style="padding: 32px 40px 24px; text-align: center; background-color: #f0fdf4; border-bottom: 1px solid #dcfce7;">
                        <h1 style="color: #047857; margin: 0; font-size: 24px; font-weight: 600;">Verify Your Email Address</h1>
                    </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                    <td style="padding: 32px 40px; text-align: center;">
                        <p style="color: #334155; font-size: 16px; line-height: 1.5; margin: 0 0 24px;">
                        Thank you for registering with <strong>Bengali Food Calorie Estimator</strong>! Please confirm your email address to complete your account setup.
                        </p>
                        
                        <a href="{verify_url}" 
                        style="
                            display: inline-block; 
                            padding: 12px 24px; 
                            background-color: #059669; 
                            color: white; 
                            text-decoration: none; 
                            border-radius: 8px; 
                            font-weight: 600;
                            font-size: 16px;
                            transition: background-color 0.2s;
                        "
                        onmouseover="this.style.backgroundColor='#047857'"
                        onmouseout="this.style.backgroundColor='#059669'"
                        >
                        Verify Email Address
                        </a>
                        
                        <p style="color: #64748b; font-size: 14px; line-height: 1.5; margin: 32px 0 0;">
                        If you did not create an account with us, please ignore this email.
                        </p>
                    </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                    <td style="padding: 24px 40px; text-align: center; background-color: #f8fafc; border-top: 1px solid #e2e8f0;">
                        <p style="color: #94a3b8; font-size: 12px; line-height: 1.5; margin: 0;">
                        © {datetime.datetime.now().year} Bengali Food Calorie Estimator. All rights reserved.<br>
                        </p>
                    </td>
                    </tr>
                </table>
                </td>
            </tr>
            </table>
        </body>
        </html>
        """
